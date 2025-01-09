import os
import json
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO
import signal
import traceback

import httpx
from pydub import AudioSegment

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError, BadRequest

#######################################
# Environment Variables
#######################################
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", "")
ELEVEN_LABS_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID", "0whGLe6wyQ2fwT9M40ZY")  # Ensure this is set correctly
MAX_MESSAGES_PER_USER = int(os.getenv("MAX_MESSAGES_PER_USER", "15"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "45"))  # Cooldown duration

#######################################
# Logging Setup
#######################################
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Change to DEBUG for more detailed logs
)
logger = logging.getLogger(__name__)

#######################################
# Rate Limit: 15 messages / 24h
#######################################
USER_MESSAGE_LIMITS = defaultdict(lambda: {
    "count": 0,
    "reset_time": datetime.utcnow() + timedelta(hours=24),
    "last_message_time": None  # Initialize with None for cooldown tracking
})

#######################################
# Check ffmpeg Availability
#######################################
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("ffmpeg is installed and accessible.")
    except Exception as e:
        logger.error("ffmpeg is not installed or not accessible.")
        raise e

#######################################
# Convert MP3 -> OGG
#######################################
def convert_mp3_to_ogg(mp3_data: bytes) -> BytesIO:
    """
    Convert MP3 bytes to OGG (Opus) for Telegram voice notes.
    """
    try:
        mp3_file = BytesIO(mp3_data)
        segment = AudioSegment.from_file(mp3_file, format="mp3")
        ogg_buffer = BytesIO()
        segment.export(
            ogg_buffer,
            format="ogg",
            codec="libopus",
            bitrate="64k",
            parameters=["-vbr", "on"]  # Enable Variable Bitrate for better quality
        )
        ogg_buffer.seek(0)
        logger.info("MP3 successfully converted to OGG.")
        return ogg_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        logger.debug(traceback.format_exc())
        return BytesIO()

#######################################
# ElevenLabs TTS
#######################################
async def elevenlabs_tts(text: str) -> bytes:
    """
    Calls ElevenLabs TTS endpoint asynchronously, returning MP3 audio bytes.
    """
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",  # Ensure this is a valid model_id
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }
    
    # Log the Voice ID and model_id being used
    logger.info(f"Using ElevenLabs Voice ID: {ELEVEN_LABS_VOICE_ID}")
    logger.info(f"Using model_id: {payload['model_id']}")
    
    async with httpx.AsyncClient() as client:
        try:
            logger.info("Sending request to ElevenLabs TTS API.")
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}",
                headers=headers,
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            logger.info("Received response from ElevenLabs TTS API.")
            return resp.content  # raw MP3
        except httpx.HTTPStatusError as e:
            # Log the response content for detailed error
            logger.error(f"HTTP Status Error: {e.response.status_code} - {e.response.text}")
            return b""
        except Exception as e:
            logger.error(f"Error calling ElevenLabs TTS: {e}")
            logger.debug(traceback.format_exc())
            return b""

#######################################
# OpenAI Chat Completion
#######################################
async def generate_openai_response(user_text: str, persona: str) -> str:
    """
    Generates a response from OpenAI's Chat Completion API.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": persona},  # Changed role from 'developer' to 'system'
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.8,  # Adjust as needed
        "max_tokens": 1024,  # Set a reasonable limit
        "n": 1,
        "stop": None
    }
    async with httpx.AsyncClient() as client:
        try:
            logger.info("Sending request to OpenAI Chat Completion API.")
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"OpenAI Response Data: {data}")
            reply = data['choices'][0]['message']['content'].strip()
            logger.info("Received response from OpenAI.")
            return reply
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API returned an error: {e.response.status_code} - {e.response.text}")
            logger.debug(traceback.format_exc())
            return "❌ Sorry, I couldn't process your request at the moment."
        except Exception as e:
            logger.error(f"Error communicating with OpenAI API: {e}")
            logger.debug(traceback.format_exc())
            return "❌ An unexpected error occurred while processing your request."

#######################################
# Telegram Handlers
#######################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start:
    - Reset daily usage and cooldown
    - Send welcome message
    """
    user_id = update.effective_user.id

    # Reset message count, reset time, and cooldown
    USER_MESSAGE_LIMITS[user_id]["count"] = 0
    USER_MESSAGE_LIMITS[user_id]["reset_time"] = datetime.utcnow() + timedelta(hours=24)
    USER_MESSAGE_LIMITS[user_id]["last_message_time"] = None  # Reset cooldown

    # Simplified and concise persona
    kasper_persona = (
        "You are KASPER, the friendly assistant. "
        "Your goal is to help and inform users based on their queries. "
        "Provide clear, concise, and accurate responses. "
        "Maintain a polite and engaging tone."
    )

    # Store persona in user context
    context.user_data['persona'] = kasper_persona

    await update.message.reply_text(
        "👻 **KASPER is here!** 👻\n\nA fresh conversation has started. You have 15 daily messages. Let's chat! 💬",
        parse_mode="Markdown"
    )
    logger.info(f"User {user_id} started a new session.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles incoming text messages:
    1. Enforce rate-limit (15 / 24h)
    2. Enforce 45-second cooldown between messages
    3. Generate response using OpenAI
    4. TTS with ElevenLabs
    5. Convert & send audio
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    rate_info = USER_MESSAGE_LIMITS[user_id]
    current_time = datetime.utcnow()

    # Check if reset time has passed
    if current_time >= rate_info["reset_time"]:
        rate_info["count"] = 0
        rate_info["reset_time"] = current_time + timedelta(hours=24)
        rate_info["last_message_time"] = None  # Reset cooldown
        logger.info(f"User {user_id} rate limit reset.")

    # Check for cooldown
    last_msg_time = rate_info.get("last_message_time")
    if last_msg_time:
        elapsed_time = (current_time - last_msg_time).total_seconds()
        if elapsed_time < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - elapsed_time)
            await update.message.reply_text(
                f"⏳ Please wait {remaining_time} more seconds before sending another message."
            )
            logger.info(f"User {user_id} is on cooldown. {remaining_time} seconds remaining.")
            return

    # Check if user has exceeded daily message limit
    if rate_info["count"] >= MAX_MESSAGES_PER_USER:
        await update.message.reply_text(
            f"⛔ You have reached the limit of {MAX_MESSAGES_PER_USER} messages for today. Please try again tomorrow."
        )
        logger.info(f"User {user_id} has exceeded the daily message limit.")
        return

    # Increment message count and set last_message_time
    rate_info["count"] += 1
    rate_info["last_message_time"] = current_time
    remaining = MAX_MESSAGES_PER_USER - rate_info["count"]
    logger.info(f"User {user_id} sent message #{rate_info['count']} of {MAX_MESSAGES_PER_USER}.")

    # Retrieve persona
    persona = context.user_data.get('persona', "You are a helpful assistant.")

    try:
        # Inform the user that the bot is processing their request
        processing_msg = await update.message.reply_text("👻 **KASPER is typing...** 👻", parse_mode="Markdown")

        # Generate response using OpenAI
        gpt_reply = await generate_openai_response(user_text, persona)

        # Handle empty responses
        if not gpt_reply:
            gpt_reply = "❓ Oops, KASPER couldn't come up with anything. (Ghostly shrug.) 🤷‍♂️"

        logger.info(f"GPT Reply for user {user_id}: {gpt_reply}")

        # TTS with ElevenLabs
        mp3_data = await elevenlabs_tts(gpt_reply)
        if not mp3_data:
            await processing_msg.edit_text("❌ Sorry, I couldn't process your request.")
            return
        logger.info(f"Received MP3 data from ElevenLabs for user {user_id}.")

        # Convert MP3 to OGG
        ogg_file = convert_mp3_to_ogg(mp3_data)
        ogg_buffer = ogg_file.getvalue()
        if not ogg_buffer:
            logger.error(f"Audio conversion failed for user {user_id}.")
            await processing_msg.edit_text("❌ Failed to convert audio. Please try again.")
            return
        logger.info(f"Successfully converted MP3 to OGG for user {user_id}.")

        # Send voice message
        ogg_bytes = BytesIO(ogg_buffer)
        ogg_bytes.name = "voice.ogg"  # Telegram requires a filename
        ogg_bytes.seek(0)  # Reset buffer position

        try:
            await update.message.reply_voice(voice=ogg_bytes)
            logger.info(f"Sent voice message to user {user_id}.")
            await processing_msg.delete()  # Remove the "KASPER is typing..." message
        except BadRequest as e:
            if "Voice_messages_forbidden" in str(e):
                logger.error(f"Voice messages are forbidden for user {user_id}.")
                await update.message.reply_text(
                    "❌ I can't send voice messages to you. Please check your Telegram settings or try sending a different type of message."
                )
            else:
                logger.error(f"BadRequest error for user {user_id}: {e}")
                logger.debug(traceback.format_exc())
                await update.message.reply_text("❌ An error occurred while sending the voice message. Please try again later.")
        except TelegramError as e:
            logger.error(f"TelegramError for user {user_id}: {e}")
            logger.debug(traceback.format_exc())
            await update.message.reply_text("❌ An unexpected error occurred while sending the voice message. Please try again later.")
        except Exception as e:
            logger.error(f"Unhandled exception while sending voice message for user {user_id}: {e}")
            logger.debug(traceback.format_exc())
            await update.message.reply_text("❌ An error occurred while sending the voice message. Please try again later.")

        # Inform the user about remaining messages
        if remaining > 0:
            await update.message.reply_text(f"🕸️ You have **{remaining}** messages left today.", parse_mode="Markdown")
            logger.info(f"User {user_id} has {remaining} messages left today.")
        else:
            await update.message.reply_text("⛔ You have no messages left for today. Please try again tomorrow.")
            logger.info(f"User {user_id} has no messages left for today.")

    #######################################
    # Graceful Shutdown Handler
    #######################################
    async def shutdown(application):
        """
        Gracefully shuts down the application.
        """
        logger.info("Shutting down gracefully...")
        # Stop the application (it will stop receiving new updates)
        await application.stop()
        # Perform any additional cleanup if necessary
        logger.info("Application has been stopped gracefully.")

    #######################################
    # Main Function
    #######################################
    def main():
        try:
            check_ffmpeg()
        except Exception as e:
            logger.critical("ffmpeg is not available. Exiting.")
            return

        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

        logger.info("👻 KASPER Telegram Bot: OpenAI Chat Completion + ElevenLabs TTS + 15/day limit started. 👻")

        # Register shutdown signals
        loop = asyncio.get_event_loop()

        for signame in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(getattr(signal, signame),
                                    lambda signame=signame: asyncio.create_task(shutdown(application)))

        # Run the bot
        try:
            application.run_polling()
        except Exception as e:
            logger.error(f"Application encountered an error: {e}")
            logger.debug(traceback.format_exc())

    if __name__ == "__main__":
        main()
