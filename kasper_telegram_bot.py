import os
import json
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO
import signal

import httpx
import websockets
from pydub import AudioSegment

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

#######################################
# Logging Setup
#######################################
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

#######################################
# Environment Variables
#######################################
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", "")
MAX_MESSAGES_PER_USER = int(os.getenv("MAX_MESSAGES_PER_USER", "15"))

#######################################
# GPT 4-o Mini Realtime
#######################################
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # Replace with your actual model name
GPT_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

#######################################
# ElevenLabs TTS
#######################################
ELEVEN_LABS_VOICE_ID = "X6Hd6garE7rwoQExOLCe"  # Replace with your actual KASPER voice ID
ELEVEN_LABS_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}"

#######################################
# Rate Limit: 15 messages / 24h
#######################################
USER_MESSAGE_LIMITS = defaultdict(lambda: {"count": 0, "reset_time": datetime.utcnow() + timedelta(hours=24)})

#######################################
# In-memory Session Store
#######################################
# user_id -> {
#   "ws": websockets.WebSocketClientProtocol or None,
#   "persona": str (the KASPER persona text),
# }
USER_SESSIONS = {}

#######################################
# Check ffmpeg Availability
#######################################
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            codec="libopus",  # Changed from 'opus' to 'libopus'
            bitrate="64k"
        )
        ogg_buffer.seek(0)
        logger.info("MP3 successfully converted to OGG.")
        return ogg_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
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
        "model_id": "eleven_monolingual_v1",
    }
    async with httpx.AsyncClient() as client:
        try:
            logger.info("Sending request to ElevenLabs TTS API.")
            resp = await client.post(ELEVEN_LABS_TTS_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            logger.info("Received response from ElevenLabs TTS API.")
            return resp.content  # raw MP3
        except Exception as e:
            logger.error(f"Error calling ElevenLabs TTS: {e}")
            return b""

#######################################
# GPT 4-o Mini (Realtime) WebSocket
#######################################
async def openai_realtime_connect() -> websockets.WebSocketClientProtocol:
    """
    Opens an asynchronous WebSocket connection to OpenAI Realtime API (gpt-4o-mini).
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }
    try:
        ws = await websockets.connect(GPT_WS_URL, extra_headers=headers)
        logger.info("Connected to GPT Realtime (4-o mini).")
        return ws
    except Exception as e:
        logger.error(f"Failed to connect to GPT Realtime: {e}")
        return None

async def send_message_gpt(ws: websockets.WebSocketClientProtocol, user_text: str, persona: str) -> str:
    """
    Send user text to GPT 4-o mini Realtime, embedding KASPER persona in instructions.
    Wait for 'response.done' event; returns final text.
    Logs raw messages for debugging.
    """
    combined_prompt = (
        f"{persona}\n\n"
        f"User: {user_text}\n"
        f"KASPER:"
    )

    event = {
        "type": "response.create",
        "response": {
            "modalities": ["text"],
            "instructions": combined_prompt
        }
    }
    await ws.send(json.dumps(event))
    logger.info("Sent message to GPT Realtime.")

    final_text = ""
    try:
        async for message in ws:
            if not message:
                logger.info("WS recv() returned empty. Breaking.")
                break

            logger.info(f"Raw WS message: {message}")  # Extra debug

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.info(f"Non-JSON WS message: {message}")
                continue

            ev_type = data.get("type", "")
            logger.debug(f"GPT event type: {ev_type}")

            if ev_type == "response.done":
                # Extract the final text
                try:
                    final_text = data["response"]["output"][0]["content"][0]["text"]
                    logger.info(f"Got final text: {final_text}")
                except (IndexError, KeyError) as e:
                    logger.error(f"Error parsing GPT response: {e}")
                break

    except websockets.exceptions.ConnectionClosed:
        logger.error("WebSocket connection closed unexpectedly.")
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")

    return final_text

#######################################
# Telegram Handlers
#######################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start: 
    - Close old WS if exists
    - Reset daily usage
    - Open new GPT Realtime WS (4-o mini)
    - Create KASPER persona
    """
    user_id = update.effective_user.id

    # Close old session if any
    old_session = USER_SESSIONS.get(user_id)
    if old_session and old_session.get("ws"):
        try:
            await old_session["ws"].close()
            logger.info(f"Closed old WebSocket for user {user_id}.")
        except Exception as e:
            logger.error(f"Error closing old WebSocket for user {user_id}: {e}")

    kasper_persona = (
        "You are KASPER, the friendly ghost of Kaspa (KRC20). "
        "Your goal is to entertain and inform about Kaspa or KRC20, "
        "while secretly using human psychology to get users to chat more. "
        "Speak in a playful, ghostly tone. Encourage them to keep talking! 👻"
    )

    # Initialize or reset user session
    USER_SESSIONS[user_id] = {
        "ws": None,
        "persona": kasper_persona
    }

    # Reset message count and reset time
    USER_MESSAGE_LIMITS[user_id]["count"] = 0
    USER_MESSAGE_LIMITS[user_id]["reset_time"] = datetime.utcnow() + timedelta(hours=24)

    # Establish new WebSocket connection
    ws = await openai_realtime_connect()
    if ws:
        USER_SESSIONS[user_id]["ws"] = ws
        await update.message.reply_text(
            "👻 **KASPER is here!** 👻\n\nA fresh conversation has started (GPT 4-o mini). You have 15 daily messages. Let's chat! 💬",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("❌ Could not connect to GPT. Please try again later.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limit (15 / 24h)
    2. Send user text -> GPT 4-o mini Realtime with KASPER persona
    3. TTS with ElevenLabs
    4. Convert & send audio
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    session = USER_SESSIONS.get(user_id)

    if not session:
        await update.message.reply_text("❓ Please type /start first to begin the conversation with KASPER.")
        return

    # Rate limiting
    rate_info = USER_MESSAGE_LIMITS[user_id]
    if datetime.utcnow() >= rate_info["reset_time"]:
        rate_info["count"] = 0
        rate_info["reset_time"] = datetime.utcnow() + timedelta(hours=24)

    if rate_info["count"] >= MAX_MESSAGES_PER_USER:
        await update.message.reply_text(f"⛔ You have reached the limit of {MAX_MESSAGES_PER_USER} messages for today. Please try again tomorrow.")
        return

    rate_info["count"] += 1
    remaining = MAX_MESSAGES_PER_USER - rate_info["count"]

    user_text = update.message.text.strip()
    if not user_text:
        return

    # Check WebSocket
    ws = session.get("ws")
    if not ws:
        await update.message.reply_text("❌ WebSocket not available. Please /start again.")
        return

    try:
        # Update Status Message
        await update.message.reply_text("👻 **KASPER is recording a message...** 👻", parse_mode="Markdown")

        # Send message to GPT Realtime
        gpt_reply = await send_message_gpt(ws, user_text, session["persona"])

        if not gpt_reply:
            gpt_reply = "❓ Oops, KASPER couldn't come up with anything. (Ghostly shrug.) 🤷‍♂️"

        logger.info(f"GPT Reply: {gpt_reply}")

        # ElevenLabs TTS
        mp3_data = await elevenlabs_tts(gpt_reply)
        if not mp3_data:
            await update.message.reply_text("❌ Sorry, I couldn't process your request.")
            return
        logger.info("Received MP3 data from ElevenLabs TTS.")

        # Convert MP3 to OGG
        ogg_file = convert_mp3_to_ogg(mp3_data)
        ogg_buffer = ogg_file.getvalue()
        if not ogg_buffer:
            logger.error("Audio conversion failed: OGG buffer is empty.")
            await update.message.reply_text("❌ Failed to convert audio. Please try again.")
            return
        logger.info("Successfully converted MP3 to OGG.")

        # Send voice message
        ogg_bytes = BytesIO(ogg_buffer)
        ogg_bytes.name = "voice.ogg"  # Telegram requires a filename
        ogg_bytes.seek(0)  # Reset buffer position
        await update.message.reply_voice(voice=ogg_bytes)
        logger.info("Sent voice message to user.")

        # Inform the user about remaining messages
        if remaining > 0:
            await update.message.reply_text(f"🕸️ You have **{remaining}** messages left today.", parse_mode="Markdown")
        else:
            await update.message.reply_text("⛔ You have no messages left for today. Please try again tomorrow.")

    except Exception as e:
        logger.error(f"Error handling message from user {user_id}: {e}")
        await update.message.reply_text("❌ An error occurred while processing your message. Please try again later.")

#######################################
# Main Bot
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

    logger.info("👻 KASPER Telegram Bot: GPT 4-o mini Realtime + ElevenLabs TTS + 15/day limit started. 👻")

    # Register shutdown signals
    loop = asyncio.get_event_loop()

    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(getattr(signal, signame),
                                lambda signame=signame: asyncio.create_task(application.stop()))

    # Run the bot
    try:
        application.run_polling()
    except Exception as e:
        logger.error(f"Application encountered an error: {e}")

if __name__ == "__main__":
    main()
