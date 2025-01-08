import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO

import httpx
import websockets
from pydub import AudioSegment

from telegram import Update, Voice
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
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # The short model name you said you'd use
GPT_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

#######################################
# ElevenLabs TTS
#######################################
ELEVEN_LABS_VOICE_ID = "X6Hd6garE7rwoQExOLCe"  # Example KASPER voice ID
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
            codec="opus",
            bitrate="64k"
        )
        ogg_buffer.seek(0)
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
            resp = await client.post(ELEVEN_LABS_TTS_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
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
    Wait for 'response.complete' event; returns final text.
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

            if ev_type == "response.complete":
                final_text = data["response"]["payload"]["text"]
                logger.info(f"Got final text: {final_text}")
                break
            elif ev_type == "response.intermediate":
                # Handle intermediate responses if needed
                pass
            elif ev_type == "error":
                logger.error(f"GPT error: {data}")
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
        "Speak in a playful, ghostly tone. Encourage them to keep talking!"
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
            "KASPER is here! A fresh conversation started (4-o mini). You have 15 daily messages. Let's chat!"
        )
    else:
        await update.message.reply_text("Could not connect to GPT. Try again later.")

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
        await update.message.reply_text("Please type /start first.")
        return

    # Rate limiting
    rate_info = USER_MESSAGE_LIMITS[user_id]
    if datetime.utcnow() >= rate_info["reset_time"]:
        rate_info["count"] = 0
        rate_info["reset_time"] = datetime.utcnow() + timedelta(hours=24)

    if rate_info["count"] >= MAX_MESSAGES_PER_USER:
        await update.message.reply_text(f"You have reached the limit of {MAX_MESSAGES_PER_USER} messages for today. Please try again later.")
        return

    rate_info["count"] += 1
    remaining = MAX_MESSAGES_PER_USER - rate_info["count"]

    user_text = update.message.text.strip()
    if not user_text:
        return

    # Check WebSocket
    ws = session.get("ws")
    if not ws:
        await update.message.reply_text("WebSocket not available. Please /start again.")
        return

    try:
        # Update Status Message
        await update.message.reply_text("Kasper is recording a message.")

        # Send message to GPT Realtime
        gpt_reply = await send_message_gpt(ws, user_text, session["persona"])

        if not gpt_reply:
            gpt_reply = "Oops, KASPER couldn't come up with anything. (Ghostly shrug.)"

        # ElevenLabs TTS
        mp3_data = await elevenlabs_tts(gpt_reply)
        if not mp3_data:
            await update.message.reply_text("Sorry, I couldn't process your request.")
            return

        ogg_file = convert_mp3_to_ogg(mp3_data)

        # Send voice message
        ogg_buffer = ogg_file.getvalue()
        if ogg_buffer:
            ogg_bytes = BytesIO(ogg_buffer)
            await update.message.reply_voice(voice=ogg_bytes)
        else:
            logger.info("No TTS audio or conversion failed.")

        # Inform the user about remaining messages
        if remaining > 0:
            await update.message.reply_text(f"You have {remaining} messages left today.")
        else:
            await update.message.reply_text("You have no messages left for today. Please try again tomorrow.")

    except Exception as e:
        logger.error(f"Error handling message from user {user_id}: {e}")
        await update.message.reply_text("An error occurred while processing your message. Please try again later.")

#######################################
# Main Bot
#######################################
async def main():
    # Build the Telegram application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("KASPER Telegram Bot: GPT 4-o mini Realtime + ElevenLabs TTS + 15/day limit.")
    
    # Run the bot until Ctrl+C or process termination
    await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped manually.")
