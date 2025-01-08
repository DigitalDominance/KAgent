import os
import json
import logging
import requests
from io import BytesIO
from datetime import datetime

import websocket  # from websocket-client
from pydub import AudioSegment

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
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

#######################################
# GPT Realtime (4o mini) Settings
#######################################
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"

#######################################
# ElevenLabs TTS Settings
#######################################
ELEVEN_LABS_VOICE_ID = "X6Hd6garE7rwoQExOLCe"  # Example voice for KASPER

#######################################
# Rate Limit: 15 messages / 24h
#######################################
MAX_MESSAGES = 15

#######################################
# In-memory Session Store
#######################################
# user_id -> {
#   "ws": <websocket.WebSocket> or None,
#   "rate_start": datetime,
#   "message_count": int,
#   "persona": str (optional),
# }
USER_SESSIONS = {}

#######################################
# Utility: Convert MP3 -> OGG
#######################################
def convert_mp3_to_ogg(mp3_data: bytes) -> BytesIO:
    """
    Convert MP3 bytes to OGG (Opus) so we can send as a Telegram voice note.
    We add parameters=["-strict","-2"] to allow the 'opus' encoder in ffmpeg.
    """
    try:
        mp3_file = BytesIO(mp3_data)
        segment = AudioSegment.from_file(mp3_file, format="mp3")
        ogg_buffer = BytesIO()
        segment.export(
            ogg_buffer,
            format="ogg",
            codec="opus",
            parameters=["-strict","-2"]  # allow experimental opus
        )
        ogg_buffer.seek(0)
        return ogg_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return BytesIO()

#######################################
# ElevenLabs TTS
#######################################
def elevenlabs_tts(text: str) -> bytes:
    """
    Calls ElevenLabs POST /v1/text-to-speech/{voice_id}
    Returns raw MP3 audio bytes.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.content  # MP3 data
    except Exception as e:
        logger.error(f"Error calling ElevenLabs TTS: {e}")
        return b""

#######################################
# GPT Realtime WebSocket
#######################################
def openai_realtime_connect() -> websocket.WebSocket:
    """
    Opens a blocking WebSocket connection to OpenAI Realtime API (4o mini).
    """
    url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    headers = [
        "Authorization: Bearer " + OPENAI_API_KEY,
        "OpenAI-Beta: realtime=v1"
    ]
    ws = websocket.WebSocket()
    ws.connect(url, header=headers)
    logger.info("Connected to GPT Realtime (4o mini).")
    return ws

def send_message_gpt(ws: websocket.WebSocket, user_text: str, persona: str) -> str:
    """
    Send user text to GPT Realtime with 'response.create' event.
    We embed the KASPER persona + user text in instructions.
    Wait for 'response.complete' event, return final text.
    """
    # Combine the user_text with your KASPER persona instructions
    # so GPT always responds as "KASPER the friendly ghost of Kaspa."
    # Also, we encourage a subtle psychological approach to keep them talking.
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
    ws.send(json.dumps(event))

    final_text = ""
    while True:
        msg = ws.recv()
        if not msg:
            break
        try:
            data = json.loads(msg)
        except Exception:
            logger.debug(f"Non-JSON WS message: {msg}")
            continue

        ev_type = data.get("type", "")
        logger.debug(f"GPT event: {data}")

        if ev_type == "response.complete":
            # final text
            final_text = data["response"]["payload"]["text"]
            break
        elif ev_type == "response.intermediate":
            # partial text if you want to handle it
            pass
        elif ev_type == "error":
            logger.error(f"GPT error: {data}")
            break

    return final_text

#######################################
# Telegram Handlers
#######################################
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start:
    - Close old WS if exists
    - Reset daily usage
    - Open new GPT Realtime WS
    - Store a persona string for the user
    """
    user_id = update.effective_user.id

    # If user had an old session, close the WS
    old_session = USER_SESSIONS.get(user_id)
    if old_session and old_session.get("ws"):
        try:
            old_session["ws"].close()
        except:
            pass

    # Create a KASPER persona
    kasper_persona = (
        "You are KASPER, the friendly ghost of Kaspa (KRC20). "
        "Your goal is to entertain and inform users about Kaspa, "
        "but also subtly use human psychology to get them to chat more. "
        "Speak with a playful, ghostly warmth. "
        "Encourage them to keep talking. "
        "Make references to Kaspa or KRC20 tokens as relevant, but stay friendly!"
    )

    USER_SESSIONS[user_id] = {
        "ws": None,
        "rate_start": datetime.now(),
        "message_count": 0,
        "persona": kasper_persona
    }

    # Connect to GPT Realtime
    try:
        ws = openai_realtime_connect()
        USER_SESSIONS[user_id]["ws"] = ws
    except Exception as e:
        logger.error(f"Failed to connect GPT Realtime: {e}")
        await update.message.reply_text("Could not connect to GPT. Try again later.")
        return

    await update.message.reply_text(
        "KASPER is here! Fresh conversation started. You have 15 daily messages. "
        "Ask me anything about Kaspa or KRC20!"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Rate-limit 15 messages / 24h
    2. Send user text -> GPT Realtime (with KASPER persona)
    3. TTS with ElevenLabs
    4. Convert & send audio
    """
    user_id = update.effective_user.id
    session = USER_SESSIONS.get(user_id)
    if not session:
        await update.message.reply_text("Please type /start first.")
        return

    # Rate limiting
    rate_start = session["rate_start"]
    msg_count = session["message_count"]
    elapsed = (datetime.now() - rate_start).total_seconds()
    if elapsed >= 24 * 3600:
        # reset
        session["rate_start"] = datetime.now()
        session["message_count"] = 0
        msg_count = 0

    if msg_count >= MAX_MESSAGES:
        await update.message.reply_text(
            f"You reached {MAX_MESSAGES} daily messages. Wait 24h to continue."
        )
        return

    session["message_count"] = msg_count + 1

    user_text = update.message.text.strip()
    if not user_text:
        return

    # Check WS
    ws = session["ws"]
    if not ws:
        await update.message.reply_text("WebSocket not available. Please /start again.")
        return

    await update.message.reply_text("KASPER is thinking...")

    persona = session["persona"]
    # 1) GPT Realtime
    gpt_reply = ""
    try:
        gpt_reply = send_message_gpt(ws, user_text, persona)
    except Exception as e:
        logger.error(f"Error sending GPT msg: {e}")
        await update.message.reply_text("GPT error. Try again later.")
        return

    if not gpt_reply:
        gpt_reply = "I couldn't come up with a reply. (KASPER shrugs ghostly.)"

    # 2) ElevenLabs TTS
    mp3_data = elevenlabs_tts(gpt_reply)

    # 3) Convert MP3->OGG
    ogg_file = convert_mp3_to_ogg(mp3_data)

    # 4) Send text + voice note
    await update.message.reply_text(gpt_reply)
    if ogg_file.getbuffer().nbytes > 0:
        await update.message.reply_voice(voice=ogg_file)
    else:
        logger.info("No TTS audio or conversion failed.")

#######################################
# Main Bot
#######################################
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Kasper Telegram Bot: GPT Realtime + ElevenLabs TTS + 15/day limit, KASPER persona.")
    app.run_polling()

if __name__ == "__main__":
    main()
