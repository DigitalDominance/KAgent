import os
import json
import logging
import asyncio
from io import BytesIO
from datetime import datetime, timedelta

import httpx
import websockets
from pydub import AudioSegment

from telegram import Update, Voice
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
ELEVEN_LABS_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID", "X6Hd6garE7rwoQExOLCe")  # Example Kasper voice ID
MAX_MESSAGES_PER_USER = int(os.getenv("MAX_MESSAGES_PER_USER", "15"))

#######################################
# GPT 4-o Mini Realtime
#######################################
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # The short model name you said you'd use
GPT_REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

#######################################
# Rate Limiting
#######################################
# user_id -> {
#   "rate_start": datetime,
#   "message_count": int,
#   "persona": str,
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
            codec="opus"
        )
        ogg_buffer.seek(0)
        return ogg_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return BytesIO()

#######################################
# ElevenLabs TTS (Asynchronous)
#######################################
async def elevenlabs_tts(text: str) -> bytes:
    """
    Calls ElevenLabs TTS endpoint asynchronously, returning MP3 audio bytes.
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
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=30)
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
        ws = await websockets.connect(GPT_REALTIME_WS_URL, extra_headers=headers)
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
    logger.debug(f"Sent to GPT: {json.dumps(event)}")

    final_text = ""
    try:
        async for message in ws:
            if not message:
                logger.info("WS recv() returned empty. Breaking.")
                break

            logger.info(f"Raw WS message: {message}")  # <--- extra debug

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
                # Handle partial text if needed
                pass
            elif ev_type == "error":
                logger.error(f"GPT error: {data}")
                break

    except websockets.ConnectionClosed:
        logger.error("WebSocket connection closed unexpectedly.")

    return final_text

#######################################
# Telegram Handlers
#######################################
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start:
    - Reset daily usage
    - Open new GPT Realtime WS (4-o mini)
    - Create KASPER persona
    """
    user_id = update.effective_user.id

    # Reset or create session
    kasper_persona = (
        "You are KASPER, the friendly ghost of Kaspa (KRC20). "
        "Your goal is to entertain and inform about Kaspa or KRC20, "
        "while secretly using human psychology to get users to chat more. "
        "Speak in a playful, ghostly tone. Encourage them to keep talking!"
    )

    # Close old WebSocket if exists
    old_session = USER_SESSIONS.get(user_id)
    if old_session and old_session.get("ws"):
        try:
            await old_session["ws"].close()
            logger.info(f"Closed old WebSocket for user {user_id}.")
        except Exception as e:
            logger.error(f"Error closing old WebSocket for user {user_id}: {e}")

    # Establish new WebSocket connection
    ws = await openai_realtime_connect()
    if not ws:
        await update.message.reply_text("Could not connect to GPT. Please try again later.")
        return

    # Initialize session
    USER_SESSIONS[user_id] = {
        "ws": ws,
        "rate_start": datetime.utcnow(),
        "message_count": 0,
        "persona": kasper_persona
    }

    await update.message.reply_text(
        "KASPER is here! A fresh conversation started (4-o mini). You have 15 daily messages. Let's chat!"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limit (15 / 24h)
    2. Send user text -> GPT 4-o mini Realtime with KASPER persona
    3. TTS with ElevenLabs
    4. Convert & send audio
    5. Inform user of remaining messages
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    session = USER_SESSIONS.get(user_id)

    if not session:
        await update.message.reply_text("Please type /start first to begin a conversation.")
        return

    # Rate limiting
    rate_start = session["rate_start"]
    msg_count = session["message_count"]
    now = datetime.utcnow()

    if now - rate_start >= timedelta(hours=24):
        session["rate_start"] = now
        session["message_count"] = 0
        msg_count = 0

    if msg_count >= MAX_MESSAGES_PER_USER:
        await update.message.reply_text(
            f"You have reached your daily limit of {MAX_MESSAGES_PER_USER} messages. Please try again tomorrow."
        )
        return

    # Increment message count
    session["message_count"] += 1
    remaining = MAX_MESSAGES_PER_USER - session["message_count"]

    user_text = update.message.text.strip()
    if not user_text:
        await update.message.reply_text("Please send a non-empty message.")
        return

    # Inform user that KASPER is recording a message
    await update.message.reply_text("KASPER is recording a message...")

    # Send message to GPT Realtime
    ws = session["ws"]
    persona = session["persona"]
    gpt_reply = await send_message_gpt(ws, user_text, persona)

    if not gpt_reply:
        gpt_reply = "Oops, KASPER couldn't come up with anything. (Ghostly shrug.)"

    # ElevenLabs TTS
    mp3_data = await elevenlabs_tts(gpt_reply)
    if not mp3_data:
        await update.message.reply_text("Sorry, I couldn't process your request at the moment.")
        return

    # Convert MP3 to OGG
    loop = asyncio.get_running_loop()
    ogg_file = await loop.run_in_executor(None, convert_mp3_to_ogg, mp3_data)

    # Send voice note
    try:
        await update.message.reply_voice(voice=ogg_file)
    except Exception as e:
        logger.error(f"Error sending voice message: {e}")
        await update.message.reply_text("Failed to send voice message.")

    # Inform user about remaining messages
    if remaining > 0:
        await update.message.reply_text(f"You have {remaining} messages left today.")
    else:
        await update.message.reply_text("You have no messages left today. Please try again tomorrow.")

#######################################
# Graceful Shutdown Handling
#######################################
async def shutdown(app):
    """
    Gracefully shuts down the bot by closing all WebSocket connections.
    """
    logger.info("Shutting down bot gracefully...")
    # Close all WebSocket connections
    for user_id, session in USER_SESSIONS.items():
        ws = session.get("ws")
        if ws:
            try:
                await ws.close()
                logger.info(f"Closed WebSocket for user {user_id}.")
            except Exception as e:
                logger.error(f"Error closing WebSocket for user {user_id}: {e}")

    await app.stop()
    await app.shutdown()
    logger.info("Bot shut down successfully.")

#######################################
# Main Bot
#######################################
async def main():
    # Build the Telegram application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # Register shutdown signals
    loop = asyncio.get_running_loop()
    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda signame=signame: asyncio.create_task(shutdown(application))
        )

    # Start the bot
    logger.info("Starting KASPER Telegram Bot...")
    await application.start()
    await application.updater.start_polling()

    # Run until shutdown
    await application.updater.idle()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped manually.")
