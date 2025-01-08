import os
import logging
import asyncio
import json
import base64
from datetime import datetime
from io import BytesIO

import websockets
from websockets import WebSocketClientProtocol

from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)

########################
# Logging
########################
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

########################
# Env Vars
########################
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ELEVEN_LABS_AGENT_ID = os.getenv("ELEVEN_LABS_AGENT_ID", "")
MAX_MESSAGES_PER_DAY = 15

########################
# Session Management
########################
# We'll store user-specific data in USER_SESSIONS[user_id]
# Each entry is a dict with:
#  {
#    "ws": WebSocketClientProtocol or None,
#    "audio_buffer": bytes,         # accumulates PCM chunks
#    "last_agent_text": str,        # final text from agent
#    "rate_start": datetime,
#    "message_count": int,
#    "is_listening": bool           # True if WS open
#  }
USER_SESSIONS = {}

########################
# PCM Decoding -> OGG
########################
def decode_base64_audio(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)

def convert_pcm_to_ogg(pcm_data: bytes) -> BytesIO:
    """
    Convert raw 16-bit, 16 kHz mono PCM to OGG/Opus.
    """
    try:
        # Create an AudioSegment from raw PCM
        segment = AudioSegment.from_raw(
            BytesIO(pcm_data),
            sample_width=2,      # 16-bit = 2 bytes
            frame_rate=16000,
            channels=1
        )
        ogg_buf = BytesIO()
        segment.export(ogg_buf, format="ogg", codec="opus")
        ogg_buf.seek(0)
        return ogg_buf
    except Exception as e:
        logger.error(f"PCM -> OGG conversion error: {e}")
        return BytesIO()

########################
# WebSocket Listener
########################
async def websocket_listener(user_id: int):
    """
    Connect to ElevenLabs wss:// for this user, read all streaming messages,
    store final text in last_agent_text, store PCM chunks in audio_buffer.
    """
    session = USER_SESSIONS[user_id]
    ws_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVEN_LABS_AGENT_ID}"

    logger.info(f"User {user_id}: connecting WebSocket to {ws_url}")
    session["audio_buffer"] = b""
    session["last_agent_text"] = ""
    session["is_listening"] = True

    try:
        async with websockets.connect(ws_url) as ws:
            session["ws"] = ws

            async for raw_msg in ws:
                raw_msg = raw_msg.strip()
                logger.debug(f"WS >> {raw_msg}")

                if not raw_msg.startswith("{"):
                    continue

                data = json.loads(raw_msg)
                msg_type = data.get("type", "")

                if msg_type == "conversation_initiation_metadata":
                    # e.g. conversation_id, agent_output_audio_format, etc.
                    logger.info(f"conversation_initiation_metadata: {data}")
                elif msg_type == "agent_response":
                    agent_text = data["agent_response_event"]["agent_response"]
                    logger.info(f"User {user_id} agent final text: {agent_text}")
                    session["last_agent_text"] = agent_text
                elif msg_type == "audio":
                    audio_b64 = data["audio_event"]["audio_base_64"]
                    chunk = decode_base64_audio(audio_b64)
                    # Accumulate raw PCM chunk in session
                    session["audio_buffer"] += chunk
                elif msg_type == "ping":
                    # respond with pong
                    event_id = data["ping_event"]["event_id"]
                    pong_msg = {"type": "pong", "event_id": event_id}
                    await ws.send(json.dumps(pong_msg))
                elif msg_type == "interruption":
                    # agent interrupted
                    logger.info(f"User {user_id} agent got interrupted.")
                elif msg_type == "internal_tentative_agent_response":
                    # partial text
                    partial_text = data["tentative_agent_response_internal_event"]["tentative_agent_response"]
                    logger.debug(f"User {user_id} partial text: {partial_text}")
                    # We won't store partial text, but you could if you want
                else:
                    logger.debug(f"Unhandled msg type: {msg_type}")

    except websockets.ConnectionClosed:
        logger.info(f"User {user_id}: WebSocket closed by server.")
    except Exception as e:
        logger.error(f"User {user_id} WS error: {e}")
    finally:
        logger.info(f"User {user_id}: WS listener finished.")
        session["ws"] = None
        session["is_listening"] = False


########################
# Telegram Handlers
########################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start -> Reset rate limit, spawn a WS listener
    """
    user_id = update.effective_user.id
    USER_SESSIONS[user_id] = {
        "ws": None,
        "audio_buffer": b"",
        "last_agent_text": "",
        "rate_start": None,
        "message_count": 0,
        "is_listening": False
    }

    await update.message.reply_text(
        "Hello! I've started a new WebSocket session to ElevenLabs. "
        "You have 15 messages per 24 hours. Ask away!"
    )

    asyncio.create_task(websocket_listener(user_id))

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Rate-limit to 15 messages per 24h
    2. Send user_transcript JSON to WS
    3. Wait 5 seconds, retrieve last_agent_text & audio_buffer
    4. Convert PCM -> OGG, send to Telegram
    """
    user_id = update.effective_user.id
    if user_id not in USER_SESSIONS:
        await update.message.reply_text("Please /start first.")
        return

    session = USER_SESSIONS[user_id]

    # -- Rate limiting --
    if session["rate_start"] is None:
        session["rate_start"] = datetime.now()
        session["message_count"] = 0
    else:
        elapsed = datetime.now() - session["rate_start"]
        if elapsed.total_seconds() >= 24 * 3600:
            session["rate_start"] = datetime.now()
            session["message_count"] = 0

    if session["message_count"] >= MAX_MESSAGES_PER_DAY:
        await update.message.reply_text(
            f"You have used your {MAX_MESSAGES_PER_DAY} daily messages. Wait 24h to continue."
        )
        return

    session["message_count"] += 1

    if not session["is_listening"] or session["ws"] is None:
        await update.message.reply_text("WebSocket is not connected. Please /start again.")
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    await update.message.reply_text("Got it, thinking...")

    try:
        # Send user_transcript to WS
        msg = {"user_transcript": user_text}
        await session["ws"].send(json.dumps(msg))
    except Exception as e:
        logger.error(f"User {user_id} could not send to WS: {e}")
        await update.message.reply_text("Error sending your text to agent.")
        return

    # Wait 5s for agent to produce final text + audio
    await asyncio.sleep(5)

    # 1) Send final text
    agent_text = session.get("last_agent_text", "")
    if agent_text:
        await update.message.reply_text(agent_text)
    else:
        await update.message.reply_text("No text response from agent yet.")

    # 2) If we have PCM in audio_buffer, convert & send
    if session["audio_buffer"]:
        pcm_data = session["audio_buffer"]
        ogg_file = convert_pcm_to_ogg(pcm_data)
        if ogg_file.getbuffer().nbytes > 0:
            await update.message.reply_voice(voice=ogg_file)
            logger.info(f"Sent {len(pcm_data)} bytes of PCM as OGG to user {user_id}")
        else:
            logger.info(f"Audio conversion failed or was empty for user {user_id}")

        # Clear buffer so we don't send it again next time
        session["audio_buffer"] = b""
    else:
        logger.info(f"No audio chunks for user {user_id}")

########################
# Main
########################

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Starting Telegram bot with raw ElevenLabs WS PCM approach.")
    app.run_polling()

if __name__ == "__main__":
    main()
