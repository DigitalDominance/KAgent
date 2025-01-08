import os
import logging
import asyncio
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
ELEVEN_LABS_AGENT_ID = os.getenv("ELEVEN_LABS_AGENT_ID", "")  # For public agent
# If your agent is private, you'll need to get a "signed_url" from your server:
#  wss://api.elevenlabs.io/v1/convai/conversation?agent_id=...&token=...

########################
# Rate Limit: 15 / 24h
########################
MAX_MESSAGES = 15

########################
# Data Structures
########################

# We'll keep a dictionary: { user_id: { "ws": <websocket>, ... } }
# - "ws": the active WebSocketClientProtocol
# - "audio_buffer": bytes accumulative from audio chunks
# - "rate_start": datetime
# - "message_count": int
# - "is_listening": bool (true if we keep the session open)
# etc.

USER_SESSIONS = {}

########################
# WebSocket Manager
########################

async def send_user_transcript(ws: WebSocketClientProtocol, transcript: str):
    """
    Send a user_transcript message over the WS:
    {
      "user_transcript": "Hello!"
    }
    """
    msg = {"user_transcript": transcript}
    await ws.send(str(msg))  # or use json.dumps(msg)

def decode_base64_audio(audio_base64: str) -> bytes:
    """
    Decodes the base64 audio from ElevenLabs.
    Could be raw PCM or MP3, check 'agent_output_audio_format' in conversation_initiation_metadata
    In the docs, default is 'pcm_16000' for agent output, but it might differ.
    """
    return base64.b64decode(audio_base64)

def convert_to_ogg(audio_bytes: bytes, is_pcm_16k: bool = True) -> BytesIO:
    """
    If it's PCM 16k, we need to wrap it in a WAV or do something before pydub can read it.
    If it's MP3, we can parse directly, etc.
    """
    try:
        if is_pcm_16k:
            # For simplicity, wrap raw PCM in a WAV container or do a pydub workaround.
            # Quick approach: create a WAV file in memory with correct headers, then load.
            # We'll skip advanced details here. 
            # If your agent actually returns MP3, just treat it as MP3.
            logger.info("Converting from raw PCM 16k to OGG.")
            # We'll do a naive approach using pydub's raw data. 
            segment = AudioSegment(
                data=audio_bytes,
                sample_width=2,      # 16-bit = 2 bytes
                frame_rate=16000,
                channels=1
            )
        else:
            # Possibly MP3
            logger.info("Assuming MP3 data.")
            segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")

        # Now export OGG
        ogg_buf = BytesIO()
        segment.export(ogg_buf, format="ogg", codec="opus")
        ogg_buf.seek(0)
        return ogg_buf

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return BytesIO()

async def websocket_listener(user_id: int):
    """
    Background task: Connect to ElevenLabs WebSocket, keep reading messages,
    store final text + final audio in memory, then the bot can send to Telegram.
    """
    session = USER_SESSIONS[user_id]
    agent_id = ELEVEN_LABS_AGENT_ID

    # For public agents:
    ws_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"
    # If private, get a signed URL from your server.

    logger.info(f"User {user_id}: connecting to {ws_url}")

    async with websockets.connect(ws_url) as ws:
        session["ws"] = ws
        session["audio_buffer"] = b""
        session["is_listening"] = True

        # The server sends messages indefinitely until conversation ends
        # We'll keep reading in a loop.
        try:
            async for raw_msg in ws:
                # raw_msg is a string; parse it
                # The docs show messages like:
                # {
                #   "type": "audio",
                #   "audio_event": { "audio_base_64": "...", "event_id": 67890 }
                # }
                # Or agent_response, user_transcript, etc.
                raw_msg = raw_msg.strip()
                logger.debug(f"WS >> {raw_msg}")

                if raw_msg.startswith("{"):
                    # Very naive approach: parse as Python dict
                    # You should do: data = json.loads(raw_msg)
                    import json
                    data = json.loads(raw_msg)
                    msg_type = data.get("type", "")

                    if msg_type == "conversation_initiation_metadata":
                        # e.g. { "conversation_id": "...", "agent_output_audio_format": "pcm_16000" }
                        pass
                    elif msg_type == "agent_response":
                        # final text
                        agent_text = data["agent_response_event"]["agent_response"]
                        logger.info(f"User {user_id} agent text: {agent_text}")
                        # We store it so the Telegram handler can send it.
                        session["last_agent_text"] = agent_text

                    elif msg_type == "audio":
                        # partial chunk
                        chunk_b64 = data["audio_event"]["audio_base_64"]
                        chunk_bytes = decode_base64_audio(chunk_b64)
                        # We'll accumulate in session["audio_buffer"]
                        session["audio_buffer"] += chunk_bytes

                    elif msg_type == "interruption":
                        # means agent got interrupted
                        pass

                    elif msg_type == "ping":
                        # The docs say we might respond with a "pong"
                        event_id = data["ping_event"]["event_id"]
                        pong_msg = {
                            "type": "pong",
                            "event_id": event_id
                        }
                        await ws.send(str(pong_msg))

                    elif msg_type in ("user_transcript", "internal_tentative_agent_response"):
                        # Probably partial user transcripts or partial agent text
                        pass

                    # etc. handle more event types if you want

        except websockets.ConnectionClosed:
            logger.info(f"User {user_id}: WebSocket closed.")
        finally:
            session["is_listening"] = False
            session["ws"] = None

    logger.info(f"User {user_id}: WS task complete.")

########################
# Telegram Bot
########################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start -> reset rate limit, create a session in USER_SESSIONS, spawn a WS listener.
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

    await update.message.reply_text("Hello! I've started a new ElevenLabs WebSocket session. You have 15 messages every 24 hours. Ask away!")

    # Spawn the websocket_listener in the background
    asyncio.create_task(websocket_listener(user_id))

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Rate-limit 15 per 24h
    2. Send user transcript over WS
    3. Wait briefly, fetch agent text + audio
    4. Send to Telegram
    """
    user_id = update.effective_user.id
    if user_id not in USER_SESSIONS:
        await update.message.reply_text("Please type /start first to begin.")
        return

    session = USER_SESSIONS[user_id]

    # Rate limit check
    if session["rate_start"] is None:
        session["rate_start"] = datetime.now()
        session["message_count"] = 0
    else:
        elapsed = datetime.now() - session["rate_start"]
        if elapsed.total_seconds() >= 24 * 3600:
            session["rate_start"] = datetime.now()
            session["message_count"] = 0

    if session["message_count"] >= MAX_MESSAGES:
        await update.message.reply_text("You have used your 15 daily messages. Wait 24h to continue.")
        return

    session["message_count"] += 1

    # If WS not connected or listening, ask them to /start again
    if not session["is_listening"] or session["ws"] is None:
        await update.message.reply_text("WebSocket not connected or ended. Please /start again.")
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    # Send transcript
    await update.message.reply_text("Got it. Let me think...")

    try:
        ws = session["ws"]
        # We'll send a user_transcript message
        msg = {
          "user_transcript": user_text
        }
        import json
        await ws.send(json.dumps(msg))
    except Exception as e:
        logger.error(f"Failed to send user_transcript: {e}")
        await update.message.reply_text("Something went wrong sending text to the agent.")
        return

    # We'll wait a bit for the agent
    await asyncio.sleep(3.0)  # arbitrary wait

    agent_text = session.get("last_agent_text", "")
    if agent_text:
        await update.message.reply_text(agent_text)

    # If there's new audio in session["audio_buffer"], convert to OGG and send
    audio_buf = session["audio_buffer"]
    if audio_buf and len(audio_buf) > 0:
        # Convert PCM or MP3 to OGG
        # The doc says default is "pcm_16000" - let's assume that:
        ogg_data = convert_to_ogg(audio_buf, is_pcm_16k=True)
        if ogg_data.getbuffer().nbytes > 0:
            await update.message.reply_voice(voice=ogg_data)

        # Clear audio buffer so we don't send it again next time
        session["audio_buffer"] = b""

    # If agent is still streaming, partial audio might come after the wait,
    # so you can repeat or let the user ask another question.

########################
# Main
########################
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Starting Telegram bot with raw WebSocket ElevenLabs approach.")
    app.run_polling()

if __name__ == "__main__":
    main()
