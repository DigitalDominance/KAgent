import logging
import os
import requests
from io import BytesIO
from datetime import datetime, timedelta

from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)

##############################
# Logging
##############################
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

##############################
# Environment Variables
##############################
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ELEVEN_LABS_API_KEY = os.environ.get("ELEVEN_LABS_API_KEY", "")
ELEVEN_LABS_AGENT_ID = os.environ.get("ELEVEN_LABS_AGENT_ID", "")

##############################
# Eleven Labs: Conversations
##############################

def converse_with_elevenlabs(user_input: str, conversation_id: str = None) -> dict:
    """
    Calls POST /v1/convai/conversations with:
      - agent_id
      - input (the user text)
      - optionally conversation_id if continuing
    Returns JSON including:
      - conversation_id
      - history (list of messages)
      - outputs (list of {text, voice_id, generation_id})
    """
    url = "https://api.elevenlabs.io/v1/convai/conversations"
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "agent_id": ELEVEN_LABS_AGENT_ID,
        "input": user_input
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Error while calling Eleven Labs conversations: {e}")
        return {}

def fetch_audio_mp3(conversation_id: str, voice_id: str, generation_id: str) -> bytes:
    """
    Calls GET /v1/convai/conversations/{conversation_id}/audio
    ?voice_id=...&generation_id=...
    Returns raw MP3 bytes from Eleven Labs.
    """
    url = (
        f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}/audio"
        f"?voice_id={voice_id}&generation_id={generation_id}"
    )
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.content  # MP3 data
    except Exception as e:
        logger.error(f"Error fetching audio from Eleven Labs: {e}")
        return b""

def convert_mp3_to_ogg(mp3_bytes: bytes) -> BytesIO:
    """
    Converts MP3 bytes to OGG/Opus for Telegram voice notes.
    """
    try:
        mp3_file = BytesIO(mp3_bytes)
        segment = AudioSegment.from_file(mp3_file, format="mp3")
        ogg_buffer = BytesIO()
        segment.export(ogg_buffer, format="ogg", codec="opus")
        ogg_buffer.seek(0)
        return ogg_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return BytesIO()

##############################
# Telegram Handlers
##############################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start - Resets conversation_id and rate-limit counters for this user.
    """
    context.user_data["conversation_id"] = None
    context.user_data["limit_start_time"] = None
    context.user_data["response_count"] = 0

    await update.message.reply_text(
        "Hello! I'm Kasper (Eleven Labs Agent). How can I help you today?"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limiting (15 responses / 24 hours).
    2. Send user text to Eleven Labs conversation.
    3. Parse the reply -> outputs array -> get text + audio.
    4. If there's audio, fetch it, convert, and send to user.
    """

    # -- RATE LIMITING LOGIC -- #
    limit_start_time = context.user_data.get("limit_start_time")
    response_count = context.user_data.get("response_count", 0)

    # If no limit_start_time, this is the user's FIRST message in a 24-hour window.
    if not limit_start_time:
        context.user_data["limit_start_time"] = datetime.now()
        context.user_data["response_count"] = 0
    else:
        # Check if 24 hours have passed since limit_start_time.
        time_diff = datetime.now() - limit_start_time
        if time_diff.total_seconds() >= 24 * 3600:
            # Reset the window.
            context.user_data["limit_start_time"] = datetime.now()
            context.user_data["response_count"] = 0
            response_count = 0  # We just reset it.

    # Now check if user has hit the 15-response limit.
    if response_count >= 15:
        await update.message.reply_text(
            "You have used all 15 responses for the day. "
            "Please wait 24 hours from your first usage to continue."
        )
        return

    # If below limit, increment response_count
    context.user_data["response_count"] = response_count + 1

    # -- ELEVEN LABS CONVERSATION -- #
    user_text = update.message.text
    conversation_id = context.user_data.get("conversation_id", None)

    # 1) POST /v1/convai/conversations
    response_data = converse_with_elevenlabs(user_text, conversation_id)
    if not response_data:
        await update.message.reply_text("Sorry, I'm having trouble connecting to the Agent.")
        return

    # Possibly update conversation_id
    new_conversation_id = response_data.get("conversation_id")
    if new_conversation_id:
        context.user_data["conversation_id"] = new_conversation_id

    # The response JSON might look like:
    # {
    #   "conversation_id": "...",
    #   "history": [...],
    #   "outputs": [
    #       {
    #         "text": "...",
    #         "voice_id": "...",
    #         "generation_id": "..."
    #       }
    #   ]
    # }
    outputs = response_data.get("outputs", [])
    if not outputs:
        await update.message.reply_text("No output from the Agent.")
        return

    # Grab the latest output
    latest = outputs[-1]  # last item
    agent_text = latest.get("text", "")
    voice_id = latest.get("voice_id")
    generation_id = latest.get("generation_id")

    # 2) Send the Agent's text to user
    await update.message.reply_text(agent_text)

    # 3) If we have voice_id & generation_id, fetch audio (MP3) and convert to OGG
    if voice_id and generation_id:
        mp3_data = fetch_audio_mp3(new_conversation_id, voice_id, generation_id)
        if mp3_data:
            ogg_buffer = convert_mp3_to_ogg(mp3_data)
            if ogg_buffer.getbuffer().nbytes > 0:
                await update.message.reply_voice(voice=ogg_buffer)

##############################
# Main Bot
##############################
def main():
    from telegram.ext import ApplicationBuilder

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))

    # Text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logging.info("Kasper Telegram Bot is running with rate limiting & Eleven Labs conversations.")
    app.run_polling()

if __name__ == "__main__":
    main()
