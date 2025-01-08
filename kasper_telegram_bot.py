import logging
import os
from io import BytesIO
from datetime import datetime

from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)

# ElevenLabs Python SDK
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface

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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", "")
ELEVEN_LABS_AGENT_ID = os.getenv("ELEVEN_LABS_AGENT_ID", "")

##############################
# ElevenLabs Client
##############################
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

##############################
# Custom "Silent" Audio Interface
##############################
class SilentAudioInterface(AudioInterface):
    """
    A no-op audio interface that doesn't record or play local audio
    but implements all abstract methods, so it's valid.
    """

    def start(self):
        pass

    def stop(self):
        pass

    def interrupt(self):
        pass

    def output(self, audio_bytes: bytes):
        pass

    def record_audio(self) -> bytes:
        return b""

    def play_audio(self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int):
        pass

##############################
# Audio Conversion (MP3->OGG)
##############################
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
    /start - Resets conversation and rate-limit counters for this user.
    Initializes a new ElevenLabs Conversation with a SilentAudioInterface.
    """
    # Reset rate-limiting
    context.user_data["limit_start_time"] = None
    context.user_data["response_count"] = 0

    # Create a new Conversation with the silent interface
    conversation = Conversation(
        client=client,
        agent_id=ELEVEN_LABS_AGENT_ID,
        requires_auth=bool(ELEVEN_LABS_API_KEY),
        audio_interface=SilentAudioInterface(),
        callback_agent_response=lambda resp: logger.info(f"Agent text: {resp}"),
        callback_agent_response_correction=lambda orig, corr: logger.info(f"Agent corrected: {orig} -> {corr}"),
        callback_user_transcript=lambda user_text: logger.info(f"User text: {user_text}"),
    )

    context.user_data["conversation"] = conversation

    await update.message.reply_text(
        "Hello! I'm Kasper (Eleven Labs Agent). You have 15 responses per 24 hours. Ask away!"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limiting (15 responses / 24 hours).
    2. Send user text to ElevenLabs conversation.
    3. Fetch text + TTS audio, send to user.
    """
    limit_start_time = context.user_data.get("limit_start_time")
    response_count = context.user_data.get("response_count", 0)

    if not limit_start_time:
        context.user_data["limit_start_time"] = datetime.now()
        context.user_data["response_count"] = 0
    else:
        time_diff = datetime.now() - limit_start_time
        if time_diff.total_seconds() >= 24 * 3600:
            context.user_data["limit_start_time"] = datetime.now()
            context.user_data["response_count"] = 0
            response_count = 0

    if response_count >= 15:
        await update.message.reply_text(
            "You have used all 15 responses for the day. "
            "Please wait 24 hours from your first usage to continue."
        )
        return

    context.user_data["response_count"] = response_count + 1

    # Check if conversation is initialized
    if "conversation" not in context.user_data:
        await update.message.reply_text("Please type /start first.")
        return

    conversation: Conversation = context.user_data["conversation"]
    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        # Let the agent handle the user's text
        agent_reply = conversation.user_message(user_text)

        # Send text to Telegram
        await update.message.reply_text(agent_reply)

        # If TTS audio is produced, send as voice note
        last_gen = conversation.last_generation
        if last_gen and last_gen.audio:
            ogg_data = convert_mp3_to_ogg(last_gen.audio)
            if ogg_data.getbuffer().nbytes > 0:
                await update.message.reply_voice(voice=ogg_data)

    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        await update.message.reply_text("Sorry, I'm having trouble connecting to the Agent.")

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

    logger.info("Kasper Telegram Bot is running with rate limiting & a custom SilentAudioInterface.")
    app.run_polling()

if __name__ == "__main__":
    main()
