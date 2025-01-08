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
    A no-op audio interface that doesn't record or play local audio.
    This allows the Conversation to function without local mic/speaker usage.
    """
    def record_audio(self):
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

    # Create a new Conversation with a silent audio interface
    conversation = Conversation(
        client=client,
        agent_id=ELEVEN_LABS_AGENT_ID,
        requires_auth=bool(ELEVEN_LABS_API_KEY),
        audio_interface=SilentAudioInterface(),
        # Example of optional callbacks:
        callback_agent_response=lambda response: logger.info(f"Agent text: {response}"),
        callback_agent_response_correction=lambda original, corrected: logger.info(f"Agent corrected: {original} -> {corrected}"),
        callback_user_transcript=lambda transcript: logger.info(f"User text: {transcript}"),
    )

    context.user_data["conversation"] = conversation

    await update.message.reply_text(
        "Hello! I'm Kasper (Eleven Labs Agent). "
        "Ask me anything (you have 15 responses every 24 hours)."
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limiting (15 responses / 24 hours).
    2. Send user text to ElevenLabs conversation.
    3. Fetch text response + TTS audio. Send both to Telegram.
    """

    # --- RATE LIMITING --- #
    limit_start_time = context.user_data.get("limit_start_time")
    response_count = context.user_data.get("response_count", 0)

    if not limit_start_time:
        # First message in a new 24-hour window
        context.user_data["limit_start_time"] = datetime.now()
        context.user_data["response_count"] = 0
    else:
        # Check if 24 hours have passed
        time_diff = datetime.now() - limit_start_time
        if time_diff.total_seconds() >= 24 * 3600:
            # Reset usage
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

    # --- ELEVEN LABS CONVERSATION --- #
    if "conversation" not in context.user_data:
        await update.message.reply_text("Please type /start first to begin.")
        return

    conversation: Conversation = context.user_data["conversation"]
    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        # 1) Send user text to the conversation
        agent_reply = conversation.user_message(user_text)  # returns agent's text

        # 2) Send agent text to user
        await update.message.reply_text(agent_reply)

        # 3) If TTS audio was generated, it should be in conversation.last_generation.audio
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

    logger.info("Kasper Telegram Bot is running with rate limiting & ElevenLabs (using required audio_interface).")
    app.run_polling()

if __name__ == "__main__":
    main()
