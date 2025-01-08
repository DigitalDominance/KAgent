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
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation

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
# Eleven Labs Client
##############################
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

##############################
# Audio Conversion
##############################
def convert_mp3_to_ogg(mp3_bytes: bytes) -> BytesIO:
    """Convert MP3 bytes to OGG/Opus for Telegram voice notes."""
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
    Creates a new ElevenLabs Conversation object.
    """
    # Reset rate limit
    context.user_data["limit_start_time"] = None
    context.user_data["response_count"] = 0

    # Create new Conversation
    conversation = Conversation(
        client=client,
        agent_id=ELEVEN_LABS_AGENT_ID,
        requires_auth=bool(ELEVEN_LABS_API_KEY),
    )
    # Optionally, conversation.start_session() if you want to explicitly begin.

    # Save it in user_data
    context.user_data["conversation"] = conversation

    await update.message.reply_text(
        "Hello! I'm Kasper (Eleven Labs Agent). How can I help you today?"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limiting (15 responses / 24 hours).
    2. Send user text to ElevenLabs conversation.
    3. Parse text + optional audio response, send to user.
    """

    # --- RATE LIMITING --- #
    limit_start_time = context.user_data.get("limit_start_time")
    response_count = context.user_data.get("response_count", 0)

    # If no start time, set it now
    if not limit_start_time:
        context.user_data["limit_start_time"] = datetime.now()
        context.user_data["response_count"] = 0
    else:
        # Check if 24 hours have passed
        time_diff = datetime.now() - limit_start_time
        if time_diff.total_seconds() >= 24 * 3600:
            # Reset window
            context.user_data["limit_start_time"] = datetime.now()
            context.user_data["response_count"] = 0
            response_count = 0

    # If user hit 15-limit
    if response_count >= 15:
        await update.message.reply_text(
            "You have used all 15 responses for the day. "
            "Please wait 24 hours from your first usage to continue."
        )
        return

    # Increment usage
    context.user_data["response_count"] = response_count + 1

    # --- ELEVEN LABS CONVERSATION --- #
    if "conversation" not in context.user_data:
        await update.message.reply_text("Please type /start first.")
        return

    conversation: Conversation = context.user_data["conversation"]

    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        # Send user text, get agent reply
        agent_reply: str = conversation.user_message(user_text)

        # Send text response to Telegram
        await update.message.reply_text(agent_reply)

        # Check for audio in last_generation
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

    logger.info("Kasper Telegram Bot is running with rate limiting & ElevenLabs Python SDK.")
    app.run_polling()

if __name__ == "__main__":
    main()
