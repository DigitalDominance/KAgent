import logging
import os
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

# ElevenLabs imports (Python SDK)
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AgentResponse

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
# Eleven Labs: Python SDK
##############################

# Create a single client for all conversations
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

##############################
# Utility: Convert MP3 to OGG
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
    /start - Resets conversation_id and rate-limit counters for this user.
    Also creates a new ElevenLabs Conversation object.
    """
    # Clear rate limit data
    context.user_data["limit_start_time"] = None
    context.user_data["response_count"] = 0

    # Create a new Conversation instance from the ElevenLabs SDK
    conversation = Conversation(
        client=client,
        agent_id=ELEVEN_LABS_AGENT_ID,
        requires_auth=bool(ELEVEN_LABS_API_KEY),
        # We won't use DefaultAudioInterface because
        # we're in Telegram, not local mic/speaker usage
    )
    # (Optionally) start_session if you want to force a new session:
    # conversation.start_session()

    context.user_data["conversation"] = conversation

    await update.message.reply_text(
        "Hello! I'm Kasper. The Friendly Ghost of KRC20!"
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Enforce rate-limiting (15 responses / 24 hours).
    2. If we have an ElevenLabs Conversation, send user text to it.
    3. Get the agent's text + optional TTS audio, and send both to the user.
    """

    # --- RATE LIMITING LOGIC --- #
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

    # --- ELEVEN LABS CONVERSATION LOGIC --- #
    if "conversation" not in context.user_data:
        await update.message.reply_text("Please type /start to begin your session.")
        return

    conversation: Conversation = context.user_data["conversation"]

    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        # Send the user's message to the conversation
        # This should return the agent's text reply and (optionally) audio data
        agent_reply: str = conversation.user_message(user_text)

        # Send agent text to Telegram
        await update.message.reply_text(agent_reply)

        # If the conversation generated audio for this message, we can retrieve it:
        # 'last_generation' is an AgentResponse that might contain .audio bytes
        last_gen: AgentResponse = conversation.last_generation
        if last_gen and last_gen.audio:
            # Convert MP3 -> OGG
            ogg_data = convert_mp3_to_ogg(last_gen.audio)
            if ogg_data.getbuffer().nbytes > 0:
                await update.message.reply_voice(voice=ogg_data)

    except Exception as e:
        logger.error(f"Error handling conversation: {e}")
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

    logging.info("Kasper Telegram Bot is running with rate limiting & Eleven Labs (Python SDK).")
    app.run_polling()

if __name__ == "__main__":
    main()
