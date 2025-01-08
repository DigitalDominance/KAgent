import logging
import os
import threading
from datetime import datetime
import asyncio
from io import BytesIO

from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    AudioInterface
)

##########################
# Logging
##########################
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

##########################
# Environment Vars
##########################
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", "")
ELEVEN_LABS_AGENT_ID = os.getenv("ELEVEN_LABS_AGENT_ID", "")

##########################
# ElevenLabs Client
##########################
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

##########################
# Data Structures
##########################
# We'll store the final agent text for each user ID,
# plus the final MP3 for each user ID once the agent stops speaking.
USER_AGENT_TEXT = {}      # { user_id: str }
USER_AGENT_AUDIO = {}     # { user_id: bytes (final MP3) }

##########################
# TelegramAudioInterface
##########################
class TelegramAudioInterface(AudioInterface):
    """
    Custom AudioInterface that:
    - Doesn't record or play local audio.
    - Accumulates TTS audio chunks in memory.
    - When agent stops speaking, we finalize the buffer so the bot can send it to Telegram.
    """

    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id
        self._buffer = BytesIO()   # accumulates partial audio
        self._recording = False

    def start(self):
        # Agent is about to speak
        logger.info(f"AudioInterface.start() for user {self.user_id}")
        self._recording = True
        self._buffer = BytesIO()

    def stop(self):
        # Agent finished speaking
        logger.info(f"AudioInterface.stop() for user {self.user_id}")
        self._recording = False

        # The entire MP3 is in self._buffer now
        mp3_data = self._buffer.getvalue()
        USER_AGENT_AUDIO[self.user_id] = mp3_data

        # Clear the buffer for next time
        self._buffer = BytesIO()

    def interrupt(self):
        # If user interrupts the agent
        logger.info(f"AudioInterface.interrupt() for user {self.user_id}")
        self._recording = False

    def output(self, audio_bytes: bytes):
        # Called repeatedly with partial chunks
        if self._recording:
            self._buffer.write(audio_bytes)

    def record_audio(self) -> bytes:
        # We don't capture user mic in Telegram
        return b""

    def play_audio(self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int):
        # We don't play local audio
        pass


##########################
# Callbacks
##########################
def run_conversation_loop(conversation: Conversation):
    """
    Runs in background thread: start_session, wait_for_session_end.
    """
    logger.info("Starting conversation session (streaming) ...")
    conversation.start_session()
    logger.info("Session started, now waiting for end ...")
    cid = conversation.wait_for_session_end()
    logger.info(f"Conversation ended. ID={cid}")

def make_callback_agent_response(user_id: int):
    """
    Returns a function that handles agent text responses.
    We'll store the final text in USER_AGENT_TEXT[user_id].
    """
    def callback(response_text: str):
        logger.info(f"[Agent partial/final text] user={user_id}, text={response_text}")
        # For simplicity, we store the entire response in a single var
        # The agent might produce partial text, then final text.
        # You can refine logic if needed.
        USER_AGENT_TEXT[user_id] = response_text
    return callback


##########################
# Utility: MP3->OGG
##########################
def convert_mp3_to_ogg(mp3_data: bytes) -> BytesIO:
    try:
        mp3_file = BytesIO(mp3_data)
        segment = AudioSegment.from_file(mp3_file, format="mp3")
        ogg_buffer = BytesIO()
        segment.export(ogg_buffer, format="ogg", codec="opus")
        ogg_buffer.seek(0)
        return ogg_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return BytesIO()


##########################
# Telegram Handlers
##########################
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start -> reset rate limit, create new conversation in a background thread.
    """
    user_id = update.effective_user.id
    context.user_data["limit_start_time"] = None
    context.user_data["response_count"] = 0

    # Create a new audio interface per user
    audio_interface = TelegramAudioInterface(user_id)

    # Build the conversation
    conversation = Conversation(
        client=client,
        agent_id=ELEVEN_LABS_AGENT_ID,
        requires_auth=bool(ELEVEN_LABS_API_KEY),
        audio_interface=audio_interface,
        callback_agent_response=make_callback_agent_response(user_id),
        # callback_agent_response_correction=..., etc. if needed
    )

    # Save the conversation in user_data
    context.user_data["conversation"] = conversation

    # Start session in background thread
    t = threading.Thread(target=run_conversation_loop, args=(conversation,))
    t.daemon = True
    t.start()

    await update.message.reply_text("New ElevenLabs conversation started. You have 15 daily messages. Ask away!")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Rate-limit check
    2. Put text into conversation.input_text_queue
    3. Wait briefly, then fetch agent's final text & audio
    4. Send to user
    """
    user_id = update.effective_user.id

    # --- Rate limiting ---
    limit_start_time = context.user_data.get("limit_start_time")
    response_count = context.user_data.get("response_count", 0)

    if not limit_start_time:
        context.user_data["limit_start_time"] = datetime.now()
        context.user_data["response_count"] = 0
    else:
        elapsed = datetime.now() - limit_start_time
        if elapsed.total_seconds() >= 24 * 3600:
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

    # Check conversation
    if "conversation" not in context.user_data:
        await update.message.reply_text("Please type /start first.")
        return

    conversation: Conversation = context.user_data["conversation"]

    user_text = update.message.text.strip()
    if not user_text:
        return

    # Put text into the queue
    logger.info(f"User {user_id} input_text_queue: {user_text}")
    conversation.input_text_queue.put(user_text)

    # Let user know we got the message
    await update.message.reply_text("Got it, thinking...")

    # The agent will produce partial/final text in callback_agent_response,
    # and TTS audio in audio_interface output -> stored in USER_AGENT_AUDIO.

    # We'll wait a little bit for the agent to respond
    await asyncio.sleep(2.0)

    # Retrieve agent text from dictionary
    agent_text = USER_AGENT_TEXT.get(user_id)
    if agent_text:
        await update.message.reply_text(agent_text)
    else:
        await update.message.reply_text("No text reply from the agent yet. Try waiting or asking again.")

    # Check if we have final MP3 audio
    mp3_data = USER_AGENT_AUDIO.get(user_id)
    if mp3_data and len(mp3_data) > 0:
        # Convert to OGG
        ogg_data = convert_mp3_to_ogg(mp3_data)
        if ogg_data.getbuffer().nbytes > 0:
            await update.message.reply_voice(voice=ogg_data)

        # Clear the stored MP3 so we don't resend it next time
        USER_AGENT_AUDIO[user_id] = b""

##########################
# Main Bot
##########################
def main():
    from telegram.ext import ApplicationBuilder

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Kasper Telegram Bot with streaming-based ElevenLabs conversation.")
    app.run_polling()

if __name__ == "__main__":
    main()
