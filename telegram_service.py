import asyncio
from telegram import Bot, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from config import config
from llm_service import llm_service


class TelegramService:
    """Telegram Bot API wrapper for sending messages, recipes, and grocery lists."""

    def __init__(self):
        token = config.TELEGRAM_BOT_TOKEN
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured")
        self._bot = Bot(token=token)
        self._loop = asyncio.new_event_loop()

    # ==================== INTERNAL HELPERS ====================

    def _run_async(self, coro):
        """Run an async coroutine from a synchronous context using a persistent event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run on our persistent loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(self._loop.run_until_complete, coro).result()
        else:
            return self._loop.run_until_complete(coro)

    def _build_recipe_keyboard(self, session_id: str) -> InlineKeyboardMarkup:
        """Build inline keyboard with meal selection buttons (1, 2, 3, All 3)."""
        buttons = [
            [
                InlineKeyboardButton("1️⃣", callback_data=f"select:{session_id}:1"),
                InlineKeyboardButton("2️⃣", callback_data=f"select:{session_id}:2"),
                InlineKeyboardButton("3️⃣", callback_data=f"select:{session_id}:3"),
            ],
            [
                InlineKeyboardButton("All 3 🔥", callback_data=f"select:{session_id}:all"),
            ],
        ]
        return InlineKeyboardMarkup(buttons)

    # ==================== PUBLIC METHODS ====================

    def send_recipes(self, chat_id: int, recipes: dict, session_id: str) -> bool:
        """
        Send formatted recipe options with inline keyboard buttons.
        Returns True on success, False on failure.
        """
        text = llm_service.format_recipes_for_telegram(recipes)
        keyboard = self._build_recipe_keyboard(session_id)

        try:
            self._run_async(
                self._bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            )
            return True
        except Exception as e:
            print(f"Error sending recipes to chat {chat_id}: {e}")
            return False

    def _split_message(self, text: str, max_len: int = 4096) -> list[str]:
        """Split a long message into chunks that fit Telegram's 4096-char limit.

        Splits on double-newlines first, then single newlines, to keep
        logical sections together.
        """
        if len(text) <= max_len:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break

            # Try to split at a double newline within the limit
            split_at = remaining.rfind("\n\n", 0, max_len)
            if split_at == -1:
                # Fall back to single newline
                split_at = remaining.rfind("\n", 0, max_len)
            if split_at == -1:
                # Last resort: hard cut at max_len
                split_at = max_len

            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip("\n")

        return chunks

    def send_message(self, chat_id: int, text: str) -> bool:
        """Send a plain text message, splitting into chunks if over 4096 chars."""
        try:
            for chunk in self._split_message(text):
                self._run_async(
                    self._bot.send_message(chat_id=chat_id, text=chunk)
                )
            return True
        except Exception as e:
            print(f"Error sending message to chat {chat_id}: {e}")
            return False

    def send_grocery_list(self, chat_id: int, grocery_text: str) -> bool:
        """Send a formatted grocery list. Returns True on success."""
        try:
            self._run_async(
                self._bot.send_message(
                    chat_id=chat_id,
                    text=grocery_text,
                    parse_mode=None,  # plain text — grocery list may have special chars
                )
            )
            return True
        except Exception as e:
            print(f"Error sending grocery list to chat {chat_id}: {e}")
            return False

    def answer_callback_query(self, callback_query_id: str, text: str = None) -> bool:
        """Answer an inline keyboard callback (removes the loading spinner)."""
        try:
            self._run_async(
                self._bot.answer_callback_query(
                    callback_query_id=callback_query_id,
                    text=text,
                )
            )
            return True
        except Exception as e:
            print(f"Error answering callback query: {e}")
            return False

    def set_my_commands(self) -> bool:
        """Register slash commands with BotFather so they appear in the Telegram command menu."""
        commands = [
            BotCommand("start", "Meet your chef"),
            BotCommand("help", "What I can do"),
            BotCommand("menu", "What I can do"),
            BotCommand("favorites", "Your go-to dishes"),
            BotCommand("cancel", "Cancel pending meal options"),
        ]
        try:
            self._run_async(self._bot.set_my_commands(commands))
            print("Telegram bot commands registered successfully")
            return True
        except Exception as e:
            print(f"Error registering bot commands: {e}")
            return False


# Global instance for convenience
telegram_service = TelegramService()
