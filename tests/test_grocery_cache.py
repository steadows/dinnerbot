"""Tests for grocery list caching and multi-meal retrieval logic."""

from unittest.mock import MagicMock, patch

import pytest


class TestGroceryCacheDB:
    """Tests for db_service grocery cache methods.

    These test the methods directly by mocking the Firestore collection
    on the already-instantiated db_service singleton.
    """

    def test_cache_grocery_list(self) -> None:
        from db_service import db_service

        mock_doc_ref = MagicMock()
        with patch.object(db_service.sessions, "document", return_value=mock_doc_ref):
            db_service.cache_grocery_list("session-123", "PROTEIN:\n- 2 lbs chicken")

            mock_doc_ref.update.assert_called_once_with(
                {"cached_grocery_list": "PROTEIN:\n- 2 lbs chicken"}
            )

    def test_get_cached_grocery_list_exists(self) -> None:
        from db_service import db_service

        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {"cached_grocery_list": "PROTEIN:\n- 2 lbs chicken"}

        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc

        with patch.object(db_service.sessions, "document", return_value=mock_doc_ref):
            result = db_service.get_cached_grocery_list("session-123")

        assert result == "PROTEIN:\n- 2 lbs chicken"

    def test_get_cached_grocery_list_no_field(self) -> None:
        from db_service import db_service

        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {"status": "completed"}

        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc

        with patch.object(db_service.sessions, "document", return_value=mock_doc_ref):
            result = db_service.get_cached_grocery_list("session-123")

        assert result is None

    def test_get_cached_grocery_list_no_doc(self) -> None:
        from db_service import db_service

        mock_doc = MagicMock()
        mock_doc.exists = False

        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc

        with patch.object(db_service.sessions, "document", return_value=mock_doc_ref):
            result = db_service.get_cached_grocery_list("session-123")

        assert result is None


class TestHandleGroceryListRouting:
    """Tests for _handle_grocery_list routing logic in main.py."""

    @patch("main.telegram_service")
    @patch("main.db_service")
    @patch("main.llm_service")
    def test_serves_cached_list(
        self, mock_llm: MagicMock, mock_db: MagicMock, mock_telegram: MagicMock
    ) -> None:
        """When a cached grocery list exists, serve it directly without calling Gemini."""
        mock_session = MagicMock()
        mock_session.id = "session-1"
        mock_session.to_dict.return_value = {
            "cached_grocery_list": "PROTEIN:\n- 2 lbs chicken",
            "selected_option": "1",
            "options": {},
        }
        mock_db.get_most_recent_completed_session.return_value = mock_session

        from main import _handle_grocery_list

        _handle_grocery_list(chat_id=123, user_id="user-1", text="shopping list")

        mock_telegram.send_grocery_list.assert_called_once_with(
            123, "PROTEIN:\n- 2 lbs chicken"
        )
        mock_llm.generate_grocery_list.assert_not_called()
        mock_llm.generate_combined_grocery_list.assert_not_called()

    @patch("main._generate_and_send_combined_grocery_list")
    @patch("main.telegram_service")
    @patch("main.db_service")
    def test_regenerates_combined_for_all_selection(
        self, mock_db: MagicMock, mock_telegram: MagicMock, mock_gen_combined: MagicMock
    ) -> None:
        """When selection was 'all' and no cache, regenerate combined list."""
        mock_session = MagicMock()
        mock_session.id = "session-1"
        mock_session.to_dict.return_value = {
            "selected_option": "all",
            "options": {
                "1": {"name": "Meal A", "ingredients": ["chicken"]},
                "2": {"name": "Meal B", "ingredients": ["beef"]},
                "3": {"name": "Meal C", "ingredients": ["salmon"]},
            },
        }
        mock_db.get_most_recent_completed_session.return_value = mock_session

        from main import _handle_grocery_list

        _handle_grocery_list(chat_id=123, user_id="user-1", text="shopping list")

        mock_gen_combined.assert_called_once()
        call_args = mock_gen_combined.call_args
        recipes = call_args[0][2]
        assert len(recipes) == 3
        assert recipes[0]["name"] == "Meal A"
        assert recipes[2]["name"] == "Meal C"

    @patch("main._generate_and_send_grocery_list")
    @patch("main.telegram_service")
    @patch("main.db_service")
    def test_regenerates_single_for_numbered_selection(
        self, mock_db: MagicMock, mock_telegram: MagicMock, mock_gen_single: MagicMock
    ) -> None:
        """When selection was a single meal and no cache, regenerate for that meal."""
        mock_session = MagicMock()
        mock_session.id = "session-1"
        mock_session.to_dict.return_value = {
            "selected_option": "2",
            "options": {
                "1": {"name": "Meal A", "ingredients": ["chicken"]},
                "2": {"name": "Meal B", "ingredients": ["beef", "carrots"]},
                "3": {"name": "Meal C", "ingredients": ["salmon"]},
            },
        }
        mock_db.get_most_recent_completed_session.return_value = mock_session

        from main import _handle_grocery_list

        _handle_grocery_list(chat_id=123, user_id="user-1", text="shopping list")

        mock_gen_single.assert_called_once()
        call_args = mock_gen_single.call_args
        assert call_args[0][2] == "Meal B"
        assert call_args[0][3] == ["beef", "carrots"]

    @patch("main.telegram_service")
    @patch("main.db_service")
    def test_falls_back_to_meal_history(
        self, mock_db: MagicMock, mock_telegram: MagicMock
    ) -> None:
        """When no completed session exists, check meal history."""
        mock_db.get_most_recent_completed_session.return_value = None
        mock_db.get_recent_meals.return_value = []
        mock_db.get_pending_session.return_value = None

        from main import _handle_grocery_list

        _handle_grocery_list(chat_id=123, user_id="user-1", text="shopping list")

        mock_telegram.send_message.assert_called()
        msg = mock_telegram.send_message.call_args[0][1]
        assert "plan dinner" in msg.lower() or "no meals" in msg.lower()

    @patch("main.telegram_service")
    @patch("main.db_service")
    def test_prompts_pending_session(
        self, mock_db: MagicMock, mock_telegram: MagicMock
    ) -> None:
        """When no completed session but pending exists, prompt user to pick."""
        mock_db.get_most_recent_completed_session.return_value = None
        mock_db.get_recent_meals.return_value = []
        mock_pending = MagicMock()
        mock_db.get_pending_session.return_value = mock_pending

        from main import _handle_grocery_list

        _handle_grocery_list(chat_id=123, user_id="user-1", text="shopping list")

        msg = mock_telegram.send_message.call_args[0][1]
        assert "pick" in msg.lower() or "options" in msg.lower()
