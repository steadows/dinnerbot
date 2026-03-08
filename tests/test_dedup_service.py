"""Tests for dedup_service — Telegram update_id deduplication."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import AlreadyExists

from dedup_service import DEDUP_COLLECTION, DedupService


@pytest.fixture()
def mock_firestore():
    """Patch Firestore client and return mock collection."""
    with patch("dedup_service.firestore") as mock_fs:
        mock_db = MagicMock()
        mock_fs.Client.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        yield mock_collection


@pytest.fixture()
def dedup(mock_firestore) -> DedupService:
    """Create a DedupService with mocked Firestore."""
    return DedupService()


class TestIsDuplicate:
    """Tests for the is_duplicate method."""

    def test_first_occurrence_returns_false(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        mock_doc_ref = MagicMock()
        mock_firestore.document.return_value = mock_doc_ref
        mock_doc_ref.create.return_value = None  # success

        result = dedup.is_duplicate(12345)

        assert result is False
        mock_firestore.document.assert_called_once_with("12345")
        mock_doc_ref.create.assert_called_once()

    def test_duplicate_returns_true(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        mock_doc_ref = MagicMock()
        mock_firestore.document.return_value = mock_doc_ref
        mock_doc_ref.create.side_effect = AlreadyExists("exists")

        result = dedup.is_duplicate(12345)

        assert result is True

    def test_stores_update_id_and_timestamp(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        mock_doc_ref = MagicMock()
        mock_firestore.document.return_value = mock_doc_ref

        dedup.is_duplicate(99999)

        create_call = mock_doc_ref.create.call_args
        data = create_call[0][0]
        assert data["update_id"] == 99999
        assert isinstance(data["processed_at"], datetime)

    def test_different_ids_not_duplicate(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        mock_doc_ref = MagicMock()
        mock_firestore.document.return_value = mock_doc_ref
        mock_doc_ref.create.return_value = None

        assert dedup.is_duplicate(111) is False
        assert dedup.is_duplicate(222) is False

        assert mock_firestore.document.call_count == 2
        mock_firestore.document.assert_any_call("111")
        mock_firestore.document.assert_any_call("222")

    def test_document_id_is_string(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        mock_doc_ref = MagicMock()
        mock_firestore.document.return_value = mock_doc_ref

        dedup.is_duplicate(42)

        mock_firestore.document.assert_called_with("42")


class TestCleanupOld:
    """Tests for the cleanup_old method."""

    def test_deletes_old_documents(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        old_doc_1 = MagicMock()
        old_doc_2 = MagicMock()
        mock_query = MagicMock()
        mock_firestore.where.return_value = mock_query
        mock_query.stream.return_value = [old_doc_1, old_doc_2]

        deleted = dedup.cleanup_old(max_age_hours=24)

        assert deleted == 2
        old_doc_1.reference.delete.assert_called_once()
        old_doc_2.reference.delete.assert_called_once()

    def test_no_old_documents(self, dedup: DedupService, mock_firestore: MagicMock) -> None:
        mock_query = MagicMock()
        mock_firestore.where.return_value = mock_query
        mock_query.stream.return_value = []

        deleted = dedup.cleanup_old(max_age_hours=24)

        assert deleted == 0

    def test_collection_name(self, mock_firestore: MagicMock) -> None:
        with patch("dedup_service.firestore") as mock_fs:
            mock_db = MagicMock()
            mock_fs.Client.return_value = mock_db
            DedupService()
            mock_db.collection.assert_called_with(DEDUP_COLLECTION)
