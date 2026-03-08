"""
Telegram webhook deduplication service backed by Firestore.

Uses atomic document creation to prevent processing the same
Telegram update_id more than once. Firestore's create() method
fails with AlreadyExists if the document exists, providing
lock-free, race-condition-safe deduplication.
"""

from datetime import datetime, timedelta, timezone

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore


DEDUP_COLLECTION = "processed_updates"


class DedupService:
    """Firestore-backed Telegram update_id deduplication."""

    def __init__(self) -> None:
        self._db = firestore.Client()
        self._collection = self._db.collection(DEDUP_COLLECTION)

    def is_duplicate(self, update_id: int) -> bool:
        """Check if an update_id has already been processed.

        Uses atomic create() — if the document already exists,
        Firestore raises AlreadyExists, meaning it's a duplicate.

        Args:
            update_id: The Telegram update_id to check.

        Returns:
            True if this update_id was already processed (duplicate).
            False if this is the first time seeing it (now marked as processed).
        """
        doc_ref = self._collection.document(str(update_id))

        try:
            doc_ref.create({
                "processed_at": datetime.now(timezone.utc),
                "update_id": update_id,
            })
            return False  # First time — not a duplicate
        except AlreadyExists:
            return True  # Already processed — duplicate

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Delete processed_updates documents older than max_age_hours.

        Called opportunistically or by a scheduled job.
        Firestore TTL policy is preferred for production use.

        Args:
            max_age_hours: Documents older than this are deleted.

        Returns:
            Number of documents deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        old_docs = (
            self._collection
            .where(filter=firestore.FieldFilter("processed_at", "<", cutoff))
            .stream()
        )

        deleted = 0
        for doc in old_docs:
            doc.reference.delete()
            deleted += 1

        if deleted:
            print(f"Dedup cleanup: deleted {deleted} documents older than {max_age_hours}h")

        return deleted


# Global instance
dedup_service = DedupService()
