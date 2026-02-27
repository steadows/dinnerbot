from datetime import datetime, timedelta, timezone
from google.cloud import firestore
from config import config

# Initialize Firestore client
db = firestore.Client(project=config.GCP_PROJECT_ID)

# Collection names
SESSIONS_COLLECTION = "meal_sessions"
MEAL_HISTORY_COLLECTION = "meal_history"
CONVERSATION_COLLECTION = "conversation_history"

SESSION_EXPIRY_DAYS = 7
CONVERSATION_HISTORY_LIMIT = 10  # Keep last N messages for context
FAVORITE_THRESHOLD = 3  # Times selected to auto-mark as favorite


class DBService:
    def __init__(self):
        self.sessions = db.collection(SESSIONS_COLLECTION)
        self.meal_history = db.collection(MEAL_HISTORY_COLLECTION)
        self.conversations = db.collection(CONVERSATION_COLLECTION)

    # ==================== SESSION METHODS ====================
    
    def expire_pending_sessions(self, user_id: str) -> int:
        """
        Mark any existing pending sessions for this user as expired.
        Returns the count of expired sessions.
        """
        pending_sessions = (
            self.sessions
            .where("user_id", "==", user_id)
            .where("status", "==", "pending_selection")
            .stream()
        )
        
        count = 0
        for doc in pending_sessions:
            doc.reference.update({"status": "expired"})
            count += 1
        
        return count

    def create_session(self, user_id: str, options: dict, platform: str = "telegram", chat_id=None) -> str:
        """
        Create a new meal session in Firestore.
        Returns the document ID of the created session.
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=SESSION_EXPIRY_DAYS)
        
        doc_data = {
            "user_id": user_id,
            "chat_id": chat_id,
            "platform": platform,
            "created_at": now,
            "expires_at": expires_at,
            "status": "pending_selection",
            "options": options,
            "selected_option": None
        }
        
        doc_ref = self.sessions.add(doc_data)
        # .add() returns a tuple of (timestamp, document_reference)
        return doc_ref[1].id

    def get_session_by_id(self, doc_id: str):
        """
        Retrieve a session by its document ID.
        Returns the document snapshot or None if not found.
        """
        doc = self.sessions.document(doc_id).get()
        if doc.exists:
            return doc
        return None

    def get_pending_session(self, user_id: str):
        """
        Retrieve the pending session for a given user.
        Returns the document snapshot or None if not found.
        """
        sessions = (
            self.sessions
            .where("user_id", "==", user_id)
            .where("status", "==", "pending_selection")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        
        for doc in sessions:
            return doc
        
        return None

    def update_session_selection(self, doc_id: str, selected_option: str) -> None:
        """
        Update a session with the user's selection and mark as completed.
        """
        self.sessions.document(doc_id).update({
            "selected_option": selected_option,
            "status": "completed"
        })

    def mark_session_failed(self, doc_id: str, error_message: str) -> None:
        """
        Mark a session as failed (e.g., SMS send failure).
        """
        self.sessions.document(doc_id).update({
            "status": "send_failed",
            "error_message": error_message
        })

    def update_session_options(self, doc_id: str, updated_options: dict) -> None:
        """
        Update specific option(s) in a session's options dict.
        updated_options: dict of position->recipe to merge into existing options.
        Example: {"2": {new_recipe_dict}} replaces only option 2.
        """
        update_data = {}
        for position, recipe in updated_options.items():
            update_data[f"options.{position}"] = recipe
        self.sessions.document(doc_id).update(update_data)

    # ==================== MEAL HISTORY METHODS ====================

    def save_meal_to_history(self, user_id: str, recipe_name: str, ingredients: list) -> None:
        """
        Save a selected meal to history, incrementing times_selected if it exists.
        """
        # Use recipe name as document ID for easy lookup/update
        doc_id = f"{user_id}_{recipe_name.lower().replace(' ', '_')}"
        doc_ref = self.meal_history.document(doc_id)
        doc = doc_ref.get()
        
        now = datetime.now(timezone.utc)
        
        if doc.exists:
            # Increment times_selected
            current_data = doc.to_dict()
            times_selected = current_data.get("times_selected", 0) + 1
            doc_ref.update({
                "times_selected": times_selected,
                "last_selected_at": now,
                # Auto-favorite if selected enough times
                "is_favorite": times_selected >= FAVORITE_THRESHOLD
            })
        else:
            # Create new entry
            doc_ref.set({
                "user_id": user_id,
                "recipe_name": recipe_name,
                "ingredients": ingredients,
                "first_selected_at": now,
                "last_selected_at": now,
                "times_selected": 1,
                "is_favorite": False,
                "feedback": None,
            })

    def get_favorites(self, user_id: str) -> list:
        """
        Get favorite meals for a user.
        Favorites are meals marked as favorite OR selected >= FAVORITE_THRESHOLD times.
        """
        favorites = (
            self.meal_history
            .where("user_id", "==", user_id)
            .where("is_favorite", "==", True)
            .stream()
        )
        
        return [doc.to_dict() for doc in favorites]

    def get_recent_meals(self, user_id: str, limit: int = 5) -> list:
        """
        Get recently selected meals for a user.
        """
        meals = (
            self.meal_history
            .where("user_id", "==", user_id)
            .order_by("last_selected_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        
        return [doc.to_dict() for doc in meals]

    def toggle_favorite(self, user_id: str, recipe_name: str) -> bool:
        """
        Toggle the favorite status of a meal.
        Returns the new favorite status.
        """
        doc_id = f"{user_id}_{recipe_name.lower().replace(' ', '_')}"
        doc_ref = self.meal_history.document(doc_id)
        doc = doc_ref.get()
        
        if doc.exists:
            current = doc.to_dict().get("is_favorite", False)
            doc_ref.update({"is_favorite": not current})
            return not current
        
        return False

    # ==================== CONVERSATION HISTORY METHODS ====================
    
    def get_conversation_history(self, user_id: str) -> list:
        """
        Get recent conversation history for context.
        Returns list of message dicts: [{"role": "user/assistant", "content": "...", "timestamp": ...}]
        """
        doc = self.conversations.document(user_id).get()
        
        if doc.exists:
            data = doc.to_dict()
            messages = data.get("messages", [])
            # Return only the last N messages
            return messages[-CONVERSATION_HISTORY_LIMIT:]
        
        return []

    def append_to_conversation(self, user_id: str, role: str, content: str, metadata: dict = None) -> None:
        """
        Append a message to conversation history.
        role: "user" or "assistant"
        metadata: optional dict with intent, session_id, recipe name, etc.
        Also updates last_interaction_at for staleness tracking.
        """
        doc_ref = self.conversations.document(user_id)
        doc = doc_ref.get()
        
        now = datetime.now(timezone.utc)
        new_message = {
            "role": role,
            "content": content,
            "timestamp": now.isoformat()
        }
        if metadata:
            new_message["metadata"] = metadata
        
        if doc.exists:
            data = doc.to_dict()
            messages = data.get("messages", [])
            messages.append(new_message)
            # Keep only last N*2 messages to prevent unbounded growth
            messages = messages[-(CONVERSATION_HISTORY_LIMIT * 2):]
            doc_ref.update({
                "messages": messages,
                "updated_at": now,
                "last_interaction_at": now,
            })
        else:
            doc_ref.set({
                "user_id": user_id,
                "messages": [new_message],
                "updated_at": now,
                "last_interaction_at": now,
            })

    def get_last_interaction_time(self, user_id: str):
        """
        Get the last_interaction_at timestamp for a user.
        Returns a datetime or None if no history exists.
        """
        doc = self.conversations.document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("last_interaction_at")
        return None

    def set_pending_feedback(self, user_id: str, recipe_name: str) -> None:
        """
        Set pending_feedback on conversation doc — stores the recipe name
        awaiting feedback after a grocery list is sent.
        """
        doc_ref = self.conversations.document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.update({"pending_feedback": recipe_name})
        else:
            doc_ref.set({
                "user_id": user_id,
                "messages": [],
                "updated_at": datetime.now(timezone.utc),
                "last_interaction_at": datetime.now(timezone.utc),
                "pending_feedback": recipe_name,
            })

    def get_pending_feedback(self, user_id: str) -> str | None:
        """
        Get the pending_feedback recipe name for a user, if any.
        Returns the recipe name string or None.
        """
        doc = self.conversations.document(user_id).get()
        if doc.exists:
            return doc.to_dict().get("pending_feedback")
        return None

    def clear_pending_feedback(self, user_id: str) -> None:
        """Clear pending_feedback for a user."""
        doc_ref = self.conversations.document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.update({"pending_feedback": None})

    def update_meal_feedback(self, user_id: str, recipe_name: str, feedback: str) -> bool:
        """
        Update a meal_history document with feedback.
        feedback values: "loved", "okay", "skip_next_time"
        Returns True if the document was found and updated.
        """
        doc_id = f"{user_id}_{recipe_name.lower().replace(' ', '_')}"
        doc_ref = self.meal_history.document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            doc_ref.update({"feedback": feedback})
            return True
        return False

    def clear_conversation_history(self, user_id: str) -> None:
        """
        Clear conversation history for a user (e.g., on request or periodically).
        """
        self.conversations.document(user_id).delete()


# Global instance for convenience
db_service = DBService()
