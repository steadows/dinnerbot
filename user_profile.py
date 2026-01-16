from datetime import datetime, timezone
from google.cloud import firestore
from config import config

# Initialize Firestore client
db = firestore.Client(project=config.GCP_PROJECT_ID)

PROFILES_COLLECTION = "user_profiles"

# Default profile configuration - customize for your family
DEFAULT_PROFILE = {
    "family_size": "2 adults and 1 toddler (2 years old)",
    "dietary_restrictions": [
        "no wheat/bread products (rice and couscous are fine)",
        "no seed oils (use butter, olive oil, avocado oil, or coconut oil)",
        "high protein focus"
    ],
    "cuisine_preferences": [
        "high protein meals",
        "toddler-friendly with hidden vegetables",
        "meals that reheat well for leftovers",
        "savory flavors, good sauces, cheese when it fits"
    ],
    "disliked_ingredients": ["mushrooms", "olives"],
    "portion_size": "large",  # Enough for leftovers for lunches
    "logistics": {
        "max_cook_time_minutes": 45,
        "skill_level": "easy",  # Quick and approachable
        "available_equipment": ["slow cooker/crockpot", "air fryer", "oven", "stovetop"],
        "weeknight_preference": True,
    },
    # Special considerations
    "special_notes": [
        "Hide vegetables creatively so the toddler gets micronutrients without noticing",
        "Portions should be generous enough for lunch leftovers the next day",
        "Prefer one-pot or simple cleanup meals when possible",
        "NO labor-intensive techniques on weeknights (no ricing cauliflower, no blending, etc.)",
        "Rice, mashed potatoes, simple sides are great - keep it practical"
    ]
}


class UserProfileService:
    def __init__(self):
        self.collection = db.collection(PROFILES_COLLECTION)

    def get_profile(self, phone_number: str) -> dict:
        """
        Get user profile, merging Firestore overrides with defaults.
        Returns the merged profile.
        """
        # Start with defaults
        profile = DEFAULT_PROFILE.copy()
        profile["logistics"] = DEFAULT_PROFILE["logistics"].copy()
        
        # Try to get Firestore overrides
        doc = self.collection.document(phone_number).get()
        
        if doc.exists:
            firestore_data = doc.to_dict()
            # Merge top-level fields
            for key in ["family_size", "dietary_restrictions", "cuisine_preferences", 
                        "disliked_ingredients", "portion_size"]:
                if key in firestore_data and firestore_data[key]:
                    profile[key] = firestore_data[key]
            
            # Merge logistics dict
            if "logistics" in firestore_data and firestore_data["logistics"]:
                for key, value in firestore_data["logistics"].items():
                    if value is not None:
                        profile["logistics"][key] = value
        
        return profile

    def update_profile(self, phone_number: str, updates: dict) -> None:
        """
        Update user profile in Firestore.
        Only updates the fields provided, preserving others.
        """
        updates["updated_at"] = datetime.now(timezone.utc)
        updates["phone_number"] = phone_number
        
        self.collection.document(phone_number).set(updates, merge=True)

    def format_profile_for_prompt(self, phone_number: str) -> str:
        """
        Format the user profile as a string for inclusion in LLM prompts.
        """
        profile = self.get_profile(phone_number)
        
        parts = [f"Family: {profile['family_size']}"]
        parts.append(f"Portion size: {profile['portion_size']} (enough for leftovers)")
        
        if profile.get("dietary_restrictions"):
            parts.append(f"Dietary requirements: {', '.join(profile['dietary_restrictions'])}")
        
        if profile.get("cuisine_preferences"):
            parts.append(f"Meal priorities: {', '.join(profile['cuisine_preferences'])}")
        
        if profile.get("disliked_ingredients"):
            parts.append(f"NEVER include: {', '.join(profile['disliked_ingredients'])}")
        
        logistics = profile.get("logistics", {})
        parts.append(f"Max prep + cook time: {logistics.get('max_cook_time_minutes', 45)} minutes")
        parts.append(f"Difficulty: {logistics.get('skill_level', 'easy')} recipes only")
        
        if logistics.get("available_equipment"):
            parts.append(f"Kitchen equipment: {', '.join(logistics['available_equipment'])}")
        
        if profile.get("special_notes"):
            parts.append("\nSpecial instructions:")
            for note in profile["special_notes"]:
                parts.append(f"- {note}")
        
        return "\n".join(parts)


# Global instance for convenience
profile_service = UserProfileService()
