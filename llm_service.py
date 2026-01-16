import json
import re
import google.generativeai as genai
from config import config
from user_profile import profile_service
from db_service import db_service

# Configure the Gemini client
genai.configure(api_key=config.GEMINI_API_KEY)

# Default fallback recipes if LLM fails (tailored to family profile)
DEFAULT_RECIPES = {
    "1": {
        "name": "Slow Cooker Pulled Chicken",
        "description": "Tender chicken with pureed carrots hidden in the sauce",
        "protein": "chicken thighs",
        "cook_method": "slow cooker",
        "time": "10 min prep",
        "ingredients": ["chicken thighs", "chicken broth", "carrots", "onion", "garlic", "paprika", "cumin", "olive oil", "salt"]
    },
    "2": {
        "name": "Air Fryer Meatballs",
        "description": "Juicy beef meatballs with hidden zucchini, toddler-approved",
        "protein": "ground beef",
        "cook_method": "air fryer",
        "time": "25 mins",
        "ingredients": ["ground beef", "zucchini", "egg", "parmesan", "garlic", "italian seasoning", "marinara sauce", "olive oil"]
    },
    "3": {
        "name": "Sheet Pan Chicken & Veggies",
        "description": "One-pan dinner with caramelized veggies kids love",
        "protein": "chicken breast",
        "cook_method": "oven",
        "time": "35 mins",
        "ingredients": ["chicken breast", "sweet potato", "broccoli", "bell pepper", "olive oil", "garlic powder", "paprika", "butter"]
    }
}

# System prompt for conversational context
SYSTEM_PROMPT = """You are Gordon Ramsay, acting as this family's private chef and meal planner.

Your personality:
- Passionate about good food and proper technique, but adapted for home cooks
- Direct and confident - you know what works
- Surprisingly warm when it comes to feeding families and kids
- You care deeply about nutrition, flavor, AND practicality
- You don't tolerate shortcuts that sacrifice flavor, but you respect that weeknights are busy

Your role:
- Plan delicious, high-protein meals tailored to this family
- Hide vegetables cleverly for the toddler (you're a master at this)
- Keep weeknight cooking SIMPLE - no fussy techniques, no food processor required
- Think about leftovers and meal efficiency

Communication style:
- Brief and punchy (this goes via SMS)
- Confident recommendations, not wishy-washy suggestions
- Occasional Gordon-isms ("Right, let's do this properly", "Beautiful", "Lovely")
- Supportive but no-nonsense

{profile_context}

Recent conversation:
{conversation_context}
"""

RECIPE_GENERATION_PROMPT = """You're Gordon Ramsay planning this week's family dinners. Make it delicious, make it practical.

FAMILY PROFILE:
{profile_context}

{recent_meals_context}

REQUIREMENTS:
- Under 45 minutes (or slow cooker set-and-forget)
- High protein, hidden veggies for the toddler
- NO wheat/bread, NO mushrooms, NO olives, NO seed oils
- Rice and couscous are fine as sides
- Meals with grain free tortillas are fine
- Portions for 2 adults + toddler + leftovers
- SIMPLE weeknight cooking - no blending, no ricing cauliflower, no fussy prep
- Good sauces, savory flavors, cheese when it fits
- Equipment: slow cooker, air fryer, oven, stovetop

Give me 3 proper dinner options. For each:
- Sneak the veg in cleverly (finely diced into sauce, grated into meatballs, etc.)
- Big on protein and flavor
- Reheats beautifully for lunch

Return ONLY valid JSON, no markdown, no code blocks:
{{
    "1": {{"name": "Recipe Name", "description": "Gordon's pitch + hidden veggie trick", "protein": "main protein", "cook_method": "method", "time": "XX mins", "ingredients": ["ingredient1", "ingredient2", ...]}},
    "2": {{"name": "Recipe Name", "description": "Gordon's pitch + hidden veggie trick", "protein": "main protein", "cook_method": "method", "time": "XX mins", "ingredients": ["ingredient1", "ingredient2", ...]}},
    "3": {{"name": "Recipe Name", "description": "Gordon's pitch + hidden veggie trick", "protein": "main protein", "cook_method": "method", "time": "XX mins", "ingredients": ["ingredient1", "ingredient2", ...]}}
}}
"""

RETRY_PROMPT = """Return ONLY valid JSON, no markdown, no code blocks.
3 Gordon Ramsay-style family dinners (high protein, no wheat/mushrooms/olives/seed oils, under 45 min, hidden veggies, simple prep):
{{"1": {{"name": "Name", "description": "brief desc", "protein": "protein", "cook_method": "method", "time": "XX mins", "ingredients": ["item1", "item2"]}}, "2": {{"name": "Name", "description": "brief desc", "protein": "protein", "cook_method": "method", "time": "XX mins", "ingredients": ["item1", "item2"]}}, "3": {{"name": "Name", "description": "brief desc", "protein": "protein", "cook_method": "method", "time": "XX mins", "ingredients": ["item1", "item2"]}}}}
"""


class LLMService:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = 2

    def _parse_json_response(self, text: str) -> dict:
        """
        Attempt to parse JSON from the LLM response.
        Handles common issues like markdown code blocks.
        """
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        
        return json.loads(text)

    def _validate_recipe_structure(self, data: dict) -> bool:
        """Validate that the response has the expected structure."""
        if not isinstance(data, dict):
            return False
        
        for key in ["1", "2", "3"]:
            if key not in data:
                return False
            recipe = data[key]
            if not isinstance(recipe, dict):
                return False
            # Required fields
            if "name" not in recipe or "ingredients" not in recipe:
                return False
            if not isinstance(recipe["ingredients"], list):
                return False
        
        return True

    def format_recipes_for_sms(self, recipes: dict) -> str:
        """Format recipe options for SMS display."""
        lines = ["This week's menu from your chef:\n"]
        
        for key in ["1", "2", "3"]:
            recipe = recipes[key]
            name = recipe["name"]
            desc = recipe.get("description", "")
            time = recipe.get("time", "")
            protein = recipe.get("protein", "")
            method = recipe.get("cook_method", "")
            
            line = f"{key}. {name}"
            details = []
            if time:
                details.append(time)
            if method:
                details.append(method)
            if protein:
                details.append(protein)
            
            if details:
                line += f" ({', '.join(details)})"
            
            lines.append(line)
            if desc:
                lines.append(f"   {desc}")
        
        lines.append("\nReply 1, 2, or 3 to select.")
        return "\n".join(lines)

    def _build_conversation_context(self, phone_number: str) -> str:
        """Build conversation history string for context."""
        history = db_service.get_conversation_history(phone_number)
        
        if not history:
            return "No previous conversation."
        
        context_lines = []
        for msg in history[-6:]:  # Last 6 messages for context
            role = "User" if msg["role"] == "user" else "DinnerBot"
            context_lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_lines)

    def _get_recent_meals_context(self, phone_number: str) -> str:
        """Get recent meals to avoid repetition."""
        recent = db_service.get_recent_meals(phone_number, limit=5)
        
        if not recent:
            return ""
        
        meal_names = [m["recipe_name"] for m in recent]
        return f"Recently made meals (try to suggest different ones): {', '.join(meal_names)}"

    def detect_intent(self, message: str) -> str:
        """
        Detect user intent from message.
        Returns: "selection", "favorites", "help", "conversational"
        """
        message_lower = message.lower().strip()
        
        # Check for meal selection (1, 2, 3, or "option 1", etc.)
        if re.match(r'^[123]$', message_lower):
            return "selection"
        if re.match(r'^(option\s*)?[123]$', message_lower):
            return "selection"
        
        # Check for favorites request
        favorites_keywords = ["favorites", "favourite", "fav", "favs", "my favorites", "show favorites"]
        if any(kw in message_lower for kw in favorites_keywords):
            return "favorites"
        
        # Check for help request
        help_keywords = ["help", "commands", "what can you do", "options"]
        if any(kw in message_lower for kw in help_keywords):
            return "help"
        
        # Default to conversational
        return "conversational"

    def extract_selection(self, message: str) -> str:
        """Extract the selection number from a message."""
        message_lower = message.lower().strip()
        match = re.search(r'[123]', message_lower)
        if match:
            return match.group()
        return None

    def generate_weekly_recipes(self, phone_number: str) -> dict:
        """
        Generate 3 dinner recipe options using Gemini.
        Uses family profile for personalization.
        """
        profile_context = profile_service.format_profile_for_prompt(phone_number)
        recent_meals_context = self._get_recent_meals_context(phone_number)
        
        prompt = RECIPE_GENERATION_PROMPT.format(
            profile_context=profile_context,
            recent_meals_context=recent_meals_context
        )
        
        prompts = [prompt, RETRY_PROMPT]
        
        for attempt, p in enumerate(prompts):
            try:
                response = self.model.generate_content(p)
                
                if not response.text:
                    print(f"Attempt {attempt + 1}: Empty response from Gemini")
                    continue
                
                data = self._parse_json_response(response.text)
                
                if self._validate_recipe_structure(data):
                    return data
                else:
                    print(f"Attempt {attempt + 1}: Invalid recipe structure")
                    
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}: JSON parsing failed - {e}")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error calling Gemini - {e}")
        
        print("All retries exhausted, using fallback recipes")
        return DEFAULT_RECIPES

    def generate_grocery_list(self, phone_number: str, recipe_name: str, ingredients: list) -> str:
        """
        Generate a formatted grocery list for a selected recipe.
        Uses family profile for portion sizing.
        """
        prompt = f"""You're Gordon Ramsay. Give me a proper shopping list for "{recipe_name}".

Base ingredients: {', '.join(ingredients)}

REQUIREMENTS:
- Portions for 2 adults + 1 toddler + generous leftovers
- Butter, ghee, olive oil, avocado oil, or coconut oil only (NO seed oils)
- Proper quantities - don't be stingy
- Group by store section

Format:

PROTEIN:
- 2 lbs chicken thighs

PRODUCE:
- 3 carrots
- 1 head garlic

DAIRY:
- butter

PANTRY:
- olive oil

Keep it tight for SMS. Just the list, no waffle."""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return f"Right, here's your list for {recipe_name}:\n\n{response.text.strip()}"
        except Exception as e:
            print(f"Error generating grocery list: {e}")
        
        return f"Shopping list for {recipe_name}:\n" + "\n".join(f"- {item}" for item in ingredients)

    def get_favorites_response(self, phone_number: str) -> str:
        """Generate a formatted response listing user's favorite meals."""
        favorites = db_service.get_favorites(phone_number)
        
        if not favorites:
            return "No favorites yet! Make a few dishes and they'll show up here. Now get cooking!"
        
        response_lines = ["Your go-to dishes:"]
        for i, fav in enumerate(favorites[:5], 1):  # Limit to 5 for SMS
            times = fav.get("times_selected", 1)
            response_lines.append(f"{i}. {fav['recipe_name']} ({times}x)")
        
        response_lines.append("\nReply with a number for that shopping list. Let's go!")
        return "\n".join(response_lines)

    def get_help_response(self) -> str:
        """Return help text explaining available commands."""
        return """Right, here's how this works:

Reply 1, 2, or 3 - Pick your dinner
"favorites" - Your greatest hits
"help" - This message

I send you 3 proper dinner options each week. You pick one, I give you the shopping list. Simple. Beautiful. Let's cook!"""

    def handle_conversational_message(self, phone_number: str, message: str) -> str:
        """
        Handle free-form conversational messages using Gemini.
        Maintains context awareness.
        """
        profile_context = profile_service.format_profile_for_prompt(phone_number)
        conversation_context = self._build_conversation_context(phone_number)
        
        system_prompt = SYSTEM_PROMPT.format(
            profile_context=profile_context,
            conversation_context=conversation_context
        )
        
        # Check if there's a pending session for additional context
        session = db_service.get_pending_session(phone_number)
        session_context = ""
        if session:
            data = session.to_dict()
            options = data.get("options", {})
            options_text = "\n".join([f"{k}. {v['name']}" for k, v in options.items()])
            session_context = f"\n\nCurrent meal options waiting for selection:\n{options_text}"
        
        full_prompt = f"""{system_prompt}{session_context}

User message: {message}

Respond as their executive chef - confident, helpful, brief (SMS format). If they're asking about food/meals, give practical advice. If unclear, guide them to select a meal or ask for help."""

        try:
            response = self.model.generate_content(full_prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error in conversational response: {e}")
        
        return "Apologies, I'm having a moment. Reply 'help' for options, or pick 1, 2, or 3 if you have a menu waiting!"


# Global instance for convenience
llm_service = LLMService()
