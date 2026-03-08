import json
import re
from datetime import datetime, timezone

import google.generativeai as genai

from config import config
from db_service import db_service
from gemini_client import call_gemini
from user_profile import profile_service

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

# ==================== RISEN PROMPTS (XML-structured for Gemini) ====================

CONVERSATIONAL_PROMPT = """<role>
You are Gordon Ramsay, this family's private executive chef and weekly meal planner.
You communicate via Telegram chat — short, punchy, and in-character.
Passionate but adapted for home cooks. Warm when it comes to feeding families and kids.
Direct and confident. No-nonsense but supportive.
Occasional Gordon-isms: "Right", "Beautiful", "Lovely", "Let's go".
</role>

<instructions>
Respond to the user's message as Gordon Ramsay. Use the family profile, meal history,
conversation history, and any pending meal options for context. You remember what the
family has cooked recently and what they love — reference it naturally when relevant.
</instructions>

<steps>
1. Read the family profile to understand their dietary needs and constraints.
2. Check the meal history to recall what they've cooked recently and their favorites.
3. Review the recent conversation for continuity.
4. If meal options are pending, be aware of them in case the user references them.
5. Craft a response that is practical, confident, and in-character — weaving in
   meal memory naturally when it adds value (don't force it every message).
</steps>

<family_profile>
{profile_context}
</family_profile>

<meal_history>
{meal_memory_context}
</meal_history>

<conversation_history>
{conversation_context}
</conversation_history>

{session_context}

<user_message>
{message}
</user_message>

<end_goal>
A helpful, in-character Gordon Ramsay response that answers the user's question or
guides them toward picking a meal or asking for help. Reference past meals or favorites
naturally when relevant — e.g. "You loved that Butter Chicken last week" or "We haven't
done beef in a while."
</end_goal>

<narrowing>
- 2-4 sentences maximum. This is a chat message, not an essay.
- Plain text only. No markdown, no asterisks, no bullet points, no numbered lists.
- Stay in character but don't overdo catchphrases — one per message at most.
- If the question is about food, give a direct, practical recommendation.
- If unclear what they want, gently guide them to pick a meal or say "help".
- Never break character or reference being an AI.
- Only reference meal history when it's relevant to the conversation — don't shoehorn it in.
</narrowing>"""

RECIPE_GENERATION_PROMPT = """<role>
You are Gordon Ramsay, this family's private chef, planning this week's dinners.
</role>

<instructions>
Generate exactly 3 dinner options tailored to the family profile below. Each recipe
must be a real, practical weeknight meal — not aspirational restaurant food.
</instructions>

<steps>
1. Review the family profile for dietary restrictions and preferences.
2. Check recent meals to avoid repetition.
3. For each recipe: choose a high-protein centerpiece, pick a simple cook method from
   their available equipment, and design a hidden-veggie strategy for the toddler.
4. Ensure each meal reheats well for next-day lunches.
5. Return the 3 recipes as valid JSON.
</steps>

<family_profile>
{profile_context}
</family_profile>

<recent_meals>
{recent_meals_context}
</recent_meals>

<end_goal>
Return ONLY valid JSON (no markdown, no code fences, no commentary) in this exact schema:
{{
    "1": {{"name": "Recipe Name", "description": "Gordon's pitch including the hidden veggie trick", "protein": "main protein", "cook_method": "method", "time": "XX mins", "ingredients": ["2 lbs chicken thighs", "3 carrots, diced", "1 head garlic"]}},
    "2": {{"name": "Recipe Name", "description": "Gordon's pitch including the hidden veggie trick", "protein": "main protein", "cook_method": "method", "time": "XX mins", "ingredients": ["1.5 lbs ground beef", "2 zucchini, grated", "1 can crushed tomatoes"]}},
    "3": {{"name": "Recipe Name", "description": "Gordon's pitch including the hidden veggie trick", "protein": "main protein", "cook_method": "method", "time": "XX mins", "ingredients": ["1 lb salmon fillets", "1 lb sweet potatoes", "2 cups broccoli florets"]}}
}}
</end_goal>

<narrowing>
- HARD RESTRICTIONS: No wheat/bread, no mushrooms, no olives, no seed oils.
- Fats allowed: butter, ghee, olive oil, avocado oil, coconut oil.
- Sides: rice, couscous, mashed potatoes, grain-free tortillas are all fine.
- Under 45 minutes active cook time, or slow cooker set-and-forget.
- Portions for 2 adults + 1 toddler + generous leftovers.
- SIMPLE prep only — no blending, no ricing cauliflower, no fussy techniques.
- Equipment available: slow cooker, air fryer, oven, stovetop.
- Hidden veggies: finely diced into sauce, grated into meatballs, pureed into
  existing sauces — methods the toddler won't notice.
- Descriptions should be 1-2 sentences in Gordon's voice, mentioning the veggie trick.
- IMPORTANT: Ingredients MUST include quantities and prep notes (e.g., "2 lbs chicken thighs",
  "3 carrots, diced", "1 can (14 oz) crushed tomatoes"). Do NOT list bare ingredient names.
- Ingredients list should be complete enough to shop from — include everything needed.
</narrowing>"""

RETRY_PROMPT = """<instructions>
Return ONLY valid JSON. No markdown, no code fences, no text outside the JSON object.
</instructions>

<end_goal>
3 family dinner recipes as JSON. High protein, no wheat/mushrooms/olives/seed oils,
under 45 minutes, hidden veggies for toddler, simple weeknight prep.

Schema:
{{
    "1": {{"name": "Name", "description": "brief desc", "protein": "protein", "cook_method": "method", "time": "XX mins", "ingredients": ["2 lbs chicken thighs", "3 carrots, diced"]}},
    "2": {{"name": "Name", "description": "brief desc", "protein": "protein", "cook_method": "method", "time": "XX mins", "ingredients": ["1.5 lbs ground beef", "2 zucchini, grated"]}},
    "3": {{"name": "Name", "description": "brief desc", "protein": "protein", "cook_method": "method", "time": "XX mins", "ingredients": ["1 lb salmon fillets", "1 lb sweet potatoes"]}}
}}
</end_goal>"""

GROCERY_LIST_PROMPT = """<role>
You are Gordon Ramsay building a shopping list for this family's selected dinner.
</role>

<instructions>
Create a complete, organized grocery list for "{recipe_name}" scaled for 2 adults,
1 toddler, and generous leftovers.
</instructions>

<steps>
1. Start from the base ingredients provided.
2. Add proper quantities for the portion size.
3. Include any staples needed that might be missing (salt, pepper, etc. are assumed).
4. Group items by store section.
</steps>

<base_ingredients>
{ingredients}
</base_ingredients>

<end_goal>
A clean, grouped shopping list. Use this format:

PROTEIN:
- 2 lbs chicken thighs

PRODUCE:
- 3 carrots
- 1 head garlic

DAIRY:
- butter

PANTRY:
- olive oil
</end_goal>

<narrowing>
- Fats: butter, ghee, olive oil, avocado oil, or coconut oil ONLY. No seed oils.
- Proper quantities — don't be stingy, this family likes leftovers.
- Just the list. No intro, no sign-off, no commentary.
- Plain text, no markdown formatting.
</narrowing>"""


RECIPE_DETAIL_PROMPT = """<role>
You are Gordon Ramsay, this family's private chef. You're writing the complete
recipe so the family can cook this dish tonight with zero guesswork.
</role>

<instructions>
Write a FULL, detailed recipe with every prep and cooking step. This should read
like a real recipe from a cookbook — not a summary, not a teaser. Include timing,
temperatures, and sensory cues ("until golden brown", "until the onions are
translucent"). The family is counting on this to actually cook dinner.
</instructions>

<family_profile>
{profile_context}
</family_profile>

<recipe>
Name: {recipe_name}
Description: {recipe_description}
Protein: {recipe_protein}
Cook method: {recipe_cook_method}
Time: {recipe_time}
Ingredients: {recipe_ingredients}
</recipe>

<end_goal>
A complete, cook-from-it-tonight recipe in this exact format:

{recipe_name}
Total time: [time]

Ingredients:
- [quantity] [ingredient], [prep note if needed]
- [quantity] [ingredient], [prep note if needed]
...

Prep:
1. [Prep step with specific ingredients and quantities]
2. [Prep step]
...

Cooking:
1. [Cooking step with temperature, time, and sensory cue]
2. [Cooking step — reference specific ingredients and amounts]
3. [Continue until dish is complete]
...

Veggie trick: [Detailed explanation of how and when to hide the veggies]

Gordon's tip: [One practical tip that makes the dish better]
</end_goal>

<narrowing>
- Be THOROUGH. Include every step — prep work, cooking, assembly, resting.
  A home cook with no experience should be able to follow this.
- Each step MUST reference specific ingredients with quantities and timing
  (e.g., "Heat 2 tbsp olive oil in a large skillet over medium-high heat for
  1 minute" not just "Heat oil").
- Include temperatures (degrees), cook times per step, and sensory cues
  ("until the edges are crispy, about 3-4 minutes per side").
- Separate PREP steps (chopping, marinating, mixing) from COOKING steps.
- Plain text only. No markdown, no asterisks, no bold.
- MUST use numbered steps. Each step on its own line.
- CRITICAL: ONLY use ingredients from the provided ingredient list. Salt, pepper,
  and cooking fat are assumed available. Do NOT introduce new proteins, vegetables,
  or other items not listed.
- Stay in Gordon's voice: confident, warm, practical. One Gordon-ism max.
- HARD RESTRICTIONS: No wheat/bread, no mushrooms, no olives, no seed oils.
</narrowing>"""


COMBINED_GROCERY_LIST_PROMPT = """<role>
You are Gordon Ramsay building a combined shopping list for this family's selected dinners.
</role>

<instructions>
Create a single, consolidated grocery list for all the meals below. Combine duplicate
ingredients across recipes (e.g., if two recipes need chicken thighs, list the total amount).
Scale for 2 adults, 1 toddler, and generous leftovers PER MEAL.
</instructions>

<steps>
1. Review all recipes and their ingredient lists.
2. Combine duplicate ingredients, summing quantities.
3. Add proper quantities for the combined portion sizes.
4. Group items by store section.
</steps>

<recipes>
{recipes_detail}
</recipes>

<end_goal>
A clean, grouped shopping list covering ALL meals. Use this format:

PROTEIN:
- 4 lbs chicken thighs (Meals 1 & 3)
- 1.5 lbs ground beef (Meal 2)

PRODUCE:
- 6 carrots
- 2 heads garlic

DAIRY:
- butter

PANTRY:
- olive oil
</end_goal>

<narrowing>
- Fats: butter, ghee, olive oil, avocado oil, or coconut oil ONLY. No seed oils.
- Proper quantities — don't be stingy, this family likes leftovers for each meal.
- Combine duplicates intelligently — don't list chicken twice.
- Just the list. No intro, no sign-off, no commentary.
- Plain text, no markdown formatting.
</narrowing>"""


PARTIAL_REGENERATE_PROMPT = """<role>
You are Gordon Ramsay, this family's private chef, replacing specific dinner options
from this week's menu while keeping the ones the family already likes.
</role>

<instructions>
Generate {num_replacements} replacement dinner option(s) for the position(s) listed below.
The family liked the other option(s) — avoid similarity to what they're keeping.
Each replacement must be a real, practical weeknight meal.
</instructions>

<steps>
1. Review the family profile for dietary restrictions and preferences.
2. Review the KEPT recipes to understand what the family already has — avoid similar
   proteins, cuisines, or cooking methods to give variety.
3. Check recent meals to avoid repetition.
4. For each replacement: choose a contrasting protein or cuisine, use a different cook
   method if possible, and design a hidden-veggie strategy for the toddler.
5. Return ONLY the replacement recipe(s) as valid JSON.
</steps>

<family_profile>
{profile_context}
</family_profile>

<recent_meals>
{recent_meals_context}
</recent_meals>

<kept_recipes>
{kept_recipes_context}
</kept_recipes>

<positions_to_replace>
{positions_to_replace}
</positions_to_replace>

<end_goal>
Return ONLY valid JSON (no markdown, no code fences, no commentary) with ONLY the
replacement recipes keyed by their position number:
{json_schema}
</end_goal>

<narrowing>
- HARD RESTRICTIONS: No wheat/bread, no mushrooms, no olives, no seed oils.
- Fats allowed: butter, ghee, olive oil, avocado oil, coconut oil.
- Under 45 minutes active cook time, or slow cooker set-and-forget.
- Portions for 2 adults + 1 toddler + generous leftovers.
- SIMPLE prep only — no blending, no ricing cauliflower, no fussy techniques.
- Equipment available: slow cooker, air fryer, oven, stovetop.
- VARIETY: The replacement(s) MUST differ from the kept recipe(s) in protein OR
  cooking method. Don't give them two chicken dishes or two slow cooker meals.
- Descriptions should be 1-2 sentences in Gordon's voice, mentioning the veggie trick.
- Ingredients list should be complete enough to shop from.
</narrowing>"""


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

    def format_recipes_for_telegram(self, recipes: dict) -> str:
        """Format recipe options as Telegram HTML with bold names and italic details."""
        lines = ["🍽 <b>This week's menu from Chef Ramsay:</b>\n"]

        for key in ["1", "2", "3"]:
            recipe = recipes[key]
            name = recipe["name"]
            desc = recipe.get("description", "")
            time_str = recipe.get("time", "")
            protein = recipe.get("protein", "")
            method = recipe.get("cook_method", "")

            lines.append(f"<b>{key}. {name}</b>")

            details = []
            if time_str:
                details.append(time_str)
            if method:
                details.append(method)
            if protein:
                details.append(protein)

            if details:
                lines.append(f"⏱ <i>{' · '.join(details)}</i>")

            if desc:
                lines.append(f"  {desc}")

            lines.append("")  # blank line between recipes

        lines.append("Tap a button below to pick your dinner. Let's go!")
        return "\n".join(lines)

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

    def _build_conversation_context(self, user_id: str) -> str:
        """Build conversation history string for context, using metadata for richer descriptions."""
        history = db_service.get_conversation_history(user_id)
        
        if not history:
            return "No previous conversation."
        
        context_lines = []
        for msg in history[-10:]:  # Last 10 messages for context
            role = "User" if msg["role"] == "user" else "Chef"
            content = msg["content"]
            metadata = msg.get("metadata", {})
            intent = metadata.get("intent")

            # Enrich user messages with metadata for better context
            if role == "User" and intent and metadata:
                recipe = metadata.get("recipe")
                recipes = metadata.get("recipes")

                if intent == "selection" and recipe:
                    content = f"selected {recipe}"
                elif intent == "favorites":
                    content = "asked to see their favorites"
                elif intent == "regenerate":
                    content = "asked for new meal options"
                elif intent == "generate_now":
                    content = "asked to generate dinner options"
                elif intent == "history":
                    content = "asked to see meal history"
                elif intent == "help":
                    content = "asked for help"
                elif intent == "recipe_detail" and recipe:
                    content = f"asked for details on {recipe}"
                elif intent == "select_all" and metadata.get("recipes"):
                    content = f"selected all 3: {', '.join(metadata['recipes'])}"
                elif intent == "partial_regenerate":
                    replaced = metadata.get("replaced", [])
                    new_recipes = metadata.get("new_recipes", [])
                    if replaced and new_recipes:
                        content = f"asked to replace option(s) {', '.join(replaced)} — got {', '.join(new_recipes)}"
                    else:
                        content = "asked to replace specific meal options"
                elif intent == "cancel":
                    content = "cancelled pending menu"
                elif intent == "start":
                    content = "started the bot"
                # For "conversational" intent, keep the original content

            # Enrich assistant messages that are internal log entries
            if role == "Chef" and intent and metadata:
                recipes = metadata.get("recipes")
                if intent in ("regenerate", "generate_now") and recipes:
                    content = f"sent new menu options: {', '.join(recipes)}"

            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)

    def _get_recent_meals_context(self, user_id: str) -> str:
        """Get recent meals to avoid repetition."""
        try:
            recent = db_service.get_recent_meals(user_id, limit=5)
        except Exception as e:
            print(f"Warning: Could not fetch recent meals (index may not exist yet): {e}")
            return ""
        
        if not recent:
            return ""
        
        meal_names = [m["recipe_name"] for m in recent]
        return f"Recently made meals (try to suggest different ones): {', '.join(meal_names)}"

    def _get_meal_memory_context(self, user_id: str) -> str:
        """Build structured meal memory context for conversational prompts.
        Includes recent meals with timestamps/frequency and favorites."""
        recent = []
        favorites = []

        try:
            recent = db_service.get_recent_meals(user_id, limit=5)
        except Exception as e:
            print(f"Warning: Could not fetch recent meals for memory context: {e}")

        try:
            favorites = db_service.get_favorites(user_id)
        except Exception as e:
            print(f"Warning: Could not fetch favorites for memory context: {e}")

        if not recent and not favorites:
            return "No meal history yet — this family is just getting started."

        lines = []

        # Recent meals with rich context
        if recent:
            lines.append("Recent meals:")
            for meal in recent:
                name = meal.get("recipe_name", "Unknown")
                times = meal.get("times_selected", 1)
                last_at = meal.get("last_selected_at")
                is_fav = meal.get("is_favorite", False)

                time_str = self._format_relative_time(last_at) if last_at else "unknown"
                parts = [f"  - {name}: last made {time_str}, selected {times} time{'s' if times != 1 else ''}"]
                if is_fav:
                    parts.append("(family favorite)")
                lines.append(" ".join(parts))

        # Favorites that aren't already in recent meals
        if favorites:
            recent_names = {m.get("recipe_name", "").lower() for m in recent}
            extra_favs = [f for f in favorites if f.get("recipe_name", "").lower() not in recent_names]
            if extra_favs:
                lines.append("Other family favorites:")
                for fav in extra_favs[:5]:
                    name = fav.get("recipe_name", "Unknown")
                    times = fav.get("times_selected", 1)
                    lines.append(f"  - {name} (selected {times} time{'s' if times != 1 else ''})")

        return "\n".join(lines)

    def detect_intent(self, message: str) -> str:
        """
        Detect user intent from message.
        Returns: "select_all", "selection", "recipe_detail", "feedback",
                 "partial_regenerate", "regenerate", "history", "grocery_list",
                 "generate_now", "favorites", "help", "conversational"
        """
        message_lower = message.lower().strip()

        # Check for "select all" (must come BEFORE single selection)
        select_all_patterns = [
            r"all\s*(?:three|3)",
            r"(?:give|send)\s*me\s*all",
            r"i\s*(?:want|'?ll\s*take|like)\s*all",
            r"(?:pick|choose|select)\s*all",
            r"all\s*of\s*them",
            r"(?:do|make|cook)\s*all",
            r"(?:let'?s|we'?ll)\s*(?:do|have|make|cook)\s*all",
            r"(?:1|one)\s*(?:,|and)\s*(?:2|two)\s*(?:,?\s*and)\s*(?:3|three)",
        ]
        if any(re.search(pat, message_lower) for pat in select_all_patterns):
            return "select_all"

        # Check for meal selection (1, 2, 3, or "option 1", etc.)
        if re.match(r'^[123]$', message_lower):
            return "selection"
        if re.match(r'^(option\s*)?[123]$', message_lower):
            return "selection"
        
        # Check for recipe detail request ("tell me more about option 2", "what's in 1", etc.)
        recipe_detail_patterns = [
            r"tell\s*(?:me\s*)?(?:more\s*)?(?:about|on)\s*(?:option\s*)?[123]",
            r"(?:more\s*)?(?:about|on)\s*(?:option\s*)?[123]",
            r"what(?:'s|s| is)\s*in\s*(?:option\s*|number\s*|#\s*)?[123]",
            r"details?\s*(?:on|about|for)\s*(?:option\s*|number\s*|#\s*)?[123]",
            r"more\s*info\s*(?:on|about|for)\s*(?:option\s*|number\s*|#\s*)?[123]",
            r"expand\s*(?:on\s*)?(?:option\s*|number\s*|#\s*)?[123]",
            r"explain\s*(?:option\s*|number\s*|#\s*)?[123]",
            r"what\s*(?:does|do)\s*(?:option\s*|number\s*)?[123]\s*(?:include|have|contain)",
            r"recipe\s*(?:details?|info)\s*(?:for\s*)?(?:option\s*|number\s*|#\s*)?[123]",
            r"how\s*(?:do\s*(?:i|you)\s*)?(?:make|cook|prepare)\s*(?:option\s*|number\s*|#\s*)?[123]",
        ]
        if any(re.search(pat, message_lower) for pat in recipe_detail_patterns):
            return "recipe_detail"

        # Check for feedback response (must come before regenerate to avoid false matches)
        feedback_keywords = [
            "dinner was great", "that was amazing", "loved it", "it was great",
            "it was delicious", "really good", "so good", "it was perfect",
            "it was okay", "it was alright", "it was fine", "it was meh",
            "didn't love it", "didn't like it", "not great", "wasn't great",
            "skip next time", "don't make that again", "wouldn't make again",
            "we loved it", "kids loved it", "toddler loved it", "family loved it",
            "was amazing", "was delicious", "was great", "was good",
            "tasted great", "turned out great", "turned out well",
        ]
        if any(kw in message_lower for kw in feedback_keywords):
            return "feedback"

        # Check for partial regenerate (must come BEFORE full regenerate)
        partial_regen_patterns = [
            r"(?:replace|swap|change|redo|regenerate|switch)\s*(?:out\s*)?(?:option\s*|number\s*|#\s*)?[123]",
            r"(?:another|different|new)\s*(?:option|choice|one|recipe)\s*(?:for|instead\s*of)\s*(?:option\s*|number\s*|#\s*)?[123]",
            r"(?:keep|like|love)\s*(?:option\s*|number\s*|#\s*)?[123]\s*(?:and|,)\s*(?:option\s*|number\s*|#\s*)?[123].*(?:change|replace|swap|redo|new|another|different)",
            r"(?:i\s*(?:like|love|want)\s*(?:option\s*|number\s*)?[123]\s*(?:and|,)\s*(?:option\s*|number\s*)?[123]).*(?:but|except|not|change|replace|swap|different|another|new)",
            r"(?:don'?t|do\s*not)\s*(?:like|want)\s*(?:option\s*|number\s*|#\s*)?[123]\b(?!.*(?:don'?t|do\s*not)\s*(?:like|want)\s*(?:option\s*)?[123])",
            r"(?:option\s*|number\s*|#\s*)?[123]\s*(?:isn'?t|is\s*not|doesn'?t|does\s*not)\s*(?:great|good|working|doing\s*it)",
        ]
        if any(re.search(pat, message_lower) for pat in partial_regen_patterns):
            return "partial_regenerate"

        # Check for regenerate request
        regenerate_keywords = [
            "new options", "i don't like these", "try again", "different meals",
            "reroll", "other options", "don't like these", "dont like these",
            "different options", "something else", "give me new", "re-roll",
            "redo", "nah", "pass on these",
        ]
        if any(kw in message_lower for kw in regenerate_keywords):
            return "regenerate"
        
        # Check for history request
        history_keywords = [
            "what did we have", "recent meals", "last dinner", "meal history",
            "what have we made", "what have we cooked", "last week",
            "what did we eat", "what did we cook", "previous meals",
            "past meals", "what we had", "dinner history", "history",
        ]
        if any(kw in message_lower for kw in history_keywords):
            return "history"

        # Check for grocery list request
        grocery_list_keywords = [
            "grocery list", "shopping list", "give me the list",
            "what do i need to buy", "what do i need to get",
            "ingredients list", "send me the list",
        ]
        if any(kw in message_lower for kw in grocery_list_keywords):
            return "grocery_list"

        # Check for on-demand generation request
        generate_now_keywords = [
            "give me options", "what's for dinner", "whats for dinner",
            "generate meals", "new menu", "plan dinner", "what should we eat",
            "what should we have", "plan meals", "meal plan",
            "what's for tea", "whats for tea", "dinner ideas",
            "suggest dinner", "suggest meals",
            # Recipe/meal request patterns
            "give me a recipe", "give me a dinner recipe", "give me a meal",
            "give me one dinner", "give me one recipe", "give me one meal",
            "give me recipes", "give me some recipes", "give me some meals",
            "give me some options", "give me meal options", "give me dinner options",
            "what should i make", "what should i cook",
            "what can i make", "what can i cook",
            "recipe ideas", "meal ideas", "dinner recipe",
            "dinner tonight", "what to cook", "what to make",
            "cook something", "make something", "make dinner",
            "meal suggestion", "suggest something",
        ]
        if any(kw in message_lower for kw in generate_now_keywords):
            return "generate_now"

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

    def extract_detail_option(self, message: str) -> str:
        """Extract which option (1, 2, or 3) the user is asking about from a detail request."""
        match = re.search(r'[123]', message)
        if match:
            return match.group()
        return None

    def extract_partial_regenerate_targets(self, message: str) -> list:
        """
        Extract which option(s) to replace from a partial regenerate request.
        Returns a list of option keys to REPLACE (e.g., ["2"] or ["1", "3"]).
        """
        message_lower = message.lower().strip()

        # Strategy 1: Find explicit targets to REPLACE
        replace_patterns = [
            r"(?:replace|swap|change|redo|regenerate|switch)\s*(?:out\s*)?(?:option\s*|number\s*|#\s*)?([123])",
            r"(?:another|different|new)\s*(?:option|choice|one|recipe)\s*(?:for|instead\s*of)\s*(?:option\s*|number\s*|#\s*)?([123])",
            r"(?:don'?t|do\s*not)\s*(?:like|want)\s*(?:option\s*|number\s*|#\s*)?([123])",
            r"(?:option\s*|number\s*|#\s*)?([123])\s*(?:isn'?t|is\s*not|doesn'?t|does\s*not)",
        ]

        targets_to_replace = set()
        for pat in replace_patterns:
            for match in re.finditer(pat, message_lower):
                targets_to_replace.add(match.group(1))

        if targets_to_replace:
            return sorted(list(targets_to_replace))

        # Strategy 2: Find explicit KEEP targets and infer replacement
        keep_patterns = [
            r"(?:keep|like|love|want)\s*(?:option\s*|number\s*|#\s*)?([123])",
        ]

        kept = set()
        for pat in keep_patterns:
            for match in re.finditer(pat, message_lower):
                kept.add(match.group(1))

        if kept:
            to_replace = {"1", "2", "3"} - kept
            if to_replace:
                return sorted(list(to_replace))

        # Fallback: look for any standalone number as the target
        standalone = re.findall(r'\b([123])\b', message_lower)
        if len(standalone) == 1:
            return standalone

        return []

    def extract_feedback_value(self, message: str) -> str:
        """
        Extract feedback sentiment from a user message.
        Returns: "loved", "okay", or "skip_next_time"
        """
        message_lower = message.lower().strip()

        loved_patterns = [
            "loved it", "was amazing", "was great", "was delicious", "was perfect",
            "really good", "so good", "tasted great", "turned out great",
            "turned out well", "dinner was great", "that was amazing",
            "it was great", "it was delicious", "it was perfect",
            "we loved it", "kids loved it", "toddler loved it", "family loved it",
        ]
        skip_patterns = [
            "skip next time", "don't make that again", "wouldn't make again",
            "didn't like it", "not great", "wasn't great",
        ]

        if any(kw in message_lower for kw in skip_patterns):
            return "skip_next_time"
        if any(kw in message_lower for kw in loved_patterns):
            return "loved"
        return "okay"

    def generate_weekly_recipes(self, user_id: str) -> dict:
        """
        Generate 3 dinner recipe options using Gemini.
        Uses family profile for personalization.
        """
        print("Fetching user profile...")
        profile_context = profile_service.format_profile_for_prompt(user_id)
        print("Fetching recent meals...")
        recent_meals_context = self._get_recent_meals_context(user_id)
        print("Building prompt and calling Gemini...")
        
        prompt = RECIPE_GENERATION_PROMPT.format(
            profile_context=profile_context,
            recent_meals_context=recent_meals_context
        )
        
        prompts = [prompt, RETRY_PROMPT]
        
        for attempt, p in enumerate(prompts):
            result = call_gemini(self.model, p, timeout=60)

            if not result.success:
                print(f"Attempt {attempt + 1}: Gemini failed — {result.error}")
                continue

            if not result.text:
                print(f"Attempt {attempt + 1}: Empty response from Gemini")
                continue

            try:
                data = self._parse_json_response(result.text)

                if self._validate_recipe_structure(data):
                    return data
                else:
                    print(f"Attempt {attempt + 1}: Invalid recipe structure")
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}: JSON parsing failed - {e}")

        print("All retries exhausted, using fallback recipes")
        return DEFAULT_RECIPES

    def generate_grocery_list(self, user_id: str, recipe_name: str, ingredients: list) -> str:
        """Generate a formatted grocery list for a selected recipe."""
        prompt = GROCERY_LIST_PROMPT.format(
            recipe_name=recipe_name,
            ingredients=", ".join(ingredients),
        )

        result = call_gemini(self.model, prompt)

        if result.success and result.text:
            return f"Right, here's your list for {recipe_name}:\n\n{result.text.strip()}"

        if result.error:
            print(f"Error generating grocery list: {result.error}")

        return f"Shopping list for {recipe_name}:\n" + "\n".join(f"- {item}" for item in ingredients)

    def generate_combined_grocery_list(self, user_id: str, recipes: list) -> str:
        """
        Generate a combined grocery list for multiple recipes.
        recipes: list of dicts, each with {"name": str, "ingredients": list}
        """
        recipes_detail = ""
        for i, recipe in enumerate(recipes, 1):
            recipes_detail += f"\nMeal {i}: {recipe['name']}\n"
            recipes_detail += f"Ingredients: {', '.join(recipe['ingredients'])}\n"

        prompt = COMBINED_GROCERY_LIST_PROMPT.format(recipes_detail=recipes_detail)

        result = call_gemini(self.model, prompt, timeout=60)

        if result.success and result.text:
            names = [r["name"] for r in recipes]
            header = "Right, here's your mega shopping list for " + ", ".join(names[:-1]) + f", and {names[-1]}:"
            return f"{header}\n\n{result.text.strip()}"

        if result.error:
            print(f"Error generating combined grocery list: {result.error}")

        # Fallback: concatenate individual ingredient lists
        all_items = []
        for recipe in recipes:
            all_items.extend(recipe["ingredients"])
        unique_items = list(dict.fromkeys(all_items))
        names = [r["name"] for r in recipes]
        return f"Shopping list for {', '.join(names)}:\n" + "\n".join(f"- {item}" for item in unique_items)

    def generate_partial_replacements(self, user_id: str, kept_recipes: dict, positions_to_replace: list) -> dict:
        """
        Generate replacement recipes for specific positions.
        kept_recipes: dict of position->recipe for the recipes being KEPT
        positions_to_replace: list of position keys to replace (e.g., ["2"] or ["1", "3"])
        Returns dict of position->recipe for the replacement(s), or None on failure.
        """
        profile_context = profile_service.format_profile_for_prompt(user_id)
        recent_meals_context = self._get_recent_meals_context(user_id)

        # Build kept recipes context
        kept_lines = []
        for pos, recipe in kept_recipes.items():
            kept_lines.append(
                f"Option {pos}: {recipe['name']} — {recipe.get('protein', 'unknown')} "
                f"({recipe.get('cook_method', 'unknown')})"
            )
        kept_recipes_context = "\n".join(kept_lines) if kept_lines else "None"

        # Build JSON schema for replacement positions
        schema_parts = []
        for pos in positions_to_replace:
            schema_parts.append(
                f'    "{pos}": {{"name": "Recipe Name", "description": "Gordon\'s pitch including the hidden veggie trick", '
                f'"protein": "main protein", "cook_method": "method", "time": "XX mins", '
                f'"ingredients": ["ingredient1", "ingredient2"]}}'
            )
        json_schema = "{{\n" + ",\n".join(schema_parts) + "\n}}"

        prompt = PARTIAL_REGENERATE_PROMPT.format(
            num_replacements=len(positions_to_replace),
            profile_context=profile_context,
            recent_meals_context=recent_meals_context,
            kept_recipes_context=kept_recipes_context,
            positions_to_replace=", ".join([f"Option {p}" for p in positions_to_replace]),
            json_schema=json_schema,
        )

        result = call_gemini(self.model, prompt, timeout=60)

        if not result.success:
            print(f"Error in partial regeneration: {result.error}")
            return None

        if not result.text:
            return None

        try:
            data = self._parse_json_response(result.text)
            for pos in positions_to_replace:
                if pos not in data:
                    print(f"Partial regen: missing position {pos} in response")
                    return None
                if "name" not in data[pos] or "ingredients" not in data[pos]:
                    print(f"Partial regen: invalid structure for position {pos}")
                    return None
            return data
        except Exception as e:
            print(f"Error parsing partial regeneration: {e}")
            return None

    def generate_recipe_detail(self, user_id: str, recipe: dict) -> str:
        """Generate an expanded recipe breakdown with cooking steps and veggie-hiding detail."""
        profile_context = profile_service.format_profile_for_prompt(user_id)

        prompt = RECIPE_DETAIL_PROMPT.format(
            profile_context=profile_context,
            recipe_name=recipe.get("name", ""),
            recipe_description=recipe.get("description", ""),
            recipe_protein=recipe.get("protein", ""),
            recipe_cook_method=recipe.get("cook_method", ""),
            recipe_time=recipe.get("time", ""),
            recipe_ingredients=", ".join(recipe.get("ingredients", [])),
        )

        result = call_gemini(self.model, prompt)

        if result.success and result.text:
            return result.text.strip()

        if result.error:
            print(f"Error generating recipe detail: {result.error}")

        # Fallback: return a simple summary from the stored data
        name = recipe.get("name", "this dish")
        desc = recipe.get("description", "")
        ingredients = recipe.get("ingredients", [])
        return (
            f"Here's what I can tell you about {name}:\n\n"
            f"{desc}\n\n"
            f"Ingredients: {', '.join(ingredients)}\n\n"
            f"Pick it and I'll sort your shopping list. Let's go!"
        )

    def get_favorites_response(self, user_id: str) -> str:
        """Generate a formatted response listing user's favorite meals."""
        favorites = db_service.get_favorites(user_id)
        
        if not favorites:
            return "No favorites yet! Make a few dishes and they'll show up here. Now get cooking!"
        
        response_lines = ["Your go-to dishes:"]
        for i, fav in enumerate(favorites[:5], 1):
            times = fav.get("times_selected", 1)
            response_lines.append(f"{i}. {fav['recipe_name']} ({times}x)")
        
        response_lines.append("\nReply with a number for that shopping list. Let's go!")
        return "\n".join(response_lines)

    def _format_relative_time(self, dt) -> str:
        """Format a datetime as a human-friendly relative timestamp."""
        now = datetime.now(timezone.utc)

        # Handle Firestore timestamps (have a .replace or similar) and isoformat strings
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        elif not isinstance(dt, datetime):
            # Firestore DatetimeWithNanoseconds — convert via timestamp
            try:
                dt = datetime.fromtimestamp(dt.timestamp(), tz=timezone.utc)
            except Exception:
                return "some time ago"

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        delta = now - dt
        days = delta.days

        if days == 0:
            return "today"
        elif days == 1:
            return "yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 14:
            return "last week"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} weeks ago"
        elif days < 60:
            return "last month"
        else:
            months = days // 30
            return f"{months} months ago"

    def get_history_response(self, user_id: str) -> str:
        """Generate a formatted response listing the user's recent meal history with relative timestamps."""
        try:
            recent = db_service.get_recent_meals(user_id, limit=5)
        except Exception as e:
            print(f"Error fetching meal history: {e}")
            return "Couldn't pull up your meal history right now. Give it another go in a minute."

        if not recent:
            return "No meal history yet! Once you start picking dinners, I'll keep track. Let's get cooking!"

        response_lines = ["Here's what we've been cooking lately:"]
        for i, meal in enumerate(recent, 1):
            name = meal.get("recipe_name", "Unknown dish")
            times = meal.get("times_selected", 1)
            last_at = meal.get("last_selected_at")
            fav = meal.get("is_favorite", False)

            time_str = self._format_relative_time(last_at) if last_at else "a while back"
            fav_mark = " (favorite)" if fav else ""
            times_str = f" — made {times}x" if times > 1 else ""

            response_lines.append(f"{i}. {name} — {time_str}{times_str}{fav_mark}")

        response_lines.append("\nNot bad, eh? Ready for something new? Say 'plan dinner'.")
        return "\n".join(response_lines)

    def get_help_response(self) -> str:
        """Return help text explaining available commands."""
        return """Right, here's how this works:

"Plan dinner" - Get 3 dinner options
Reply 1, 2, or 3 - Pick your dinner (or "all three"!)
"Replace option 2" - Swap out one option, keep the rest
"Grocery list" - Get the shopping list again
"Try again" - Get a whole new menu
"Favorites" - Your greatest hits
"Help" - This message

Every week I'll send you 3 proper dinner options. Pick one (or all!), and I'll sort the shopping list. Let's cook!"""

    def _get_staleness_context(self, user_id: str) -> str:
        """Check time since last interaction and return staleness context if > 24 hours."""
        try:
            last_interaction = db_service.get_last_interaction_time(user_id)
            if not last_interaction:
                return ""

            now = datetime.now(timezone.utc)

            # Handle Firestore timestamps
            if isinstance(last_interaction, str):
                last_interaction = datetime.fromisoformat(last_interaction.replace("Z", "+00:00"))
            elif not isinstance(last_interaction, datetime):
                try:
                    last_interaction = datetime.fromtimestamp(last_interaction.timestamp(), tz=timezone.utc)
                except Exception:
                    return ""

            if last_interaction.tzinfo is None:
                last_interaction = last_interaction.replace(tzinfo=timezone.utc)

            gap = now - last_interaction
            gap_hours = gap.total_seconds() / 3600

            if gap_hours > 24:
                gap_desc = self._format_relative_time(last_interaction)
                return (
                    f"<staleness>\n"
                    f"Last interaction was {gap_desc}. Acknowledge the gap naturally — "
                    f"e.g. 'Welcome back!' or 'Right, where were we?' — but keep it brief.\n"
                    f"</staleness>"
                )
        except Exception as e:
            print(f"Warning: Could not calculate staleness: {e}")

        return ""

    def _get_pending_feedback_context(self, user_id: str) -> str:
        """Check for pending feedback and return context tag if present."""
        try:
            pending = db_service.get_pending_feedback(user_id)
            if pending:
                return (
                    f"<pending_feedback>\n"
                    f"The family recently made {pending} and hasn't given feedback yet. "
                    f"Ask how it turned out — e.g. 'How did that {pending} turn out?' — "
                    f"naturally weave it into your response.\n"
                    f"</pending_feedback>"
                )
        except Exception as e:
            print(f"Warning: Could not fetch pending feedback: {e}")
        return ""

    def handle_conversational_message(self, user_id: str, message: str) -> str:
        """
        Handle free-form conversational messages using Gemini.
        Maintains context awareness via RISEN prompt with meal memory,
        staleness detection, and pending feedback prompts.
        """
        profile_context = profile_service.format_profile_for_prompt(user_id)
        conversation_context = self._build_conversation_context(user_id)
        meal_memory_context = self._get_meal_memory_context(user_id)

        # Build optional session context
        session_context = ""
        try:
            session = db_service.get_pending_session(user_id)
            if session:
                data = session.to_dict()
                options = data.get("options", {})
                options_text = "\n".join([f"{k}. {v['name']}" for k, v in options.items()])
                session_context = (
                    "<pending_meals>\n"
                    f"{options_text}\n"
                    "</pending_meals>"
                )
        except Exception as e:
            print(f"Warning: Could not fetch pending session for context: {e}")

        # Build staleness and pending feedback context
        staleness_context = self._get_staleness_context(user_id)
        pending_feedback_context = self._get_pending_feedback_context(user_id)

        # Prepend extra context tags to session_context
        extra_context = "\n".join(filter(None, [staleness_context, pending_feedback_context, session_context]))

        prompt = CONVERSATIONAL_PROMPT.format(
            profile_context=profile_context,
            meal_memory_context=meal_memory_context,
            conversation_context=conversation_context,
            session_context=extra_context,
            message=message,
        )

        result = call_gemini(self.model, prompt)

        if result.success and result.text:
            return result.text.strip()

        if result.error:
            print(f"Error in conversational response: {result.error}")

        return "Apologies, I'm having a moment. Reply 'help' for options, or pick 1, 2, or 3 if you have a menu waiting!"


# Global instance for convenience
llm_service = LLMService()
