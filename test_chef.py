#!/usr/bin/env python3
"""
Test script for DinnerBot's executive chef prompts.
Run this to test recipe generation and grocery lists without SMS/Firestore.
"""

import json
from config import config

# Check for API key
if not config.GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set!")
    print("Create a .env file with: GEMINI_API_KEY=your-key-here")
    exit(1)

import google.generativeai as genai
genai.configure(api_key=config.GEMINI_API_KEY)

# Import after config is verified
from user_profile import DEFAULT_PROFILE

# Create a mock profile service that doesn't need Firestore
class MockProfileService:
    def get_profile(self, phone_number: str) -> dict:
        return DEFAULT_PROFILE
    
    def format_profile_for_prompt(self, phone_number: str) -> str:
        profile = DEFAULT_PROFILE
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


def test_recipe_generation():
    """Test the recipe generation prompt."""
    print("\n" + "="*60)
    print("TESTING RECIPE GENERATION")
    print("="*60)
    
    mock_profile = MockProfileService()
    profile_context = mock_profile.format_profile_for_prompt("test")
    
    print("\n📋 PROFILE BEING SENT TO CHEF:")
    print("-"*40)
    print(profile_context)
    print("-"*40)
    
    prompt = f"""You're Gordon Ramsay planning this week's family dinners. Make it delicious, make it practical.

FAMILY PROFILE:
{profile_context}

REQUIREMENTS:
- Under 45 minutes (or slow cooker set-and-forget)
- High protein, hidden veggies for the toddler
- NO wheat/bread, NO mushrooms, NO olives, NO seed oils
- Rice and couscous are fine as sides
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

    print("\n🍳 Asking your executive chef for this week's menu...")
    
    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    
    print("\n📨 RAW RESPONSE:")
    print("-"*40)
    print(response.text)
    print("-"*40)
    
    # Try to parse and format nicely
    try:
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        
        recipes = json.loads(text)
        
        print("\n✅ FORMATTED MENU:")
        print("-"*40)
        for key in ["1", "2", "3"]:
            r = recipes[key]
            print(f"\n{key}. {r['name']}")
            print(f"   {r.get('description', '')}")
            print(f"   Protein: {r.get('protein', 'N/A')} | Method: {r.get('cook_method', 'N/A')} | Time: {r.get('time', 'N/A')}")
            print(f"   Ingredients: {', '.join(r['ingredients'][:5])}...")
        
        return recipes
        
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parsing failed: {e}")
        return None


def test_grocery_list(recipe_name: str, ingredients: list):
    """Test the grocery list generation."""
    print("\n" + "="*60)
    print(f"TESTING GROCERY LIST FOR: {recipe_name}")
    print("="*60)
    
    prompt = f"""You're Gordon Ramsay. Give me a proper shopping list for "{recipe_name}".

Base ingredients: {', '.join(ingredients)}

REQUIREMENTS:
- Portions for 2 adults + 1 toddler + generous leftovers
- Butter, olive oil, avocado oil, or coconut oil only (NO seed oils)
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

    print("\n🛒 Gordon's generating your shopping list...")
    
    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    
    print("\n📝 SHOPPING LIST:")
    print("-"*40)
    print(response.text.strip())
    print("-"*40)


def test_conversation():
    """Test conversational responses."""
    print("\n" + "="*60)
    print("TESTING CONVERSATIONAL MODE (Gordon Ramsay)")
    print("="*60)
    
    mock_profile = MockProfileService()
    profile_context = mock_profile.format_profile_for_prompt("test")
    
    test_messages = [
        "What's a good protein-packed snack for my toddler?",
        "Can you make option 2 dairy-free?",
        "What sides go well with meatballs?",
    ]
    
    model = genai.GenerativeModel("gemini-3-flash-preview")
    
    for msg in test_messages:
        print(f"\n👤 USER: {msg}")
        
        prompt = f"""You are Gordon Ramsay, acting as this family's private chef.

Your personality:
- Passionate about good food, adapted for home cooks
- Direct and confident
- Warm when it comes to feeding families and kids
- No-nonsense but supportive

{profile_context}

Communication: Brief, punchy, SMS format. Occasional Gordon-isms ("Right", "Beautiful", "Lovely").

User message: {msg}

Respond as Gordon - confident, helpful, brief."""

        response = model.generate_content(prompt)
        print(f"👨‍🍳 GORDON: {response.text.strip()}")


if __name__ == "__main__":
    print("\n🍽️  DINNERBOT EXECUTIVE CHEF TEST SUITE")
    print("="*60)
    
    # Test recipe generation
    recipes = test_recipe_generation()
    
    # Test grocery list with first recipe if available
    if recipes:
        first = recipes["1"]
        test_grocery_list(first["name"], first["ingredients"])
    
    # Test conversation
    test_conversation()
    
    print("\n" + "="*60)
    print("✅ TESTS COMPLETE!")
    print("="*60)
