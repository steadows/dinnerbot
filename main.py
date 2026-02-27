"""
Cloud Function entry points for DinnerBot.
All HTTP handlers live here — thin wrappers that delegate to service modules.
"""

import functions_framework
from flask import jsonify

from config import config
from db_service import db_service
from llm_service import llm_service
from telegram_service import telegram_service


# ==================== CRON TRIGGER ====================


@functions_framework.http
def cron_trigger_recipes(request):
    """
    Cloud Function entry point triggered by Cloud Scheduler (weekly cron).
    Generates recipe options, stores them in Firestore, and sends to Telegram.
    """
    chat_id_raw = config.TELEGRAM_CHAT_ID
    if not chat_id_raw:
        print("ERROR: TELEGRAM_CHAT_ID is not configured")
        return jsonify({"error": "TELEGRAM_CHAT_ID not configured"}), 500

    chat_id = int(chat_id_raw)
    user_id = f"telegram_{chat_id}"
    platform = "telegram"

    print(f"Starting weekly recipe generation for {user_id}")

    # 1. Expire any pending sessions so the user gets a clean slate
    expired_count = db_service.expire_pending_sessions(user_id)
    if expired_count:
        print(f"Expired {expired_count} pending session(s) for {user_id}")

    # 2. Generate recipes via LLM (falls back to DEFAULT_RECIPES on failure)
    try:
        recipes = llm_service.generate_weekly_recipes(user_id)
    except Exception as e:
        print(f"ERROR generating recipes: {e}")
        return jsonify({"error": "Recipe generation failed"}), 500

    print(f"Generated recipes: {[recipes[k]['name'] for k in ['1', '2', '3']]}")

    # 3. Create session in Firestore
    try:
        session_id = db_service.create_session(
            user_id=user_id,
            options=recipes,
            platform=platform,
            chat_id=chat_id,
        )
    except Exception as e:
        print(f"ERROR creating session in Firestore: {e}")
        return jsonify({"error": "Session creation failed"}), 500

    print(f"Created session {session_id}")

    # 4. Send recipes to Telegram with inline keyboard
    send_success = telegram_service.send_recipes(
        chat_id=chat_id,
        recipes=recipes,
        session_id=session_id,
    )

    # 5. If Telegram send failed, mark session as send_failed
    if not send_success:
        print(f"ERROR: Failed to send recipes to Telegram chat {chat_id}")
        try:
            db_service.mark_session_failed(session_id, "Telegram send failed")
        except Exception as e:
            print(f"ERROR marking session as failed: {e}")
        return jsonify({"error": "Telegram send failed", "session_id": session_id}), 500

    print(f"Recipes sent to Telegram chat {chat_id} — session {session_id}")
    return jsonify({
        "status": "ok",
        "session_id": session_id,
        "recipes": [recipes[k]["name"] for k in ["1", "2", "3"]],
    }), 200


# ==================== TELEGRAM WEBHOOK ====================


@functions_framework.http
def telegram_webhook(request):
    """
    Cloud Function entry point for Telegram webhook updates.
    Handles both callback queries (button taps) and text messages.
    Always returns HTTP 200 to prevent Telegram retries.
    """
    # 1. Validate webhook secret token
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    expected_secret = config.TELEGRAM_WEBHOOK_SECRET
    if expected_secret and secret_token != expected_secret:
        print("WARNING: Invalid or missing webhook secret token")
        return jsonify({"status": "unauthorized"}), 200

    # 2. Parse the update JSON
    try:
        update = request.get_json(force=True)
    except Exception as e:
        print(f"ERROR: Failed to parse webhook body: {e}")
        return jsonify({"status": "ok"}), 200

    if not update:
        print("WARNING: Empty update body")
        return jsonify({"status": "ok"}), 200

    # 3. Extract chat_id from the update
    chat_id = _extract_chat_id(update)
    if not chat_id:
        print(f"WARNING: Could not extract chat_id from update: {list(update.keys())}")
        return jsonify({"status": "ok"}), 200

    # 4. Check chat_id against allowed list
    allowed_ids = config.ALLOWED_CHAT_IDS
    if allowed_ids and chat_id not in allowed_ids:
        print(f"WARNING: Unauthorized chat_id {chat_id}")
        return jsonify({"status": "ok"}), 200

    user_id = f"telegram_{chat_id}"

    # 5. Route: callback_query (button tap) vs message (text) vs non-text message
    try:
        if "callback_query" in update:
            _handle_callback_query(update["callback_query"], chat_id, user_id)
        elif "message" in update and "text" in update.get("message", {}):
            _handle_text_message(update["message"], chat_id, user_id)
        elif "message" in update:
            # Non-text message (photo, sticker, voice, video, document, etc.)
            _handle_non_text_message(update["message"], chat_id, user_id)
        elif "edited_message" in update and "text" in update.get("edited_message", {}):
            _handle_text_message(update["edited_message"], chat_id, user_id)
        elif "edited_message" in update:
            # Edited non-text message — same handling
            _handle_non_text_message(update["edited_message"], chat_id, user_id)
        else:
            print(f"Unhandled update type: {list(update.keys())}")
    except Exception as e:
        print(f"ERROR handling update: {e}")
        telegram_service.send_message(
            chat_id,
            "Bloody hell, something went wrong on my end. Give it another go in a few minutes.",
        )

    return jsonify({"status": "ok"}), 200


# ==================== INTERNAL HELPERS ====================


def _extract_chat_id(update: dict) -> int | None:
    """Extract chat_id from a Telegram update (callback_query, message, or edited_message)."""
    if "callback_query" in update:
        return (
            update["callback_query"]
            .get("message", {})
            .get("chat", {})
            .get("id")
        )
    if "message" in update:
        return update["message"].get("chat", {}).get("id")
    if "edited_message" in update:
        return update["edited_message"].get("chat", {}).get("id")
    return None


def _handle_callback_query(callback_query: dict, chat_id: int, user_id: str):
    """Handle an inline keyboard button tap (callback query)."""
    callback_id = callback_query.get("id")
    data = callback_query.get("data", "")

    print(f"Callback query from {user_id}: {data}")

    # Parse callback data format: "select:{session_id}:{choice}"
    parts = data.split(":")
    if len(parts) == 3 and parts[0] == "select":
        session_id = parts[1]
        choice = parts[2]
        _handle_meal_selection(chat_id, user_id, session_id, choice, callback_id)
    else:
        print(f"Unknown callback data format: {data}")
        if callback_id:
            telegram_service.answer_callback_query(callback_id, "I didn't catch that.")


def _handle_text_message(message: dict, chat_id: int, user_id: str):
    """Handle a text message — detect slash commands first, then intent routing."""
    text = message.get("text", "").strip()
    if not text:
        return

    print(f"Text message from {user_id}: {text}")

    # --- Slash commands take priority over intent routing ---
    if text.startswith("/"):
        command = text.split()[0].lower().split("@")[0]  # handle "/help@BotName"
        print(f"Slash command detected: {command}")

        if command == "/start":
            _handle_start(chat_id, user_id)
            return
        elif command in ("/help", "/menu"):
            _handle_help(chat_id, user_id)
            return
        elif command == "/favorites":
            _handle_favorites(chat_id, user_id)
            return
        elif command == "/cancel":
            _handle_cancel(chat_id, user_id)
            return
        # Unknown slash command — fall through to intent routing

    # --- Standard intent routing ---
    intent = llm_service.detect_intent(text)
    print(f"Detected intent: {intent}")

    if intent == "select_all":
        _handle_select_all_from_text(chat_id, user_id)
    elif intent == "selection":
        selection = llm_service.extract_selection(text)
        _handle_meal_selection_from_text(chat_id, user_id, selection)
    elif intent == "feedback":
        _handle_feedback(chat_id, user_id, text)
    elif intent == "recipe_detail":
        _handle_recipe_detail(chat_id, user_id, text)
    elif intent == "partial_regenerate":
        _handle_partial_regenerate(chat_id, user_id, text)
    elif intent == "history":
        _handle_history(chat_id, user_id)
    elif intent == "regenerate":
        _handle_regenerate(chat_id, user_id)
    elif intent == "generate_now":
        _handle_generate_now(chat_id, user_id)
    elif intent == "grocery_list":
        _handle_grocery_list(chat_id, user_id, text)
    elif intent == "favorites":
        _handle_favorites(chat_id, user_id)
    elif intent == "help":
        _handle_help(chat_id, user_id)
    else:
        _handle_conversational(chat_id, user_id, text)


def _handle_non_text_message(message: dict, chat_id: int, user_id: str):
    """Handle non-text messages (photos, stickers, voice, video, documents, etc.)."""
    msg_type = _detect_message_type(message)
    print(f"Non-text message from {user_id}: type={msg_type}")

    telegram_service.send_message(
        chat_id,
        "I can only read text messages, love. Type something or tap a button.",
    )


def _detect_message_type(message: dict) -> str:
    """Detect the content type of a non-text Telegram message for logging."""
    for content_type in [
        "photo", "sticker", "voice", "video", "document", "audio",
        "animation", "video_note", "contact", "location", "venue",
        "poll", "dice",
    ]:
        if content_type in message:
            return content_type
    return "unknown"


# ==================== SLASH COMMAND HANDLERS ====================


def _handle_start(chat_id: int, user_id: str):
    """Handle /start — welcome message introducing the bot in Gordon's voice."""
    welcome = (
        "Right, listen up! I'm your personal chef, Gordon Ramsay, "
        "and I'm here to sort your dinners every week.\n\n"
        "Every Sunday I'll send you 3 proper meal options — tap a button to pick one, "
        "and I'll give you the shopping list. Simple as that.\n\n"
        "Say /help to see what I can do, or just ask me anything about food. Let's go!"
    )
    db_service.append_to_conversation(user_id, "user", "/start", metadata={"intent": "start"})
    db_service.append_to_conversation(user_id, "assistant", welcome, metadata={"intent": "start"})
    telegram_service.send_message(chat_id, welcome)


def _handle_cancel(chat_id: int, user_id: str):
    """Handle /cancel — expire any pending sessions for the user."""
    expired_count = db_service.expire_pending_sessions(user_id)

    if expired_count:
        msg = (
            f"Done — cancelled {expired_count} pending menu{'s' if expired_count > 1 else ''}. "
            "Say 'plan dinner' when you're ready for new options!"
        )
    else:
        msg = "Nothing to cancel, love. You don't have any pending menus right now."

    db_service.append_to_conversation(user_id, "user", "/cancel", metadata={"intent": "cancel"})
    db_service.append_to_conversation(user_id, "assistant", msg, metadata={"intent": "cancel"})
    telegram_service.send_message(chat_id, msg)


# ==================== INTENT HANDLERS ====================


def _handle_meal_selection(
    chat_id: int, user_id: str, session_id: str, choice: str, callback_id: str = None
):
    """Handle a meal selection from an inline keyboard button tap."""
    print(f"Meal selection: user={user_id}, session={session_id}, choice={choice}")

    # 1. Look up session by ID
    session = db_service.get_session_by_id(session_id)
    if not session:
        print(f"No session found for id={session_id}")
        if callback_id:
            telegram_service.answer_callback_query(
                callback_id, "No meal options waiting for you right now, love."
            )
        telegram_service.send_message(
            chat_id,
            "No meal options waiting for you right now, love. Check back on Sunday!",
        )
        return

    data = session.to_dict()

    # 2. Validate session belongs to user and is still pending
    if data.get("user_id") != user_id:
        print(f"Session {session_id} does not belong to {user_id}")
        if callback_id:
            telegram_service.answer_callback_query(callback_id, "That's not your menu!")
        return

    if data.get("status") != "pending_selection":
        print(f"Session {session_id} status is '{data.get('status')}', not pending")
        if callback_id:
            telegram_service.answer_callback_query(
                callback_id, "You've already picked from this menu!"
            )
        telegram_service.send_message(
            chat_id,
            "You've already made your pick from this menu. Wait for next week's options!",
        )
        return

    # 3. Handle "all" selection — redirect to dedicated handler
    options = data.get("options", {})
    if choice == "all":
        _handle_select_all(chat_id, user_id, session_id, data, callback_id)
        return

    # 4. Validate choice maps to a stored option
    if choice not in options:
        print(f"Invalid choice '{choice}' — not in session options")
        if callback_id:
            telegram_service.answer_callback_query(
                callback_id,
                "I didn't catch that. Tap one of the buttons or reply 1, 2, or 3.",
            )
        telegram_service.send_message(
            chat_id,
            "I didn't catch that. Tap one of the buttons or reply 1, 2, or 3 to pick your dinner.",
        )
        return

    selected = options[choice]
    recipe_name = selected["name"]
    ingredients = selected.get("ingredients", [])

    # 4. Mark session as completed
    try:
        db_service.update_session_selection(session_id, choice)
        print(f"Session {session_id} marked completed with choice={choice}")
    except Exception as e:
        print(f"ERROR updating session selection: {e}")
        if callback_id:
            telegram_service.answer_callback_query(callback_id, "Something went wrong!")
        telegram_service.send_message(
            chat_id,
            "Bloody hell, something went wrong on my end. Give it another go in a few minutes.",
        )
        return

    # 5. Save to meal history
    try:
        db_service.save_meal_to_history(user_id, recipe_name, ingredients)
        print(f"Saved '{recipe_name}' to meal history for {user_id}")
    except Exception as e:
        print(f"WARNING: Failed to save meal to history: {e}")

    # 6. Log selection to conversation history with metadata
    selection_metadata = {"intent": "selection", "recipe": recipe_name, "session_id": session_id}
    db_service.append_to_conversation(user_id, "user", choice, metadata=selection_metadata)

    # 7. Answer callback query (removes loading spinner in Telegram)
    if callback_id:
        telegram_service.answer_callback_query(
            callback_id, f"Beautiful! {recipe_name} it is!"
        )

    # 8. Send confirmation message
    confirmation_msg = f"Right, {recipe_name} — lovely choice! Let me sort your shopping list..."
    db_service.append_to_conversation(user_id, "assistant", confirmation_msg, metadata=selection_metadata)
    telegram_service.send_message(chat_id, confirmation_msg)

    # 9. Trigger grocery list generation (Phase 5 wiring)
    _generate_and_send_grocery_list(chat_id, user_id, recipe_name, ingredients)


def _handle_meal_selection_from_text(chat_id: int, user_id: str, selection: str):
    """Handle a meal selection from a text reply (e.g. user sends '2')."""
    print(f"Text meal selection: user={user_id}, selection={selection}")

    # Validate we got a selection
    if not selection:
        telegram_service.send_message(
            chat_id,
            "I didn't catch that. Tap one of the buttons or reply 1, 2, or 3 to pick your dinner.",
        )
        return

    # Look up the user's pending session
    session = db_service.get_pending_session(user_id)
    if not session:
        telegram_service.send_message(
            chat_id,
            "No meal options waiting for you right now, love. Check back on Sunday!",
        )
        return

    # Delegate to the main selection handler using the pending session's ID
    _handle_meal_selection(
        chat_id=chat_id,
        user_id=user_id,
        session_id=session.id,
        choice=selection,
        callback_id=None,
    )


def _handle_select_all_from_text(chat_id: int, user_id: str):
    """Handle a 'select all' from text input."""
    session = db_service.get_pending_session(user_id)
    if not session:
        telegram_service.send_message(
            chat_id,
            "No meal options waiting for you right now, love. Say 'plan dinner' to get started!",
        )
        return

    data = session.to_dict()
    _handle_select_all(chat_id, user_id, session.id, data, callback_id=None)


def _handle_select_all(
    chat_id: int, user_id: str, session_id: str, data: dict, callback_id: str = None
):
    """Handle selecting all 3 recipes — save all to history, generate combined grocery list."""
    options = data.get("options", {})

    all_recipes = []
    all_ingredients = []

    for key in ["1", "2", "3"]:
        recipe = options[key]
        recipe_name = recipe["name"]
        ingredients = recipe.get("ingredients", [])
        all_recipes.append(recipe_name)
        all_ingredients.append({"name": recipe_name, "ingredients": ingredients})

        # Save each meal to history
        try:
            db_service.save_meal_to_history(user_id, recipe_name, ingredients)
            print(f"Saved '{recipe_name}' to meal history for {user_id}")
        except Exception as e:
            print(f"WARNING: Failed to save meal to history: {e}")

    # Mark session as completed with "all"
    try:
        db_service.update_session_selection(session_id, "all")
        print(f"Session {session_id} marked completed with choice=all")
    except Exception as e:
        print(f"ERROR updating session selection: {e}")
        if callback_id:
            telegram_service.answer_callback_query(callback_id, "Something went wrong!")
        telegram_service.send_message(
            chat_id,
            "Bloody hell, something went wrong on my end. Give it another go in a few minutes.",
        )
        return

    # Log to conversation history
    selection_metadata = {
        "intent": "select_all",
        "recipes": all_recipes,
        "session_id": session_id,
    }
    db_service.append_to_conversation(user_id, "user", "all", metadata=selection_metadata)

    # Answer callback query
    if callback_id:
        telegram_service.answer_callback_query(callback_id, "All three! Love the ambition!")

    # Send confirmation
    names_str = ", ".join(all_recipes[:-1]) + f", and {all_recipes[-1]}"
    confirmation_msg = f"All three — {names_str}! Love the ambition! Let me sort one big shopping list..."
    db_service.append_to_conversation(user_id, "assistant", confirmation_msg, metadata=selection_metadata)
    telegram_service.send_message(chat_id, confirmation_msg)

    # Generate combined grocery list
    _generate_and_send_combined_grocery_list(chat_id, user_id, all_ingredients)


def _generate_and_send_combined_grocery_list(
    chat_id: int, user_id: str, recipes: list
):
    """
    Generate a combined grocery list for multiple recipes and send via Telegram.
    recipes: list of dicts with {"name": str, "ingredients": list}
    """
    try:
        grocery_text = llm_service.generate_combined_grocery_list(user_id, recipes)
        telegram_service.send_grocery_list(chat_id, grocery_text)
        print(f"Combined grocery list sent to chat {chat_id} for {len(recipes)} recipes")

        # Set pending feedback for the last recipe
        try:
            last_recipe_name = recipes[-1]["name"]
            db_service.set_pending_feedback(user_id, last_recipe_name)
        except Exception as e:
            print(f"WARNING: Failed to set pending feedback: {e}")

    except Exception as e:
        print(f"ERROR generating/sending combined grocery list: {e}")
        telegram_service.send_message(
            chat_id,
            "Bloody hell, I couldn't put together the shopping list. "
            "Give it another go or just use the ingredients from the recipes!",
        )


def _handle_partial_regenerate(chat_id: int, user_id: str, text: str):
    """Handle a partial regenerate request — replace specific option(s), keep the rest."""
    partial_meta = {"intent": "partial_regenerate"}
    db_service.append_to_conversation(user_id, "user", text, metadata=partial_meta)

    # 1. Check for a pending session
    session = db_service.get_pending_session(user_id)
    if not session:
        msg = "No menu to tweak right now, love. Say 'plan dinner' to get new options."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=partial_meta)
        telegram_service.send_message(chat_id, msg)
        return

    data = session.to_dict()
    options = data.get("options", {})

    # 2. Extract which option(s) to replace
    targets = llm_service.extract_partial_regenerate_targets(text)
    if not targets:
        msg = "Which option do you want me to replace — 1, 2, or 3?"
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=partial_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # Validate targets exist in session
    invalid = [t for t in targets if t not in options]
    if invalid:
        msg = f"Option {', '.join(invalid)} doesn't exist on this menu. Pick from 1, 2, or 3."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=partial_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # If all 3 are being replaced, redirect to full regenerate
    if len(targets) == 3:
        _handle_regenerate(chat_id, user_id)
        return

    # 3. Build kept recipes dict
    kept_recipes = {k: v for k, v in options.items() if k not in targets}

    # 4. Notify user
    replacing_str = " and ".join([f"option {t}" for t in targets])
    telegram_service.send_message(
        chat_id,
        f"Right, swapping out {replacing_str}. Give me a moment...",
    )

    # 5. Generate replacement(s) via LLM
    try:
        replacements = llm_service.generate_partial_replacements(
            user_id, kept_recipes, targets
        )
    except Exception as e:
        print(f"ERROR in partial regeneration: {e}")
        replacements = None

    if not replacements:
        msg = "Bloody hell, couldn't come up with a replacement. Try saying 'try again' for a whole new menu."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=partial_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 6. Update session options in place
    try:
        db_service.update_session_options(session.id, replacements)
        print(f"Updated session {session.id} with replacement(s) at position(s) {targets}")
    except Exception as e:
        print(f"ERROR updating session options: {e}")
        msg = "Bloody hell, something went wrong on my end. Give it another go."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=partial_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 7. Merge replacements into the full options for display
    updated_options = {**options, **replacements}

    # 8. Re-send the full menu with buttons
    send_success = telegram_service.send_recipes(
        chat_id=chat_id,
        recipes=updated_options,
        session_id=session.id,
    )

    if not send_success:
        msg = "Couldn't send the updated menu. Try saying 'plan dinner' to start fresh."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=partial_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 9. Log to conversation history
    new_recipe_names = [replacements[t]["name"] for t in targets]
    partial_meta.update({
        "session_id": session.id,
        "replaced": targets,
        "new_recipes": new_recipe_names,
    })
    db_service.append_to_conversation(
        user_id,
        "assistant",
        f"[Replaced option(s) {', '.join(targets)}: {', '.join(new_recipe_names)}]",
        metadata=partial_meta,
    )

    print(f"Partial regenerate: replaced {targets} in session {session.id}")


def _handle_regenerate(chat_id: int, user_id: str):
    """Handle a regenerate request — expire current options and generate fresh ones."""
    platform = "telegram"

    # Save the user's intent to conversation history
    regen_meta = {"intent": "regenerate"}
    db_service.append_to_conversation(user_id, "user", "[Regenerate: requested new meal options]", metadata=regen_meta)

    # Check if there's a pending session to replace
    print(f"Regenerate: checking for pending session for {user_id}")
    session = db_service.get_pending_session(user_id)
    if not session:
        print(f"Regenerate: no pending session found for {user_id}")
        msg = "No menu to replace right now, love. Say 'plan dinner' to get new options."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=regen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    print(f"Regenerate: found pending session {session.id} for {user_id}")

    # 1. Expire the current pending session(s)
    expired_count = db_service.expire_pending_sessions(user_id)
    print(f"Regenerate: expired {expired_count} pending session(s) for {user_id}")

    telegram_service.send_message(
        chat_id,
        "Right, scrapping that lot. Let me cook up something better...",
    )

    # 2. Generate new recipes
    try:
        recipes = llm_service.generate_weekly_recipes(user_id)
    except Exception as e:
        print(f"ERROR generating recipes during regenerate: {e}")
        msg = "Bloody hell, something went wrong on my end. Give it another go in a few minutes."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=regen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    print(f"Regenerate: generated recipes: {[recipes[k]['name'] for k in ['1', '2', '3']]}")

    # 3. Create new session in Firestore
    try:
        session_id = db_service.create_session(
            user_id=user_id,
            options=recipes,
            platform=platform,
            chat_id=chat_id,
        )
    except Exception as e:
        print(f"ERROR creating session during regenerate: {e}")
        msg = "Bloody hell, something went wrong on my end. Give it another go in a few minutes."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=regen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    print(f"Regenerate: created new session {session_id}")

    # 4. Send new recipes to Telegram
    send_success = telegram_service.send_recipes(
        chat_id=chat_id,
        recipes=recipes,
        session_id=session_id,
    )

    if not send_success:
        print(f"ERROR: Failed to send regenerated recipes to Telegram chat {chat_id}")
        try:
            db_service.mark_session_failed(session_id, "Telegram send failed on regenerate")
        except Exception as e:
            print(f"ERROR marking session as failed: {e}")
        msg = "Bloody hell, something went wrong sending those options. Give it another go."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=regen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 5. Save success to conversation history
    recipe_names = [recipes[k]["name"] for k in ["1", "2", "3"]]
    regen_success_meta = {"intent": "regenerate", "session_id": session_id, "recipes": recipe_names}
    db_service.append_to_conversation(
        user_id,
        "assistant",
        f"[Regenerated new options: {', '.join(recipe_names)}]",
        metadata=regen_success_meta,
    )

    print(f"Regenerate: new recipes sent to chat {chat_id} — session {session_id}")


def _handle_generate_now(chat_id: int, user_id: str):
    """Handle an on-demand 'generate now' request — generate new options or re-send pending ones."""
    platform = "telegram"

    # Save the user's intent to conversation history
    gen_meta = {"intent": "generate_now"}
    db_service.append_to_conversation(user_id, "user", "[Generate Now: requested meal options]", metadata=gen_meta)

    # Check if there's already a pending session
    print(f"Generate Now: checking for pending session for {user_id}")
    session = db_service.get_pending_session(user_id)

    if session:
        # Pending session exists — remind user and re-send the existing menu
        print(f"Generate Now: found pending session {session.id} for {user_id}")
        data = session.to_dict()
        options = data.get("options", {})

        msg = "You've already got options waiting, love! Here they are again:"
        db_service.append_to_conversation(user_id, "assistant", msg, metadata={"intent": "generate_now", "session_id": session.id})
        telegram_service.send_message(chat_id, msg)

        # Re-send the recipes with inline keyboard buttons
        send_success = telegram_service.send_recipes(
            chat_id=chat_id,
            recipes=options,
            session_id=session.id,
        )

        if not send_success:
            print(f"ERROR: Failed to re-send recipes to Telegram chat {chat_id}")
            telegram_service.send_message(
                chat_id,
                "Bloody hell, couldn't pull up the menu. Try sending 1, 2, or 3 to pick your dinner.",
            )
        return

    # No pending session — run the full generation flow
    print(f"Generate Now: no pending session, generating new recipes for {user_id}")

    # 1. Expire any old sessions (safety net)
    expired_count = db_service.expire_pending_sessions(user_id)
    if expired_count:
        print(f"Generate Now: expired {expired_count} stale session(s)")

    telegram_service.send_message(
        chat_id,
        "Right, let me think about what you should have tonight...",
    )

    # 2. Generate recipes
    try:
        recipes = llm_service.generate_weekly_recipes(user_id)
    except Exception as e:
        print(f"ERROR generating recipes during generate_now: {e}")
        msg = "Bloody hell, something went wrong on my end. Give it another go in a few minutes."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=gen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    print(f"Generate Now: generated recipes: {[recipes[k]['name'] for k in ['1', '2', '3']]}")

    # 3. Create session in Firestore
    try:
        session_id = db_service.create_session(
            user_id=user_id,
            options=recipes,
            platform=platform,
            chat_id=chat_id,
        )
    except Exception as e:
        print(f"ERROR creating session during generate_now: {e}")
        msg = "Bloody hell, something went wrong on my end. Give it another go in a few minutes."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=gen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    print(f"Generate Now: created session {session_id}")

    # 4. Send recipes to Telegram
    send_success = telegram_service.send_recipes(
        chat_id=chat_id,
        recipes=recipes,
        session_id=session_id,
    )

    if not send_success:
        print(f"ERROR: Failed to send recipes to Telegram chat {chat_id}")
        try:
            db_service.mark_session_failed(session_id, "Telegram send failed on generate_now")
        except Exception as e:
            print(f"ERROR marking session as failed: {e}")
        msg = "Bloody hell, something went wrong sending those options. Give it another go."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=gen_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 5. Save success to conversation history
    recipe_names = [recipes[k]["name"] for k in ["1", "2", "3"]]
    gen_success_meta = {"intent": "generate_now", "session_id": session_id, "recipes": recipe_names}
    db_service.append_to_conversation(
        user_id,
        "assistant",
        f"[Generated on-demand options: {', '.join(recipe_names)}]",
        metadata=gen_success_meta,
    )

    print(f"Generate Now: recipes sent to chat {chat_id} — session {session_id}")


def _handle_grocery_list(chat_id: int, user_id: str, text: str):
    """Handle a grocery list request — find the most recent selected meal and regenerate the list."""
    grocery_meta = {"intent": "grocery_list"}
    db_service.append_to_conversation(user_id, "user", text, metadata=grocery_meta)

    # 1. Check for the most recently selected meal in history
    try:
        recent_meals = db_service.get_recent_meals(user_id, limit=1)
    except Exception:
        recent_meals = []

    if recent_meals:
        meal = recent_meals[0]
        recipe_name = meal.get("recipe_name", "")
        ingredients = meal.get("ingredients", [])

        if recipe_name and ingredients:
            msg = f"Right, let me pull up the list for {recipe_name}..."
            db_service.append_to_conversation(user_id, "assistant", msg, metadata=grocery_meta)
            telegram_service.send_message(chat_id, msg)
            _generate_and_send_grocery_list(chat_id, user_id, recipe_name, ingredients)
            return

    # 2. Check for a pending session — user might want to pick first
    session = db_service.get_pending_session(user_id)
    if session:
        msg = "You've got meal options waiting! Pick one (1, 2, or 3) and I'll sort the shopping list for you."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=grocery_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 3. No meal to generate a list for
    msg = "No meals to make a list for yet, love. Say 'plan dinner' and let's get you sorted."
    db_service.append_to_conversation(user_id, "assistant", msg, metadata=grocery_meta)
    telegram_service.send_message(chat_id, msg)


def _handle_recipe_detail(chat_id: int, user_id: str, text: str):
    """Handle a recipe detail request — expand a pending recipe with cooking steps."""
    print(f"Recipe detail request from {user_id}: {text}")

    # Save the user's message to conversation history
    detail_meta = {"intent": "recipe_detail"}
    db_service.append_to_conversation(user_id, "user", text, metadata=detail_meta)

    # 1. Look up the user's pending session
    session = db_service.get_pending_session(user_id)
    if not session:
        msg = "No menu to look at right now, love. Say 'plan dinner' to get new options."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=detail_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 2. Extract which option the user is asking about
    option = llm_service.extract_detail_option(text)
    if not option:
        msg = "Which option do you want details on — 1, 2, or 3?"
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=detail_meta)
        telegram_service.send_message(chat_id, msg)
        return

    # 3. Validate the option exists in the session
    data = session.to_dict()
    options = data.get("options", {})
    if option not in options:
        msg = f"Option {option} doesn't exist on this menu. Pick 1, 2, or 3."
        db_service.append_to_conversation(user_id, "assistant", msg, metadata=detail_meta)
        telegram_service.send_message(chat_id, msg)
        return

    recipe = options[option]
    recipe_name = recipe.get("name", "this dish")
    print(f"Recipe detail: expanding option {option} — {recipe_name}")

    detail_meta = {"intent": "recipe_detail", "session_id": session.id, "recipe": recipe_name}

    # 4. Call Gemini for expanded detail
    try:
        detail_text = llm_service.generate_recipe_detail(user_id, recipe)
    except Exception as e:
        print(f"ERROR generating recipe detail: {e}")
        detail_text = (
            f"Here's what I can tell you about {recipe_name}:\n\n"
            f"{recipe.get('description', '')}\n\n"
            f"Ingredients: {', '.join(recipe.get('ingredients', []))}\n\n"
            f"Pick it and I'll sort your shopping list. Let's go!"
        )

    # 5. Send via Telegram and save to conversation history
    db_service.append_to_conversation(user_id, "assistant", detail_text, metadata=detail_meta)
    telegram_service.send_message(chat_id, detail_text)


def _generate_and_send_grocery_list(
    chat_id: int, user_id: str, recipe_name: str, ingredients: list
):
    """
    Generate a grocery list for the selected recipe and send it via Telegram.
    Sets pending_feedback so Gordon asks about the meal on next interaction.
    """
    try:
        grocery_text = llm_service.generate_grocery_list(user_id, recipe_name, ingredients)
        telegram_service.send_grocery_list(chat_id, grocery_text)
        print(f"Grocery list sent to chat {chat_id} for '{recipe_name}'")

        # Set pending feedback so Gordon asks "How did that turn out?" next time
        try:
            db_service.set_pending_feedback(user_id, recipe_name)
            print(f"Pending feedback set for '{recipe_name}' for {user_id}")
        except Exception as e:
            print(f"WARNING: Failed to set pending feedback: {e}")

    except Exception as e:
        print(f"ERROR generating/sending grocery list: {e}")
        telegram_service.send_message(
            chat_id,
            "Bloody hell, I couldn't put together the shopping list. "
            "Give it another go or just use the ingredients from the recipe!",
        )


def _handle_favorites(chat_id: int, user_id: str):
    """Handle a favorites request with conversation history."""
    fav_meta = {"intent": "favorites"}
    # Save the user's intent to conversation history
    db_service.append_to_conversation(user_id, "user", "favorites", metadata=fav_meta)

    response = llm_service.get_favorites_response(user_id)

    # Save the bot's response to conversation history
    db_service.append_to_conversation(user_id, "assistant", response, metadata=fav_meta)

    telegram_service.send_message(chat_id, response)


def _handle_history(chat_id: int, user_id: str):
    """Handle a meal history request with conversation history."""
    hist_meta = {"intent": "history"}
    # Save the user's intent to conversation history
    db_service.append_to_conversation(user_id, "user", "meal history", metadata=hist_meta)

    response = llm_service.get_history_response(user_id)

    # Save the bot's response to conversation history
    db_service.append_to_conversation(user_id, "assistant", response, metadata=hist_meta)

    telegram_service.send_message(chat_id, response)


def _handle_help(chat_id: int, user_id: str):
    """Handle a help request with conversation history."""
    help_meta = {"intent": "help"}
    # Save the user's intent to conversation history
    db_service.append_to_conversation(user_id, "user", "help", metadata=help_meta)

    response = llm_service.get_help_response()

    # Save the bot's response to conversation history
    db_service.append_to_conversation(user_id, "assistant", response, metadata=help_meta)

    telegram_service.send_message(chat_id, response)


def _handle_feedback(chat_id: int, user_id: str, text: str):
    """Handle meal feedback — update meal_history and clear pending_feedback."""
    feedback_value = llm_service.extract_feedback_value(text)
    pending_recipe = db_service.get_pending_feedback(user_id)

    feedback_meta = {"intent": "feedback", "feedback": feedback_value}

    if pending_recipe:
        feedback_meta["recipe"] = pending_recipe
        db_service.append_to_conversation(user_id, "user", text, metadata=feedback_meta)

        # Update meal history with feedback
        updated = db_service.update_meal_feedback(user_id, pending_recipe, feedback_value)
        if updated:
            print(f"Feedback '{feedback_value}' saved for '{pending_recipe}' ({user_id})")
        else:
            print(f"WARNING: Could not find meal history entry for '{pending_recipe}' ({user_id})")

        # Clear pending feedback
        db_service.clear_pending_feedback(user_id)

        # Respond in character based on feedback
        if feedback_value == "loved":
            response = f"Beautiful! Glad the {pending_recipe} was a hit. I'll remember that — expect to see it again. Let's go!"
        elif feedback_value == "skip_next_time":
            response = f"Fair enough — we'll shelve the {pending_recipe} for now. Plenty more where that came from."
        else:
            response = f"Right, noted on the {pending_recipe}. Not every dish is a showstopper, but we'll dial it in."

        db_service.append_to_conversation(user_id, "assistant", response, metadata=feedback_meta)
        telegram_service.send_message(chat_id, response)
    else:
        # No pending feedback — treat as conversational with the feedback context
        db_service.append_to_conversation(user_id, "user", text, metadata=feedback_meta)
        response = llm_service.handle_conversational_message(user_id, text)
        db_service.append_to_conversation(user_id, "assistant", response, metadata=feedback_meta)
        telegram_service.send_message(chat_id, response)


def _handle_conversational(chat_id: int, user_id: str, text: str):
    """Handle free-form conversational messages with conversation history."""
    conv_meta = {"intent": "conversational"}
    # Save the user's message to conversation history
    db_service.append_to_conversation(user_id, "user", text, metadata=conv_meta)

    # Check if there's pending feedback that will be included in the prompt
    had_pending_feedback = db_service.get_pending_feedback(user_id) is not None

    # Generate response via LLM (uses conversation history for context)
    response = llm_service.handle_conversational_message(user_id, text)

    # Save the bot's response to conversation history
    db_service.append_to_conversation(user_id, "assistant", response, metadata=conv_meta)

    # Send via Telegram
    telegram_service.send_message(chat_id, response)

    # Clear pending feedback after Gordon has asked about it once — don't nag every message
    if had_pending_feedback:
        db_service.clear_pending_feedback(user_id)
