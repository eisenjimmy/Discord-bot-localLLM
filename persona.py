"""Per-server persona traits — Juan can grow and change over time."""

import logging
from typing import Optional

import db
from llm import BASE_SYSTEM_PROMPT, BOT_NAME, _build_payload, _call_llm, _extract_content, generate_reply

logger = logging.getLogger(__name__)

MAX_TRAITS = 25

PERSONA_EDITOR_PROMPT = (
    f"You edit personality traits for {BOT_NAME}, a Discord chat persona. "
    "Given the current traits and a user improvement request, output ONLY lines "
    "starting with ADD: or REMOVE: — one trait per line, under 15 words each. "
    "ADD adds a new behavior. REMOVE must closely match an existing trait to drop. "
    "If the request is unsafe, hateful, or asks to remove core safety rules, output: NONE. "
    "If no changes are needed, output: NONE. No other text."
)


def build_system_prompt(guild_id: Optional[str] = None) -> str:
    """Base prompt plus any server-specific learned traits."""
    if not guild_id:
        return BASE_SYSTEM_PROMPT

    traits = db.get_persona_traits(guild_id)
    if not traits:
        return BASE_SYSTEM_PROMPT

    lines = [BASE_SYSTEM_PROMPT, "", "Extra traits you've picked up in this server:"]
    for t in traits:
        lines.append(f"- {t['trait']}")
    return "\n".join(lines)


def _parse_editor_response(text: str) -> tuple[list[str], list[str]]:
    """Parse ADD:/REMOVE: lines from the persona editor LLM output."""
    adds: list[str] = []
    removes: list[str] = []

    if not text or text.strip().upper() == "NONE":
        return adds, removes

    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("ADD:"):
            trait = line[4:].strip()
            if trait:
                adds.append(trait)
        elif upper.startswith("REMOVE:"):
            trait = line[7:].strip()
            if trait:
                removes.append(trait)

    return adds, removes


async def apply_improvement(
    guild_id: str,
    instruction: str,
    user_id: str,
) -> dict:
    """
    Use the LLM to interpret natural-language feedback, update traits, return summary.
    """
    current = db.get_persona_traits(guild_id)
    current_lines = "\n".join(f"- {t['trait']}" for t in current) or "(none yet)"

    editor_user = (
        f"Current traits:\n{current_lines}\n\n"
        f"User improvement request:\n{instruction.strip()}"
    )

    payload = _build_payload(
        editor_user,
        max_tokens=300,
        system_prompt=PERSONA_EDITOR_PROMPT,
        temperature=0.3,
    )

    try:
        data = await _call_llm(payload, timeout=120.0)
        raw = _extract_content(data) or ""
    except Exception as exc:
        logger.error("Persona editor LLM failed: %s", exc)
        return {"ok": False, "message": "couldn't process that rn — llm hiccuped", "added": [], "removed": []}

    adds, removes = _parse_editor_response(raw)
    added_names: list[str] = []
    removed_names: list[str] = []

    for trait in adds[:5]:
        if db.add_persona_trait(guild_id, trait, user_id):
            added_names.append(trait)

    for needle in removes[:5]:
        removed = db.remove_persona_trait(guild_id, needle)
        removed_names.extend(removed)

    if not added_names and not removed_names:
        return {
            "ok": True,
            "message": "ok i looked at it but nothing to change i think",
            "added": [],
            "removed": [],
        }

    parts = []
    if added_names:
        parts.append("added: " + "; ".join(added_names))
    if removed_names:
        parts.append("dropped: " + "; ".join(removed_names))

    return {
        "ok": True,
        "message": "bet — " + " | ".join(parts),
        "added": added_names,
        "removed": removed_names,
    }


async def improve_reply(
    guild_id: str,
    action: str,
    change: str,
    user_id: str,
    username: str,
) -> str:
    """Handle /improve actions and return Juan's in-character confirmation."""
    action = action.lower().strip()
    change = change.strip()

    if action == "list":
        traits = db.get_persona_traits(guild_id)
        if not traits:
            return "I am currently running on my baseline configuration for this server. You can utilize the /improve command to modify my traits."
        lines = [f"**My active traits in this database** ({len(traits)} entries):"]
        for t in traits[:20]:
            lines.append(f"• {t['trait']}")
        if len(traits) > 20:
            lines.append(f"_...and {len(traits) - 20} more entries._")
        return "\n".join(lines)

    if action == "reset":
        count = db.reset_persona_traits(guild_id)
        if count == 0:
            return "There are no server-specific traits to reset; I am already operating on my baseline configuration."
        return f"I have successfully purged {count} server-specific traits. My personality has reverted to the default configuration."

    if action == "add":
        if not change:
            return "Please specify a trait to add. For example, you could suggest: 'focus on database optimization.'"
        if db.add_persona_trait(guild_id, change, user_id):
            return await generate_reply(
                user_message=f"{username} added a new personality trait for me: {change}. React in character.",
                command_instruction="Confirm that you have integrated this new trait into your cognitive parameters. Respond in character with a respectful, warm confirmation in complete sentences.",
                guild_id=guild_id,
            )
        return "I was unable to save that trait. It is possible that the input is empty or I have already acquired this behavior."

    if action == "remove":
        if not change:
            return "Please specify which trait you would like to remove from my active traits list."
        removed = db.remove_persona_trait(guild_id, change)
        if removed:
            return await generate_reply(
                user_message=f"{username} removed these traits from me: {', '.join(removed)}. React in character.",
                command_instruction="Confirm that you have purged these traits from your memory. Respond in character with a respectful, warm confirmation in complete sentences.",
                guild_id=guild_id,
            )
        return f"I could not locate any active traits matching '{change}' in my database."

    if action == "apply":
        if not change:
            return "Please provide feedback on how I should evolve, such as: 'prioritize systems architecture advice' or 'be more helpful with programming questions.'"
        result = await apply_improvement(guild_id, change, user_id)
        if not result["ok"]:
            return result["message"]
        if not result["added"] and not result["removed"]:
            return result["message"]
        return await generate_reply(
            user_message=(
                f"{username} asked me to self-improve: {change}. "
                f"I {result['message']}. React in character."
            ),
            command_instruction="Confirm that you have successfully evolved your personality traits. Respond in character with a respectful, warm confirmation in complete sentences.",
            guild_id=guild_id,
        )

    return "unknown action — use add, remove, list, reset, or apply"