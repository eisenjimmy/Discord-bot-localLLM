"""Slash command handlers for the Discord bot."""

import logging
import os
import re
import time
from typing import TYPE_CHECKING

import discord
from discord import app_commands

import db
from llm import BOT_NAME, CHANNEL_CONTEXT_LIMIT, MEMORY_CONTEXT_LIMIT, generate_reply
from meme import create_meme, format_meme_reply
from persona import improve_reply
from safety import REFUSAL_MESSAGE, check_rate_limit, is_unsafe_request

if TYPE_CHECKING:
    from main import LocalLLMBot

logger = logging.getLogger(__name__)

# Juan doesn't know these are commands — just reacts like himself
INSTRUCTIONS = {
    "mention": (
        "Someone mentioned you in the chat. Provide a highly capable, knowledgeable, "
        "and respectful response. Always answer in complete, high-effort sentences."
    ),
    "ask": (
        "They asked you a question. Provide a thorough, intellectually sound, and "
        "technically detailed answer. Be respectful, clear, and helpful, always in "
        "complete sentences."
    ),
    "roast": (
        "Deliver a playful, lighthearted roast. Keep it entirely respectful and free "
        "of cruelty or offensive topics, using complete sentences and a warm tone. "
        "Never attack race, religion, gender, sexuality, disability, nationality, "
        "age, body, or health."
    ),
    "rank": (
        "Evaluate their idea and rate it from 1 to 10. Write a complete, high-effort "
        "sentence explaining your rating with constructive, honest feedback."
    ),
    "summarize": (
        "Recap the recent conversation in the channel. Provide a thorough, "
        "complete-sentence summary of the main topics discussed."
    ),
    "meme": (
        "You have generated a meme. Provide the image URL accompanied by a complete-sentence "
        "caption that is respectful and relevant."
    ),
}


class BotCommands(app_commands.Group):
    """Root command group — individual commands are registered on the tree."""


def register_commands(bot: "LocalLLMBot") -> None:
    """Attach all slash commands to the bot."""

    @bot.tree.command(name="ask", description="Ask the bot a question")
    @app_commands.describe(question="What do you want to know?")
    async def ask(interaction: discord.Interaction, question: str):
        await _handle_llm_command(interaction, bot, question, "ask")

    @bot.tree.command(
        name="search",
        description=f"Search the web and answer with what {BOT_NAME} finds",
    )
    @app_commands.describe(query="What to look up online")
    async def search_cmd(interaction: discord.Interaction, query: str):
        await _handle_llm_command(
            interaction, bot, query, "ask", force_search=True
        )

    @bot.tree.command(
        name="meme",
        description="Generate a meme via memegen.link",
    )
    @app_commands.describe(
        idea="Describe the meme (AI picks template + text)",
        template="Template ID (drake, buzz, doge, sparta, etc.)",
        top="Top text (if not using idea)",
        bottom="Bottom text (if not using idea)",
    )
    async def meme(
        interaction: discord.Interaction,
        idea: str = "",
        template: str = "drake",
        top: str = "",
        bottom: str = "",
    ):
        await interaction.response.defer(thinking=True)

        allowed, error = check_rate_limit(interaction.user.id)
        if not allowed:
            await interaction.followup.send(error, ephemeral=True)
            return

        if not idea.strip() and not top.strip() and not bottom.strip():
            await interaction.followup.send(
                "Please provide a meme idea or specify the top and bottom text parameters.",
                ephemeral=True,
            )
            return

        url, caption = await create_meme(
            template=template,
            top=top,
            bottom=bottom,
            idea=idea,
        )

        if not url:
            await interaction.followup.send(caption)
            return

        await interaction.followup.send(format_meme_reply(url, caption))

    @bot.tree.command(name="roast", description="Get a playful, safe roast")
    @app_commands.describe(target="Who or what to roast (optional)")
    async def roast(interaction: discord.Interaction, target: str = ""):
        prompt = target.strip() or f"roast {interaction.user.display_name}"
        await _handle_llm_command(interaction, bot, prompt, "roast")

    @bot.tree.command(name="rank", description="Rate an idea from 1 to 10")
    @app_commands.describe(idea="The idea to rate")
    async def rank(interaction: discord.Interaction, idea: str):
        await _handle_llm_command(interaction, bot, idea, "rank")

    @bot.tree.command(
        name="summarize",
        description="Summarize recent messages in this channel",
    )
    @app_commands.describe(
        count="Number of messages to summarize (default 30, max 100)"
    )
    async def summarize(
        interaction: discord.Interaction,
        count: app_commands.Range[int, 1, 100] = 30,
    ):
        await interaction.response.defer(thinking=True)

        allowed, error = check_rate_limit(interaction.user.id)
        if not allowed:
            await interaction.followup.send(error, ephemeral=True)
            return

        if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.followup.send("This command only works in server text channels.")
            return

        guild_id = str(interaction.guild.id)
        channel_id = str(interaction.channel.id)

        # Cap messages sent to LLM — large summaries are the #1 timeout cause
        llm_count = min(count, 40)
        messages = db.get_recent_messages(guild_id, channel_id, limit=llm_count)
        if not messages:
            await interaction.followup.send(
                "I have not recorded any messages in this channel's database yet. Please converse here before requesting a summary."
            )
            return

        channel_context = db.format_messages_for_prompt(messages)
        memory_context = db.format_memories_for_prompt(
            guild_id, channel_id, limit=MEMORY_CONTEXT_LIMIT
        )

        reply = await generate_reply(
            user_message=f"Summarize the last {len(messages)} messages.",
            memory_context=memory_context,
            channel_context=channel_context,
            command_instruction=INSTRUCTIONS["summarize"],
            guild_id=guild_id,
        )

        await interaction.followup.send(reply)

    @bot.tree.command(
        name="remember",
        description="Save a server inside joke or recurring fact",
    )
    @app_commands.describe(key="Short label", value="What to remember")
    async def remember(
        interaction: discord.Interaction,
        key: str,
        value: str,
    ):
        if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message(
                "This command only works in server text channels.",
                ephemeral=True,
            )
            return

        saved = db.save_memory(
            guild_id=str(interaction.guild.id),
            channel_id=str(interaction.channel.id),
            user_id=str(interaction.user.id),
            key=key,
            value=value,
        )

        if saved:
            await interaction.response.send_message(
                f"I have successfully committed **{key.strip().lower()}** to memory. My data logging parameters are functioning perfectly."
            )
        else:
            await interaction.response.send_message(
                "Please provide a non-empty value for me to commit to my memory database.",
                ephemeral=True,
            )

    @bot.tree.command(name="forget", description="Remove a saved memory by key")
    @app_commands.describe(key="The memory key to delete")
    async def forget(interaction: discord.Interaction, key: str):
        if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message(
                "This command only works in server text channels.",
                ephemeral=True,
            )
            return

        removed = db.forget_memory(
            guild_id=str(interaction.guild.id),
            channel_id=str(interaction.channel.id),
            key=key,
        )

        if removed:
            await interaction.response.send_message(
                f"I have successfully deleted the memory key **{key.strip().lower()}** from my database."
            )
        else:
            await interaction.response.send_message(
                f"I could not locate a memory matching the key **{key.strip().lower()}**.",
                ephemeral=True,
            )

    @bot.tree.command(
        name="lore",
        description="Show saved server memories for this channel",
    )
    async def lore(interaction: discord.Interaction):
        if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message(
                "This command only works in server text channels.",
                ephemeral=True,
            )
            return

        memories = db.get_memories(
            str(interaction.guild.id),
            str(interaction.channel.id),
        )

        if not memories:
            await interaction.response.send_message(
                "My database does not contain any recorded lore or memories for this channel."
            )
            return

        lines = [f"**Recorded memories for this channel** ({len(memories)} entries):"]
        for mem in memories[:25]:
            lines.append(f"• **{mem['key']}**: {mem['value']}")

        if len(memories) > 25:
            lines.append(f"_...and {len(memories) - 25} more._")

        await interaction.response.send_message("\n".join(lines))

    @bot.tree.command(
        name="improve",
        description=f"Add, remove, or evolve {BOT_NAME}'s personality for this server",
    )
    @app_commands.describe(
        action="add | remove | list | reset | apply",
        change="Trait text, or feedback for apply mode",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="add — add a trait", value="add"),
            app_commands.Choice(name="remove — drop a trait", value="remove"),
            app_commands.Choice(name="list — show traits", value="list"),
            app_commands.Choice(name="reset — wipe all traits", value="reset"),
            app_commands.Choice(name="apply — AI self-improve", value="apply"),
        ]
    )
    async def improve(
        interaction: discord.Interaction,
        action: str,
        change: str = "",
    ):
        if not interaction.guild:
            await interaction.response.send_message(
                "this only works in a server",
                ephemeral=True,
            )
            return

        if action in ("add", "remove", "apply") and not change.strip():
            await interaction.response.send_message(
                "A parameter value for `change` is required when using the add, remove, or apply actions.",
                ephemeral=True,
            )
            return

        if action == "apply":
            await interaction.response.defer(thinking=True)
        else:
            await interaction.response.defer()

        allowed, error = check_rate_limit(interaction.user.id)
        if not allowed:
            await interaction.followup.send(error, ephemeral=True)
            return

        guild_id = str(interaction.guild.id)
        reply = await improve_reply(
            guild_id=guild_id,
            action=action,
            change=change,
            user_id=str(interaction.user.id),
            username=interaction.user.display_name,
        )
        await interaction.followup.send(reply)


async def _handle_llm_command(
    interaction: discord.Interaction,
    bot: "LocalLLMBot",
    user_message: str,
    command: str,
    force_search: bool = False,
) -> None:
    """Shared handler for LLM-powered slash commands."""
    await interaction.response.defer(thinking=True)

    allowed, error = check_rate_limit(interaction.user.id)
    if not allowed:
        await interaction.followup.send(error, ephemeral=True)
        return

    if is_unsafe_request(user_message):
        await interaction.followup.send(REFUSAL_MESSAGE)
        return

    guild_id = str(interaction.guild.id) if interaction.guild else "dm"
    channel_id = (
        str(interaction.channel.id)
        if isinstance(interaction.channel, discord.abc.Messageable)
        else "unknown"
    )

    memory_context = ""
    channel_context = ""

    if interaction.guild and isinstance(interaction.channel, discord.TextChannel):
        memory_context = db.format_memories_for_prompt(
            guild_id, channel_id, limit=MEMORY_CONTEXT_LIMIT
        )
        recent = db.get_recent_messages(
            guild_id, channel_id, limit=CHANNEL_CONTEXT_LIMIT
        )
        channel_context = db.format_messages_for_prompt(recent)

    reply = await generate_reply(
        user_message=user_message,
        memory_context=memory_context,
        channel_context=channel_context,
        command_instruction=INSTRUCTIONS.get(command, INSTRUCTIONS["ask"]),
        guild_id=guild_id,
        force_search=force_search,
        command=command,
    )

    await interaction.followup.send(reply)


async def handle_mention_reply(
    message: discord.Message,
    bot: "LocalLLMBot",
    cleaned_content: str,
) -> None:
    """Generate and send a reply when the bot is mentioned."""
    allowed, error = check_rate_limit(message.author.id)
    if not allowed:
        await message.reply(error, mention_author=False)
        return

    if is_unsafe_request(cleaned_content):
        await message.reply(REFUSAL_MESSAGE, mention_author=False)
        return

    async with message.channel.typing():
        guild_id = str(message.guild.id) if message.guild else "dm"
        channel_id = str(message.channel.id)

        memory_context = db.format_memories_for_prompt(
            guild_id, channel_id, limit=MEMORY_CONTEXT_LIMIT
        )
        recent = db.get_recent_messages(
            guild_id, channel_id, limit=CHANNEL_CONTEXT_LIMIT
        )
        channel_context = db.format_messages_for_prompt(recent)

        user_message = cleaned_content or "You were mentioned without any accompanying message text."

        reply = await generate_reply(
            user_message=user_message,
            memory_context=memory_context,
            channel_context=channel_context,
            command_instruction=INSTRUCTIONS["mention"],
            guild_id=guild_id,
            command="mention",
        )

        await message.reply(reply, mention_author=False)