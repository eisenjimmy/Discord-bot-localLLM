"""
Local LLM Discord Bot — entry point.

Replies only when mentioned, prefixed, or via slash commands.
Logs channel messages for context and stores server lore in SQLite.
"""

import os
from dotenv import load_dotenv

# Load environment variables from custom env file if specified, else default to .env
env_file = os.getenv("ENV_FILE", ".env")
load_dotenv(env_file)

import asyncio
import logging
import sys

import discord
from discord.ext import commands

import db
from commands import handle_mention_reply, register_commands
from llm import check_ollama_health, close_client, keepalive_loop, warmup_model

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
BOT_PREFIX = os.getenv("BOT_PREFIX", "!bot")
BOT_NAME = os.getenv("BOT_NAME", "Juan").strip()
GUILD_ID = os.getenv("GUILD_ID", "")  # optional — instant slash command sync for dev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class LocalLLMBot(commands.Bot):
    """Discord bot with local Ollama backend."""

    def __init__(self) -> None:
        # message_content is required for @mentions and !bot prefix replies.
        # Enable it in the Discord Developer Portal under Bot → Privileged Gateway Intents.
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=BOT_PREFIX, intents=intents)

    async def setup_hook(self) -> None:
        """Initialize DB and register commands — don't block login on slow tasks."""
        logger.info("setup_hook: initializing database...")
        db.init_db()
        register_commands(self)
        logger.info("setup_hook: commands registered")

        # Sync slash commands in background so Discord login doesn't hang
        self.loop.create_task(self._sync_commands_background())
        self.loop.create_task(keepalive_loop())

    async def _sync_commands_background(self) -> None:
        """Sync slash commands without blocking the gateway connection."""
        try:
            if GUILD_ID:
                guild = discord.Object(id=int(GUILD_ID))
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info("Synced %d slash command(s) to guild %s", len(synced), GUILD_ID)
            else:
                synced = await self.tree.sync()
                logger.info("Synced %d slash command(s) globally", len(synced))
        except Exception as exc:
            logger.error("Slash command sync failed: %s", exc)

    async def _warmup_background(self) -> None:
        """Warm up LLM in background — don't block Discord ready state."""
        if await check_ollama_health():
            logger.info("llama-server is reachable — warming up in background...")
            await warmup_model()
        else:
            logger.warning(
                "llama-server not reachable. Start Gemma on :8080 for replies."
            )

    async def on_ready(self) -> None:
        """Log startup — bot is live even if LLM warmup is still running."""
        logger.info("Logged in as %s (id: %s)", self.user, self.user.id if self.user else "?")
        logger.info("Prefix: %s | Juan is online.", BOT_PREFIX)
        self.loop.create_task(self._warmup_background())


bot = LocalLLMBot()


@bot.event
async def on_message(message: discord.Message) -> None:
    """Log messages and respond only to mentions or prefix commands."""
    # Ignore the bot's own messages
    if message.author.bot:
        return

    # Log messages from guild text channels for context
    if message.guild and isinstance(message.channel, discord.TextChannel):
        db.log_message(
            guild_id=str(message.guild.id),
            channel_id=str(message.channel.id),
            user_id=str(message.author.id),
            username=message.author.display_name,
            content=message.content,
        )

    # Determine if we should respond
    should_respond = False
    cleaned_content = message.content

    # 1. Bot is mentioned
    if bot.user and bot.user in message.mentions:
        should_respond = True
        # Strip mention from content so the LLM sees clean text
        cleaned_content = message.content
        for mention in message.mentions:
            cleaned_content = cleaned_content.replace(f"<@{mention.id}>", "")
            cleaned_content = cleaned_content.replace(f"<@!{mention.id}>", "")
        cleaned_content = cleaned_content.strip()

    # 2. Message starts with configured prefix
    elif message.content.startswith(BOT_PREFIX):
        should_respond = True
        cleaned_content = message.content[len(BOT_PREFIX):].strip()

    if should_respond:
        await handle_mention_reply(message, bot, cleaned_content)
        return

    # Allow any legacy prefix commands (none defined, but keeps extensibility)
    await bot.process_commands(message)


def main() -> None:
    """Validate config and start the bot."""
    if not DISCORD_TOKEN:
        logger.error(
            "DISCORD_TOKEN is not set. Copy .env.example to .env and add your token."
        )
        sys.exit(1)

    logger.info("Connecting to Discord...")
    try:
        bot.run(DISCORD_TOKEN, log_handler=None, log_level=logging.INFO)
    except discord.PrivilegedIntentsRequired:
        logger.error(
            "Message Content Intent is not enabled for this bot.\n"
            "Fix: https://discord.com/developers/applications\n"
            "  → Your App → Bot → Privileged Gateway Intents\n"
            "  → Turn ON 'Message Content Intent'\n"
            "  → Save, then restart the bot."
        )
        sys.exit(1)
    except discord.LoginFailure:
        logger.error("Invalid Discord token. Check DISCORD_TOKEN in .env")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        try:
            asyncio.run(close_client())
        except Exception:
            pass


if __name__ == "__main__":
    main()