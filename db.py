"""SQLite persistence for server memories and channel message logs."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "bot_data.db"
MAX_MESSAGES_PER_CHANNEL = 500


def get_bot_name() -> str:
    """Get active bot name from env variables."""
    return os.getenv("BOT_NAME", "Juan").strip()



def _connect() -> sqlite3.Connection:
    """Return a connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                bot_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(guild_id, channel_id, bot_name, key)
            );

            CREATE TABLE IF NOT EXISTS message_logs (
                id INTEGER PRIMARY KEY,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_message_logs_channel
                ON message_logs(guild_id, channel_id, id DESC);

            CREATE TABLE IF NOT EXISTS persona_traits (
                id INTEGER PRIMARY KEY,
                guild_id TEXT NOT NULL,
                bot_name TEXT NOT NULL,
                trait TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(guild_id, bot_name, trait)
            );
            """
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_memory(
    guild_id: str,
    channel_id: str,
    user_id: str,
    key: str,
    value: str,
) -> bool:
    """Insert or replace a server memory by key."""
    key = key.strip().lower()
    if not key:
        return False

    bot_name = get_bot_name()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO memories (guild_id, channel_id, bot_name, user_id, key, value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(guild_id, channel_id, bot_name, key) DO UPDATE SET
                value = excluded.value,
                user_id = excluded.user_id,
                created_at = excluded.created_at
            """,
            (guild_id, channel_id, bot_name, user_id, key, value.strip(), _now_iso()),
        )
    return True


def forget_memory(guild_id: str, channel_id: str, key: str) -> bool:
    """Delete a memory by key. Returns True if a row was removed."""
    key = key.strip().lower()
    bot_name = get_bot_name()
    with _connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM memories
            WHERE guild_id = ? AND channel_id = ? AND bot_name = ? AND key = ?
            """,
            (guild_id, channel_id, bot_name, key),
        )
        return cursor.rowcount > 0


def get_memories(
    guild_id: str,
    channel_id: Optional[str] = None,
) -> list[dict]:
    """Return memories for a guild, optionally filtered to one channel."""
    bot_name = get_bot_name()
    with _connect() as conn:
        if channel_id:
            rows = conn.execute(
                """
                SELECT key, value, user_id, channel_id, created_at
                FROM memories
                WHERE guild_id = ? AND channel_id = ? AND bot_name = ?
                ORDER BY key ASC
                """,
                (guild_id, channel_id, bot_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT key, value, user_id, channel_id, created_at
                FROM memories
                WHERE guild_id = ? AND bot_name = ?
                ORDER BY channel_id ASC, key ASC
                """,
                (guild_id, bot_name),
            ).fetchall()
    return [dict(row) for row in rows]


def format_memories_for_prompt(
    guild_id: str,
    channel_id: str,
    limit: int = 20,
) -> str:
    """Format relevant memories as context for the LLM."""
    memories = get_memories(guild_id, channel_id)
    if not memories:
        # Fall back to guild-wide lore if channel has none
        memories = get_memories(guild_id)[:limit]

    if not memories:
        return ""

    bot_name = get_bot_name()
    lines = [f"{bot_name} remembers:"]
    for mem in memories[:limit]:
        value = _clip_message(mem["value"], 120)
        lines.append(f"- {mem['key']}: {value}")
    return "\n".join(lines)


def log_message(
    guild_id: str,
    channel_id: str,
    user_id: str,
    username: str,
    content: str,
) -> None:
    """Store a channel message and prune old entries."""
    if not content or not content.strip():
        return

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO message_logs
                (guild_id, channel_id, user_id, username, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (guild_id, channel_id, user_id, username, content.strip(), _now_iso()),
        )

        # Keep only the most recent N messages per channel
        conn.execute(
            """
            DELETE FROM message_logs
            WHERE id IN (
                SELECT id FROM message_logs
                WHERE guild_id = ? AND channel_id = ?
                ORDER BY id DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (guild_id, channel_id, MAX_MESSAGES_PER_CHANNEL),
        )


def get_recent_messages(
    guild_id: str,
    channel_id: str,
    limit: int = 15,
) -> list[dict]:
    """Return the most recent messages for a channel (oldest first)."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT username, content, created_at
            FROM message_logs
            WHERE guild_id = ? AND channel_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (guild_id, channel_id, limit),
        ).fetchall()

    # Reverse so chronological order is preserved for the LLM
    return [dict(row) for row in reversed(rows)]


def _clip_message(content: str, max_chars: int = 200) -> str:
    """Truncate long messages so prompts stay fast."""
    content = content.strip().replace("\n", " ")
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 3].rstrip() + "..."


def add_persona_trait(guild_id: str, trait: str, user_id: str) -> bool:
    """Add a personality trait for a server. Returns False if duplicate or empty."""
    trait = trait.strip()
    if not trait or len(trait) > 200:
        return False

    bot_name = get_bot_name()
    with _connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO persona_traits (guild_id, bot_name, trait, user_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (guild_id, bot_name, trait, user_id, _now_iso()),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def remove_persona_trait(guild_id: str, needle: str) -> list[str]:
    """Remove traits matching needle (case-insensitive substring). Returns removed names."""
    needle = needle.strip().lower()
    if not needle:
        return []

    bot_name = get_bot_name()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT trait FROM persona_traits WHERE guild_id = ? AND bot_name = ?",
            (guild_id, bot_name),
        ).fetchall()

        removed: list[str] = []
        for row in rows:
            trait = row["trait"]
            if needle in trait.lower():
                conn.execute(
                    "DELETE FROM persona_traits WHERE guild_id = ? AND bot_name = ? AND trait = ?",
                    (guild_id, bot_name, trait),
                )
                removed.append(trait)
        return removed


def get_persona_traits(guild_id: str) -> list[dict]:
    """Return all persona traits for a server."""
    bot_name = get_bot_name()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT trait, user_id, created_at
            FROM persona_traits
            WHERE guild_id = ? AND bot_name = ?
            ORDER BY id ASC
            """,
            (guild_id, bot_name),
        ).fetchall()
    return [dict(row) for row in rows]


def reset_persona_traits(guild_id: str) -> int:
    """Delete all persona traits for a server. Returns count removed."""
    bot_name = get_bot_name()
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM persona_traits WHERE guild_id = ? AND bot_name = ?",
            (guild_id, bot_name),
        )
        return cursor.rowcount


def format_messages_for_prompt(messages: list[dict], max_chars: int = 200) -> str:
    """Format recent channel messages as LLM context."""
    if not messages:
        return ""

    lines = ["Recent chat:"]
    for msg in messages:
        lines.append(f"{msg['username']}: {_clip_message(msg['content'], max_chars)}")
    return "\n".join(lines)