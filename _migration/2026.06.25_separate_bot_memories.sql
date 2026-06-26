-- ====================================================================
-- MIGRATION: Separate Bot Memories and Persona Traits
-- DATE: 2026-06-25
--
-- PURPOSE:
--   Adds a 'bot_name' column to 'memories' and 'persona_traits' tables,
--   updates their unique constraints, and migrates existing entries 
--   to be owned by 'Juan' by default. This prevents bots from sharing
--   lore/memories or server-specific personality traits.
--
-- EXECUTION:
--   Run this script against the SQLite database file `bot_data.db`:
--   $ sqlite3 bot_data.db < _migration/2026.06.25_separate_bot_memories.sql
--
-- ROLLBACK:
--   To undo these changes, run the rollback script:
--   $ sqlite3 bot_data.db < _migration/2026.06.25_separate_bot_memories_rollback.sql
-- ====================================================================

PRAGMA foreign_keys=OFF;

BEGIN TRANSACTION;

-- --------------------------------------------------------------------
-- 1. Migrate memories Table
-- --------------------------------------------------------------------

-- Rename old table
ALTER TABLE memories RENAME TO memories_old;

-- Create new table with bot_name column and updated unique constraint
CREATE TABLE memories (
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

-- Copy existing memories and attribute them to 'Juan' by default
INSERT INTO memories (id, guild_id, channel_id, bot_name, user_id, key, value, created_at)
SELECT id, guild_id, channel_id, 'Juan', user_id, key, value, created_at
FROM memories_old;

-- Drop old table
DROP TABLE memories_old;

-- --------------------------------------------------------------------
-- 2. Migrate persona_traits Table
-- --------------------------------------------------------------------

-- Rename old table
ALTER TABLE persona_traits RENAME TO persona_traits_old;

-- Create new table with bot_name column and updated unique constraint
CREATE TABLE persona_traits (
    id INTEGER PRIMARY KEY,
    guild_id TEXT NOT NULL,
    bot_name TEXT NOT NULL,
    trait TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(guild_id, bot_name, trait)
);

-- Copy existing persona traits and attribute them to 'Juan' by default
INSERT INTO persona_traits (id, guild_id, bot_name, trait, user_id, created_at)
SELECT id, guild_id, 'Juan', trait, user_id, created_at
FROM persona_traits_old;

-- Drop old table
DROP TABLE persona_traits_old;

COMMIT;
