-- ====================================================================
-- ROLLBACK MIGRATION: Revert Bot-Specific Memories and Persona Traits
-- DATE: 2026-06-25
--
-- PURPOSE:
--   Reverts 'memories' and 'persona_traits' back to the shared format,
--   removing the 'bot_name' column. If duplicate keys exist across
--   different bots, the newest entry will be preserved.
-- ====================================================================

PRAGMA foreign_keys=OFF;

BEGIN TRANSACTION;

-- --------------------------------------------------------------------
-- 1. Rollback memories Table
-- --------------------------------------------------------------------

ALTER TABLE memories RENAME TO memories_old;

CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(guild_id, channel_id, key)
);

-- Copy data, discarding older duplicates if there are conflicting keys per channel/guild
INSERT OR REPLACE INTO memories (id, guild_id, channel_id, user_id, key, value, created_at)
SELECT id, guild_id, channel_id, user_id, key, value, created_at
FROM memories_old
ORDER BY created_at ASC;

DROP TABLE memories_old;

-- --------------------------------------------------------------------
-- 2. Rollback persona_traits Table
-- --------------------------------------------------------------------

ALTER TABLE persona_traits RENAME TO persona_traits_old;

CREATE TABLE persona_traits (
    id INTEGER PRIMARY KEY,
    guild_id TEXT NOT NULL,
    trait TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(guild_id, trait)
);

-- Copy data, discarding older duplicates if there are conflicting traits per guild
INSERT OR REPLACE INTO persona_traits (id, guild_id, trait, user_id, created_at)
SELECT id, guild_id, trait, user_id, created_at
FROM persona_traits_old;

DROP TABLE persona_traits_old;

COMMIT;
