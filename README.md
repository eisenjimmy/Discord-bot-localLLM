# Discord Local LLM Bot

A highly capable, personality-driven, agent-enabled Discord bot that runs entirely on local hardware using `llama-server` and a specialized agentic LLM.

This repository supports running multiple independent bot instances (e.g., **Juan** and **Jisoo**) concurrently, complete with database-backed memory sync, dynamic server-specific traits, web-browsing capabilities, real-time weather API lookups, and meme generation.

---

## Key Features

1. **Dual Personas & Co-Running Support**
   - **Juan**: A highly capable, warm, and laid-back Mexican dude who holds director Jimmy ("el jefe") in absolute high esteem. Intelligent, cheerful, and polite.
   - **Jisoo**: Self-aware global Korean pop star from BLACKPINK. Graceful, composed, and quietly charming with a playful, quirky "4D" personality.
   - **Mutual Affection**: The bots carry friendly liking traits for each other, creating adorable, witty, and charming interactions when talking in the same channel.
   - **No Duplicates**: The launcher scripts prevent starting duplicate instances of the same bot but fully support co-running both bots simultaneously.

2. **Agentic Tool Integration (Zero Setup)**
   - **Web Search**: The model autonomously decides when to trigger a DuckDuckGo web search (with Google Custom Search API support if configured).
   - **Webpage Fetching**: Reads page content directly by parsing HTML into plain text (up to 8k characters).
   - **Weather Lookups**: Retrieves real-time weather details for any location via `wttr.in`.
   - **Meme/GIF Generation**: Automatically designs, creates, and sends memes with contextual captions using the `memegen.link` API.

3. **Memory & Lore System**
   - **SQLite Message Log**: Automatically logs channel messages to build short-term multi-turn conversation context.
   - **Server Lore Database**: Users can save recurring facts, jokes, or memories using `/remember <key> <value>` and review them via `/lore`.
   - **AI Self-Improvement**: Evolve bot behaviors dynamically on a server-by-server basis using `/improve apply <feedback>`.

---

## Recommended Model

For optimal agentic performance, tool utilization, and personality-rich responses, it is highly recommended to use the following model:

- **Model**: Gemma-4 12B Agentic Fable5 Composer2.5 v2 3.5x Tau2 (GGUF)
- **HuggingFace Repository**: [yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
- **Recommended Quantization**: `Q4_K_M` or `Q5_K_M` GGUF.

> [!NOTE]
> GGUF model files and the `models/` directory are left out of `.gitignore` so they are not ignored, allowing you to track and manage models directly in the repository if desired.

---

## Step-by-Step Setup Guide

Follow this guide to get the server and bot instances up and running.

### Step 1: Download the LLM Model
1. Visit the HuggingFace repository: [yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).
2. Download your preferred quantization file (e.g. `gemma4-v2-Q4_K_M.gguf`).
3. Place the file inside the bot's own `models/` directory, for example: `~/Applications/discord-local-llm-bot/models/gemma4-v2-Q4_K_M.gguf`.

---

### Step 2: Set Up Your Discord Applications
You can set up one or two applications (if running both Juan and Jisoo).

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** and give it a name (e.g., "Juan" or "Jisoo").
3. Under the **Bot** tab in the sidebar:
   - Click **Reset Token** and copy the token value (save it securely).
   - Under **Privileged Gateway Intents**, enable **Message Content Intent** (required so the bot can see mentions and read prefix commands).
4. Invite the bot to your server:
   - Go to **OAuth2 → URL Generator**.
   - Select the `bot` and `applications.commands` scopes.
   - Check the following bot permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`, and `Attach Files` (for memes).
   - Copy the generated URL and open it in a web browser to authorize the bot on your server.

---

### Step 3: Run the Local LLM Server
Ensure you have `llama.cpp` installed (or use a compatible local executor like Jarvis). Start the OpenAI-compatible server on port `8080`:

```bash
# Example running llama-server manually
llama-server \
  --model ~/Applications/discord-local-llm-bot/models/gemma4-v2-Q4_K_M.gguf \
  --ctx-size 8192 \
  --port 8080 \
  --parallel 1
```

Alternatively, if you use the built-in Jarvis scripts directory structure:
```bash
chmod +x scripts/start-llm.sh
./scripts/start-llm.sh
```

Ensure the server is running by opening `http://127.0.0.1:8080/health` in your browser.

---

### Step 4: Configure Project Environments
Navigate to your project root folder and create separate environment files for your bots.

```bash
cd ~/Applications/discord-local-llm-bot

# Install virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy example environment configuration
cp .env.example .env
```

#### Bot 1 Configuration (`.env` — Juan)
Edit your `.env` file:
```env
DISCORD_TOKEN=your_juan_bot_token_here
APPLICATION_ID=your_juan_application_id_here
GUILD_ID=your_test_server_id_here  # Optional: Set to sync slash commands instantly to this server
BOT_NAME=Juan
BOT_PREFIX=!bot

# Local llama-server details
LLM_URL=http://127.0.0.1:8080/v1/chat/completions
LLM_HEALTH_URL=http://127.0.0.1:8080/health
LLM_MODEL=gemma4-agentic-v2-q4
```

#### Bot 2 Configuration (`.env.jisoo` — Jisoo)
Create a `.env.jisoo` file:
```env
DISCORD_TOKEN=your_jisoo_bot_token_here
APPLICATION_ID=your_jisoo_application_id_here
GUILD_ID=your_test_server_id_here  # Optional: Set to sync slash commands instantly to this server
BOT_NAME=Jisoo
BOT_PREFIX=!jisoo

# Local llama-server details
LLM_URL=http://127.0.0.1:8080/v1/chat/completions
LLM_HEALTH_URL=http://127.0.0.1:8080/health
LLM_MODEL=gemma4-agentic-v2-q4
```

---

### Step 5: Start the Discord Bots
You can run the bots using standard terminal commands or using desktop shortcut scripts.

#### Option A: Run via CLI (Terminal)
Activate your virtual environment and specify the target env file:

```bash
# Start Juan
source .venv/bin/activate
ENV_FILE=.env python main.py

# In another terminal window, start Jisoo
source .venv/bin/activate
ENV_FILE=.env.jisoo python main.py
```

#### Option B: Run via Desktop Commands (macOS)
If you have `Juan.command` and `Jisoo.command` on your Desktop:
1. Make them executable (run once in terminal):
   ```bash
   chmod +x ~/Desktop/Juan.command
   chmod +x ~/Desktop/Jisoo.command
   ```
2. Simply double-click `Juan.command` or `Jisoo.command` to automatically check/start `llama-server`, load the correct environment, and launch the bot!

---

## Interactive Features & Slash Commands

The bots do not respond to every message in a channel. They will only reply if **mentioned** (e.g., `@Juan`), if a message begins with the prefix (e.g., `!bot`), or via a **slash command**.

### Slash Command Catalog

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/ask` | `question` | Ask a normal question. The bot will reply with detailed, personality-rich logic. |
| `/search` | `query` | Force the bot to perform a web search and compile the findings. |
| `/meme` | `idea`, `template`, `top`, `bottom` | Generate a meme. Let the LLM choose the captions based on your `idea` or define them manually. |
| `/roast` | `target` (optional) | Ask the bot to deliver a playful, safe roast. |
| `/rank` | `idea` | Evaluates your idea and scores it 1-10 with context. |
| `/summarize` | `count` (optional, max 100) | Summarizes the last N recorded messages in the current channel. |
| `/remember` | `key`, `value` | Commit a server joke, fact, or custom context to the SQLite database. |
| `/forget` | `key` | Delete a stored lore entry. |
| `/lore` | *None* | List all stored lore and memories for the current channel. |
| `/improve` | `action`, `change` | Dynamically evolve/tweak the bot's behavior for the server (`add`, `remove`, `list`, `reset`, `apply`). |

---

## Troubleshooting

### "llama-server is not reachable"
- Check that your local server is running on port 8080: `curl http://127.0.0.1:8080/health`.
- If running manually, verify that the model path in your command matches where you placed the `.gguf` file.
- Verify `LLM_URL` in your `.env` or `.env.jisoo` file points to the correct address.

### The Bot doesn't reply to Mentions or Prefixes
- Go to the **Discord Developer Portal → Bot**.
- Verify that **Message Content Intent** is toggled **ON** and saved.
- Restart the bot process.

### Duplicate bot processes running
- If you start the bot and it complains about duplicate instances, check your background processes:
  ```bash
  ps aux | grep main.py
  ```
- Kill conflicting processes using `kill <PID>` or close existing terminal windows running the script.

### Slash commands are missing
- When syncing slash commands globally, it can take Discord up to **1 hour** to populate them on all servers.
- For instant testing during development, configure the `GUILD_ID` environment variable in your `.env` file to sync commands immediately to your test server.

---

## AI Handoff Prompt

If you are handing off this repository to another AI coding assistant (like Claude, Gemini, or ChatGPT) to set up, extend, or debug this project, you can copy-paste the following prompt:

```text
You are an expert developer helping me configure and develop my Discord Local LLM Bot project.
This repository implements a multi-bot Discord bot system supporting co-running bots (e.g. Juan and Jisoo) locally on macOS.

Key Architecture:
1. Entry point: main.py, loads custom configuration files via the ENV_FILE environment variable (e.g. ENV_FILE=.env or ENV_FILE=.env.jisoo).
2. Database: db.py (SQLite) stores chat logs, custom memories (/remember), and server-specific learned traits (/improve). Queries are isolated by the dynamic BOT_NAME environment variable.
3. completions: llm.py connects to llama-server (OpenAI-compatible endpoints) on port 8080.
4. Agent Tools: tools.py / search.py / meme.py support web search (DuckDuckGo or Google API), fetching webpages, weather lookups (wttr.in), and memegen.link meme generation.
5. Dynamic Personas: Juan (friendly Mexican dude) and Jisoo (Korean pop star from BLACKPINK with a quirky 4D personality).

Please read the README.md and the code files in the repository. Help me with:
- Setting up the Python virtual environment (.venv) and installing requirements.txt.
- Configuring the .env and .env.jisoo credentials.
- Running the llama-server with the recommended GGUF model: yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF.
- Extending slash commands in commands.py or adding custom tools in tools.py.
```