#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
# Source your env file with OPENAI_API_KEY
source ~/.env.machomelab 2>/dev/null || source ~/clawd/.env.machomelab 2>/dev/null
cd /Users/kayleighbot/Projects/claweval
source .venv/bin/activate
exec python3 -m claweval run --resume --scoring hybrid >> /tmp/claweval-final.log 2>&1
