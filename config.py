import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# ANTHROPIC CONFIGURATION
# ============================================================

ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY"
)

if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found in .env"
    )


CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-sonnet-4-6"
)


# ============================================================
# DEVELOPMENT CONFIGURATION
# ============================================================

DEBUG_MODE = (
    os.getenv(
        "DEBUG_MODE",
        "true"
    ).lower() == "true"
)


MAX_PUBMED_RESULTS = int(
    os.getenv(
        "MAX_PUBMED_RESULTS",
        "2"
    )
)


# ============================================================
# LLM
# ============================================================

def get_llm():

    return ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,
        api_key=ANTHROPIC_API_KEY
    )