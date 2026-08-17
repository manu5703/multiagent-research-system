import hashlib
import json
from pathlib import Path


CACHE_DIR = Path("cache")

CACHE_DIR.mkdir(
    exist_ok=True
)


def _cache_key(prompt):

    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


def get_cached(prompt):

    key = _cache_key(prompt)

    path = CACHE_DIR / f"{key}.json"

    if not path.exists():

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data["response"]


def save_cached(
    prompt,
    response
):

    key = _cache_key(prompt)

    path = CACHE_DIR / f"{key}.json"

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "response": response
            },
            f,
            indent=2
        )