import os
import sys
import json
import shutil
import urllib.request

# Allow overriding the URL via environment variable
CATALOG_URL = os.getenv(
    "SHL_CATALOG_URL",
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)

TIMEOUT_SEC = 30
MAX_RETRIES = 3


def download_with_retry(url: str, timeout: int, retries: int) -> str:
    """Download raw text content from url, retrying up to `retries` times."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SHL-Copilot/1.0"}
    )
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            last_err = e
            print(f"  Attempt {attempt}/{retries} failed: {e}")
    raise RuntimeError(f"All {retries} download attempts failed. Last error: {last_err}")


def sanitize_json(raw: str) -> str:
    """Remove raw control characters that cause strict JSON parse failures."""
    # Replace tabs and carriage returns; keep newlines (valid in JSON whitespace)
    return raw.replace("\t", " ").replace("\r", "")


def validate_catalog(data: object) -> None:
    """Raise ValueError if the parsed data doesn't look like a valid catalog."""
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at the root, got {type(data).__name__}."
        )
    if len(data) == 0:
        raise ValueError("Catalog array is empty — the remote may have returned an error page.")

    # Spot-check first item for expected fields
    required_keys = {"name", "description"}
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError(f"Expected catalog items to be objects, got {type(first).__name__}.")
    missing = required_keys - first.keys()
    if missing:
        raise ValueError(
            f"First catalog item is missing expected fields: {missing}. "
            "The remote may have returned malformed data."
        )


def main():
    dest_dir  = os.path.join(os.path.dirname(__file__), "data")
    dest_path = os.path.join(dest_dir, "shl_product_catalog.json")
    tmp_path  = dest_path + ".tmp"
    bak_path  = dest_path + ".bak"

    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading SHL product catalog from:\n  {CATALOG_URL}\n")

    try:
        # 1. Download with retries and explicit timeout
        raw = download_with_retry(CATALOG_URL, TIMEOUT_SEC, MAX_RETRIES)

        # 2. Sanitize control characters before parsing
        raw = sanitize_json(raw)

        # 3. Parse JSON
        try:
            catalog = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e

        # 4. Validate the catalog structure
        validate_catalog(catalog)
        print(f"Validation passed — {len(catalog)} products found.")

        # 5. Backup existing file before overwriting
        if os.path.exists(dest_path):
            shutil.copy2(dest_path, bak_path)
            print(f"Backed up existing catalog → {bak_path}")

        # 6. Atomic write: write to .tmp then rename to avoid partial writes
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, dest_path)

        print(f"Catalog saved successfully → {dest_path}")

    except Exception as e:
        # Clean up temp file if it was created
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
