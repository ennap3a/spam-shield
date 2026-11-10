"""Inbox Checker: reads messages from a CSV file and asks the Spam Shield API about each one.

This program does NOT import spam_shield. It only talks to the API over HTTP,
exactly like a phone app or a website would.
"""

import argparse
import csv
import os
import sys

import requests

API_URL = os.environ.get("SPAM_SHIELD_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 30

RISK_ICONS = {"safe": "✅", "suspicious": "⚠️ ", "high": "🚨"}


def read_inbox(path: str) -> list[dict]:
    """Read the CSV file and return one dict per message: {"sender": ..., "text": ...}."""
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def api_is_running() -> bool:
    """Ask /health whether the API is awake."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT_SECONDS)
    except requests.ConnectionError:
        return False
    return response.status_code == 200


def check_messages(texts: list[str]) -> list[dict]:
    """Send all messages in ONE request to /predict/batch and return the results."""
    response = requests.post(
        f"{API_URL}/predict/batch", json={"texts": texts}, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()["results"]


def print_report(messages: list[dict], results: list[dict]) -> None:
    """Print one line per message, then a summary."""
    counts = {"safe": 0, "suspicious": 0, "high": 0}

    for message, result in zip(messages, results, strict=True):
        level = result["risk_level"]
        counts[level] += 1
        icon = RISK_ICONS[level]
        print(f"{icon} {result['label']:<9} {result['risk_score']:>4.0%}  {message['sender']}")

    print()
    print(f"📬 Checked {len(messages)} messages")
    print(
        f"✅ Safe: {counts['safe']}   "
        f"⚠️  Suspicious: {counts['suspicious']}   "
        f"🚨 High risk: {counts['high']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an inbox CSV with the Spam Shield API.")
    parser.add_argument("csv_file", help="CSV file with 'sender' and 'text' columns")
    args = parser.parse_args()

    if not api_is_running():
        print(f"❌ Can't reach the API at {API_URL}. Is uvicorn running?")
        sys.exit(1)

    messages = read_inbox(args.csv_file)
    texts = [message["text"] for message in messages]
    results = check_messages(texts)
    print_report(messages, results)


if __name__ == "__main__":
    main()
