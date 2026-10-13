# Medicine clock. This is not an agent. It only calls log_reminder at two times.
# Run: python reminder.py
# Try once: python reminder.py once morning

import asyncio  # run the async MCP call from this normal script
import sys  # read "once morning" from the command line

from apscheduler.schedulers.blocking import BlockingScheduler  # wait until the clock times

import storage  # read the morning and evening times
from mcp_client import call_named_tool  # call the reminder tool on the MCP server


def _hour_minute(text: str, fallback_hour: int) -> tuple[int, int]:
    # Use the saved HH:MM text. Use the fallback hour when the cell is blank.
    raw = (text or "").strip()
    if not raw or ":" not in raw:
        return fallback_hour, 0
    # Split "09:00" into hour and minute.
    hour_text, minute_text = raw.split(":", 1)
    return int(hour_text), int(minute_text)


def remind(slot: str) -> str:
    # slot is "morning" or "evening". The tool writes the CSV and the email.
    text = asyncio.run(call_named_tool("log_reminder", {"slot": slot}))
    # Show the tool result in this window.
    print(text)
    return text


def start_clock() -> None:
    # Read the two times from data/patient.csv.
    patient = storage.read_patient()
    morning_hour, morning_minute = _hour_minute(patient.get("morning_time", ""), 9)
    evening_hour, evening_minute = _hour_minute(patient.get("evening_time", ""), 21)
    # This scheduler stays running and fires the jobs.
    scheduler = BlockingScheduler()
    scheduler.add_job(remind, "cron", hour=morning_hour, minute=morning_minute, args=["morning"])
    scheduler.add_job(remind, "cron", hour=evening_hour, minute=evening_minute, args=["evening"])
    # Tell the user which times are active.
    print(
        f"Medicine clock is running at {morning_hour:02d}:{morning_minute:02d} "
        f"and {evening_hour:02d}:{evening_minute:02d}. Leave this window open."
    )
    # Block here until the window is closed.
    scheduler.start()


# Run the clock, or one reminder, only when this file is started directly.
if __name__ == "__main__":
    # "python reminder.py once morning" sends one reminder and exits.
    if len(sys.argv) == 3 and sys.argv[1] == "once":
        slot = sys.argv[2].strip().lower()
        if slot not in ("morning", "evening"):
            raise SystemExit("Use morning or evening.")
        remind(slot)
    else:
        # No extra words means keep the clock running.
        start_clock()
