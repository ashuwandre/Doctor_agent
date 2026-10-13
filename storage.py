# CSV storage. The tools read and write these files. This file does not choose a tool.

import csv  # read and write comma-separated files
import os  # build folder paths
from datetime import datetime  # timestamp for each saved row

# Folder that contains this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder for every data file.
DATA_DIR = os.path.join(BASE_DIR, "data")
# Folder for email text when SMTP is not filled in.
EMAIL_DIR = os.path.join(DATA_DIR, "emails")
# Folder for uploaded lab photos and PDFs.
REPORT_DIR = os.path.join(DATA_DIR, "reports")

# One path for each CSV file.
PATIENT_CSV = os.path.join(DATA_DIR, "patient.csv")
DECISIONS_CSV = os.path.join(DATA_DIR, "decisions.csv")
CHAT_CSV = os.path.join(DATA_DIR, "chat_history.csv")
APPOINTMENTS_CSV = os.path.join(DATA_DIR, "appointments.csv")
REMINDERS_CSV = os.path.join(DATA_DIR, "reminders.csv")
DOCTORS_CSV = os.path.join(DATA_DIR, "doctors.csv")

# Column names. Every writer uses these same names.
PATIENT_COLUMNS = [
    "mother_name",
    "notify_email",
    "tac_min",
    "tac_max",
    "creatinine_max",
    "medicine",
    "morning_time",
    "evening_time",
]
DECISION_COLUMNS = [
    "timestamp",
    "tac",
    "creatinine",
    "tac_high",
    "creatinine_high",
    "summary",
    "action",
    "appointment_time",
]
CHAT_COLUMNS = ["timestamp", "role", "message"]
APPOINTMENT_COLUMNS = ["timestamp", "when_time", "reason"]
REMINDER_COLUMNS = ["timestamp", "slot", "medicine", "message"]
DOCTOR_COLUMNS = ["name", "email", "available", "times"]


def now_text() -> str:
    # Current local time as text for a CSV cell.
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def ensure_csv(path: str, columns: list[str]) -> None:
    # Make the folder if it is missing.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Leave the file alone when it already has a header.
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    # Create the file and write only the header row.
    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()


def read_rows(path: str, columns: list[str]) -> list[dict[str, str]]:
    # Make sure the file exists before reading.
    ensure_csv(path, columns)
    # Return the data rows. The header is not included.
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def append_row(path: str, columns: list[str], row: dict[str, str]) -> None:
    # Make sure the header exists before adding a row.
    ensure_csv(path, columns)
    # Add one row. A missing key is stored as empty text.
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow({key: row.get(key, "") for key in columns})


def init_files() -> None:
    # Create the email and report folders.
    os.makedirs(EMAIL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    # Create the history files if they are missing.
    ensure_csv(DECISIONS_CSV, DECISION_COLUMNS)
    ensure_csv(CHAT_CSV, CHAT_COLUMNS)
    ensure_csv(APPOINTMENTS_CSV, APPOINTMENT_COLUMNS)
    ensure_csv(REMINDERS_CSV, REMINDER_COLUMNS)
    ensure_csv(DOCTORS_CSV, DOCTOR_COLUMNS)
    # Create a blank patient row only when the file is missing.
    if not os.path.exists(PATIENT_CSV) or os.path.getsize(PATIENT_CSV) == 0:
        ensure_csv(PATIENT_CSV, PATIENT_COLUMNS)
        append_row(
            PATIENT_CSV,
            PATIENT_COLUMNS,
            {
                "mother_name": "",
                "notify_email": "",
                "tac_min": "",
                "tac_max": "",
                "creatinine_max": "",
                "medicine": "tablets",
                "morning_time": "09:00",
                "evening_time": "21:00",
            },
        )


def read_patient() -> dict[str, str]:
    # Make sure the patient file exists.
    init_files()
    # Read every patient row. This project keeps one row.
    rows = read_rows(PATIENT_CSV, PATIENT_COLUMNS)
    # Return an empty row when the file has only a header.
    if not rows:
        return {key: "" for key in PATIENT_COLUMNS}
    # Return the first row.
    return rows[0]


def save_email_copy(subject: str, body: str, to_addr: str) -> str:
    # Make sure the emails folder exists.
    init_files()
    # Use the time in the file name so each copy is unique.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EMAIL_DIR, f"{stamp}.txt")
    # Write the address, subject, and body as plain text.
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"To: {to_addr}\nSubject: {subject}\n\n{body}\n")
    # Return the path so the tool can tell the agent where the copy is.
    return path


# Create the folders and CSV files as soon as this module is imported.
init_files()
