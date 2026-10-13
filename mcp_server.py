"""MCP server: a list of tools the doctor agent can call.

This file does not decide which tool runs.
The agent sends every tool name and description to the chat model.
The model reads those descriptions and returns the tool it wants.
MCPServer copies each function docstring into the tool description the model sees.

mcp_client.py starts this server over stdio. You can also run it alone:

    python mcp_server.py
"""

import json
import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Annotated

from dotenv import load_dotenv
try:
    # mcp 1.x, which is what the financial-adviser environment has.
    from mcp.server.fastmcp import FastMCP as MCPServer
except ModuleNotFoundError:
    # mcp 2.x renamed FastMCP to MCPServer.
    from mcp.server.mcpserver import MCPServer
from pydantic import Field

import storage


# Load .env from this folder so mail settings work when the agent starts the server.
load_dotenv(os.path.join(storage.BASE_DIR, ".env"))

# instructions are sent with the tool list. The model still picks a tool from each description.
mcp = MCPServer(
    "doctor-tools",
    instructions=(
        "Choose a tool only by reading its description. "
        "Call the one tool whose description matches the current job. "
        "Do not call a tool for a job its description tells you to avoid."
    ),
)


def _notify_address() -> str:
    """Address for summaries. .env wins, then the email cell in patient.csv."""
    for name in ("NOTIFY_EMAIL", "EMAIL_FROM", "SMTP_USER"):
        from_env = os.getenv(name, "").strip()
        if from_env:
            return from_env
    return storage.read_patient().get("notify_email", "").strip()


def deliver_email(subject: str, body: str) -> str:
    """Send one email, or save the text if SMTP is not filled in.

    Tools call this helper. It is not itself a tool, so the model cannot pick it.
    """
    to_addr = _notify_address()
    host = os.getenv("SMTP_HOST", "").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    port_text = os.getenv("SMTP_PORT", "587").strip() or "587"
    # EMAIL_MODE=local keeps the message in data/emails and does not send it.
    local_only = os.getenv("EMAIL_MODE", "").strip().lower() == "local"
    ready = bool(host and user and password and to_addr) and not local_only

    if not ready:
        path = storage.save_email_copy(subject, body, to_addr)
        if local_only:
            return f"Email was saved locally, not sent. A copy was saved at {path}."
        return (
            "Email was not sent because SMTP settings or the notify address are empty. "
            f"A copy was saved at {path}."
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = user
    message["To"] = to_addr
    message.set_content(body)
    try:
        with smtplib.SMTP(host, int(port_text), timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
    except Exception as exc:
        path = storage.save_email_copy(subject, body, to_addr)
        return f"Email failed ({exc}). A copy was saved at {path}."
    return f"Email sent to {to_addr}."


def _wiki(title: str) -> str:
    """One Wikipedia page as a short line. Empty string if the page cannot be read."""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    request = urllib.request.Request(url, headers={"User-Agent": "doctor-agent-study/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = str(data.get("extract", "")).strip()[:500]
    if not text:
        return ""
    return f"- {data.get('title', title)}: {text}"


# Public background from Wikipedia. Her own limits are read_patient_limits.
@mcp.tool(structured_output=False)
def search_reference_ranges(
    query: Annotated[
        str,
        Field(description="Short public search question about tacrolimus (tac) or creatinine ranges."),
    ],
) -> str:
    """Search Wikipedia for general text about tacrolimus (tac) or creatinine ranges.

    Pick this tool when you need published background on those ranges.
    Do not pick this tool for this patient's own limits. Use read_patient_limits for those.
    Do not pick this tool to send email, save a lab decision, save chat, book a doctor time, or log a medicine reminder.
    The text you get back is general reading. It is not an instruction for this patient.
    """
    words = query.lower().replace(",", " ").split()
    searches = []
    if any(word.startswith("tac") for word in words):
        searches.append("Tacrolimus")
    if any(word.startswith("creatin") for word in words):
        searches.append("Creatinine")
    if not searches and query.strip():
        searches.append(query.strip())

    last_error = ""
    lines = []
    for title in searches:
        try:
            line = _wiki(title)
        except Exception as exc:
            last_error = str(exc)
            line = ""
        if line:
            lines.append(line)

    if not lines:
        detail = last_error or "no results"
        return (
            f"Web search failed ({detail}). "
            "Continue with the limits from read_patient_limits."
        )

    return "General Wikipedia results. These are not this patient's own limits.\n" + "\n".join(lines)


@mcp.tool(structured_output=False)
def read_patient_limits() -> str:
    """Read the tac and creatinine limits saved for this patient.

    Pick this tool when you need the ranges her transplant doctor gave, or her medicine name and reminder times.
    Do not pick this tool to search the public web. Use search_reference_ranges for that.
    Blank tac_min, tac_max, or creatinine_max means the limit was not typed in yet. Do not treat a lab value as high in that case.
    """
    patient = storage.read_patient()
    lines = [f"{key}: {patient.get(key, '')}" for key in storage.PATIENT_COLUMNS]
    if not patient.get("tac_max", "").strip() or not patient.get("creatinine_max", "").strip():
        lines.append(
            "Blank number limits mean they are not set. "
            "Do not call a value high until those cells are filled from her doctor."
        )
    return "\n".join(lines)


@mcp.tool(structured_output=False)
def send_email(
    subject: Annotated[str, Field(description="Short email subject for the lab summary.")],
    body: Annotated[str, Field(description="Full email body: lab values, limits, web notes, and the decision.")],
) -> str:
    """Email the lab-report summary to the configured address.

    Pick this tool when a lab summary is ready to deliver.
    Do not pick this tool for the 9 AM or 9 PM medicine reminder. log_reminder already sends that.
    Do not pick this tool to look up ranges or to save history.
    """
    return deliver_email(subject, body)


@mcp.tool(structured_output=False)
def save_decision(
    tac: Annotated[str, Field(description="Tac level read from the report. Digits only, such as 8.2.")],
    creatinine: Annotated[str, Field(description="Creatinine read from the report. Digits only, such as 1.4.")],
    tac_high: Annotated[str, Field(description="yes if tac is above tac_max, otherwise no.")],
    creatinine_high: Annotated[str, Field(description="yes if creatinine is above creatinine_max, otherwise no.")],
    summary: Annotated[str, Field(description="Short summary of the report and what was decided.")],
    action: Annotated[str, Field(description="What was done, such as emailed summary or asked for a doctor time.")],
    appointment_time: Annotated[
        str,
        Field(description="Time the person gave for the doctor visit. Empty string if there is no visit."),
    ] = "",
) -> str:
    """Save one lab-report decision so later chat can read it.

    Pick this tool after you finish one report: values checked, summary written, email handled.
    Do not pick this tool for ordinary chat lines. Use save_chat for those.
    Do not pick this tool for the medicine clock. Use log_reminder for that.
    """
    stamp = storage.now_text()
    storage.append_row(
        storage.DECISIONS_CSV,
        storage.DECISION_COLUMNS,
        {
            "timestamp": stamp,
            "tac": tac,
            "creatinine": creatinine,
            "tac_high": tac_high,
            "creatinine_high": creatinine_high,
            "summary": summary,
            "action": action,
            "appointment_time": appointment_time,
        },
    )
    return f"Saved the lab decision at {stamp}."


@mcp.tool(structured_output=False)
def save_appointment(
    when_time: Annotated[str, Field(description="The free time the person typed, in their own words.")],
    reason: Annotated[str, Field(description="Why the visit was requested, including which lab value was high.")],
) -> str:
    """Save a requested doctor visit after a person gives a time.

    Pick this tool only when a human has already provided a time they are free.
    Do not pick this tool to decide whether a value is high. Read read_patient_limits first.
    Do not invent a time. This does not call a hospital.
    """
    stamp = storage.now_text()
    storage.append_row(
        storage.APPOINTMENTS_CSV,
        storage.APPOINTMENT_COLUMNS,
        {
            "timestamp": stamp,
            "when_time": when_time,
            "reason": reason,
        },
    )
    return f"Saved the appointment request for {when_time} at {stamp}."


@mcp.tool(structured_output=False)
def read_doctors() -> str:
    """Read the two doctors, their available flag, and their times.

    Pick this tool when tac or creatinine is above the saved limit and a visit may be needed.
    A doctor with available=yes can take a visit. A doctor with available=no cannot.
    Do not use this tool for a medicine reminder or for ordinary chat.
    """
    rows = storage.read_rows(storage.DOCTORS_CSV, storage.DOCTOR_COLUMNS)
    if not rows:
        return "No doctors are saved in data/doctors.csv."
    lines = ["Doctors:"]
    for row in rows:
        lines.append(
            f"- {row.get('name', '')} | {row.get('email', '')} | "
            f"available={row.get('available', '')} | times={row.get('times', '')}"
        )
    lines.append("Book only a doctor whose available value is yes, and only at one of that doctor's times.")
    return "\n".join(lines)


def _slot_for(choice: str, times: str) -> str:
    """Morning uses an AM time. Evening uses a PM time."""
    parts = [part.strip() for part in times.split(";") if part.strip()]
    morning = choice.strip().lower().startswith("morning")
    for part in parts:
        clock = part.upper()
        if morning and "AM" in clock:
            return part
        if not morning and "PM" in clock:
            return part
    return parts[0] if parts else choice.strip()


def send_appointment_emails(when_choice: str, summary: str) -> dict:
    """Email the available doctor and also email the family the same summary.

    The doctor message is practice text addressed to that doctor.
    The family message uses the normal summary address.
    """
    rows = storage.read_rows(storage.DOCTORS_CSV, storage.DOCTOR_COLUMNS)
    chosen = None
    for row in rows:
        if row.get("available", "").strip().lower() == "yes":
            chosen = row
            break
    if chosen is None:
        raise RuntimeError("No doctor is marked available in data/doctors.csv.")

    slot = _slot_for(when_choice, chosen.get("times", ""))
    doctor_name = chosen.get("name", "")
    doctor_email = chosen.get("email", "")
    report = summary.strip() or "No report summary was saved."
    doctor_subject = "Dummy appointment request"
    doctor_body = (
        "DUMMY APPOINTMENT EMAIL\n"
        f"Doctor: {doctor_name}\n"
        f"Selected time: {when_choice} ({slot})\n"
        "Report summary:\n"
        f"{report}\n"
        "This is practice text. No clinic was contacted.\n"
    )
    doctor_path = storage.save_email_copy(doctor_subject, doctor_body, doctor_email)
    family_subject = "Appointment summary"
    family_body = (
        f"Appointment set with {doctor_name}.\n"
        f"Selected time: {when_choice} ({slot})\n"
        "Report summary:\n"
        f"{report}\n"
        "This is a family summary, not a medical instruction.\n"
    )
    family_note = deliver_email(family_subject, family_body)
    stamp = storage.now_text()
    storage.append_row(
        storage.APPOINTMENTS_CSV,
        storage.APPOINTMENT_COLUMNS,
        {
            "timestamp": stamp,
            "when_time": f"{when_choice} ({slot})",
            "reason": f"{doctor_name}: {report.replace(chr(10), ' ')}",
        },
    )
    return {
        "doctor": doctor_name,
        "email": doctor_email,
        "choice": when_choice,
        "slot": slot,
        "doctor_path": doctor_path,
        "family_note": family_note,
    }


@mcp.tool(structured_output=False)
def send_doctor_email(
    doctor_name: Annotated[str, Field(description="Name of the available doctor, such as Dr Patil.")],
    when_time: Annotated[str, Field(description="Visit slot. Morning means Monday 10 AM. Evening means Friday 5 PM.")],
    reason: Annotated[str, Field(description="Report summary to include in the appointment email.")],
) -> str:
    """Save a dummy appointment email to one available doctor and a summary for the family.

    Pick this tool only after the person has chosen Morning or Evening.
    Morning means Monday 10 AM. Evening means Friday 5 PM.
    Pick the doctor whose available cell is yes and whose times match that slot.
    Include the doctor name, the selected time, and the report summary.
    Do not pick a doctor whose available cell is no.
    This saves practice text. It does not contact a real clinic.
    """
    rows = storage.read_rows(storage.DOCTORS_CSV, storage.DOCTOR_COLUMNS)
    chosen = None
    for row in rows:
        if row.get("name", "").strip().lower() == doctor_name.strip().lower():
            chosen = row
            break
    if chosen is None:
        return "No doctor with that name. Call read_doctors and use the exact name."
    if chosen.get("available", "").strip().lower() != "yes":
        return f"{chosen.get('name', '')} is not available. Choose a doctor whose available cell is yes."

    slot = _slot_for(when_time, chosen.get("times", ""))
    info = send_appointment_emails(when_time if when_time.lower().startswith(("morning", "evening")) else slot, reason)
    return (
        f"Dummy appointment email for {info['doctor']} at {info['slot']} was saved at {info['doctor_path']}. "
        f"Family summary: {info['family_note']}"
    )


@mcp.tool(structured_output=False)
def read_history() -> str:
    """Read recent lab decisions and chat lines.

    Pick this tool when the user asks about a past report, a past decision, or earlier messages.
    Do not pick this tool to store a new decision or a new message.
    """
    decisions = storage.read_rows(storage.DECISIONS_CSV, storage.DECISION_COLUMNS)[-15:]
    chats = storage.read_rows(storage.CHAT_CSV, storage.CHAT_COLUMNS)[-15:]
    appointments = storage.read_rows(storage.APPOINTMENTS_CSV, storage.APPOINTMENT_COLUMNS)[-15:]

    lines = ["Past lab decisions:"]
    if not decisions:
        lines.append("(none yet)")
    for row in decisions:
        lines.append(
            f"- {row.get('timestamp', '')} | tac={row.get('tac', '')} "
            f"creatinine={row.get('creatinine', '')} | "
            f"tac_high={row.get('tac_high', '')} creatinine_high={row.get('creatinine_high', '')} | "
            f"action={row.get('action', '')} | time={row.get('appointment_time', '')} | "
            f"{row.get('summary', '')}"
        )

    lines.append("Past appointment requests:")
    if not appointments:
        lines.append("(none yet)")
    for row in appointments:
        lines.append(
            f"- {row.get('timestamp', '')} | {row.get('when_time', '')} | {row.get('reason', '')}"
        )

    lines.append("Past chat:")
    if not chats:
        lines.append("(none yet)")
    for row in chats:
        lines.append(f"- {row.get('timestamp', '')} {row.get('role', '')}: {row.get('message', '')}")
    return "\n".join(lines)


@mcp.tool(structured_output=False)
def save_chat(
    role: Annotated[str, Field(description="Who spoke. Use user or assistant.")],
    message: Annotated[str, Field(description="The exact chat line to keep.")],
) -> str:
    """Save one chat line so the next question can use the history.

    Pick this tool after the user speaks, and again after you answer.
    role must be user or assistant.
    Do not pick this tool to store a lab decision. Use save_decision for that.
    """
    clean_role = role.strip().lower()
    if clean_role not in ("user", "assistant"):
        return "Chat was not saved. role must be user or assistant."
    stamp = storage.now_text()
    storage.append_row(
        storage.CHAT_CSV,
        storage.CHAT_COLUMNS,
        {"timestamp": stamp, "role": clean_role, "message": message},
    )
    return f"Saved the {clean_role} chat line at {stamp}."


@mcp.tool(structured_output=False)
def log_reminder(
    slot: Annotated[str, Field(description="Which dose. Use morning for 9 AM or evening for 9 PM.")],
) -> str:
    """Log the 9 AM or 9 PM medicine reminder and email it.

    Pick this tool when it is time for the morning or evening tablets.
    slot must be morning or evening.
    Do not pick this tool for a lab report, a doctor visit, or a chat question.
    Do not also call send_email for the same reminder. This tool sends that email itself.
    """
    clean_slot = slot.strip().lower()
    if clean_slot not in ("morning", "evening"):
        return "Reminder was not saved. slot must be morning or evening."

    patient = storage.read_patient()
    medicine = patient.get("medicine", "").strip() or "tablets"
    if clean_slot == "morning":
        clock = patient.get("morning_time", "").strip() or "09:00"
        text = f"Morning reminder ({clock}): time for {medicine}."
    else:
        clock = patient.get("evening_time", "").strip() or "21:00"
        text = f"Evening reminder ({clock}): time for {medicine}."

    stamp = storage.now_text()
    storage.append_row(
        storage.REMINDERS_CSV,
        storage.REMINDER_COLUMNS,
        {
            "timestamp": stamp,
            "slot": clean_slot,
            "medicine": medicine,
            "message": text,
        },
    )
    mail_result = deliver_email("Medicine reminder", text)
    return f"Saved the {clean_slot} reminder at {stamp}. {mail_result}"


if __name__ == "__main__":
    # stdio is the pipe the agent uses. Do not print extra text here.
    mcp.run(transport="stdio")
