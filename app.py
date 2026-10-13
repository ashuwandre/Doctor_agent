# Simple screen. Run: streamlit run app.py

import os  # save the uploaded file

import streamlit as st  # the web page


def error_text(exc: BaseException) -> str:
    # TaskGroup hides the real error. Show the inner message instead.
    parts = []

    def walk(err: BaseException) -> None:
        children = getattr(err, "exceptions", None)
        if children:
            for child in children:
                walk(child)
        elif str(err) and str(err) not in parts:
            parts.append(str(err))

    walk(exc)
    return "\n".join(parts) or str(exc)

import storage  # CSV paths and the history table
from graph import run_chat, run_report  # the one agent
from mcp_server import send_appointment_emails  # doctor email plus the family summary


# Page title.
st.title("Doctor agent")
# Tell the family these limits are sample numbers, not a medical instruction.
st.write(
    "Family summary only, not a medical instruction. "
    "The limits in data/patient.csv are sample study numbers: tac 5 to 8, creatinine max 1.5. "
    "Replace them with the limits from her transplant doctor. "
    "Fill the Azure lines in .env before checking a report."
)

# The user uploads a text file, a photo, or a PDF.
uploaded = st.file_uploader("Lab text, photo, or PDF", type=["txt", "png", "jpg", "jpeg", "pdf"])
# Start the agent only after a file is chosen and the button is clicked.
if uploaded is not None and st.button("Check report"):
    # Make the reports folder if it is missing.
    os.makedirs(storage.REPORT_DIR, exist_ok=True)
    # Save the upload under data/reports.
    path = os.path.join(storage.REPORT_DIR, uploaded.name)
    with open(path, "wb") as file:
        file.write(uploaded.getvalue())
    try:
        # A new report starts a new appointment choice.
        st.session_state["appointment_set"] = None
        # Run the graph. It may pause and ask for a free time.
        st.session_state["run"] = run_report(path)
    except Exception as exc:
        # Show a missing Azure key or a bad file on the page.
        st.error(error_text(exc))

# Read the last graph result saved in this browser session.
run = st.session_state.get("run")
if run and (run.get("summary") or run.get("appointment")):
    # Show the lab summary and whether a visit is needed as soon as Check report returns.
    st.subheader("Report summary")
    st.text(run.get("summary") or "")
    st.subheader("Appointment needed")
    st.write(run.get("appointment") or "")

# Chat uses the same agent with no photo.
st.subheader("Chat")
question = st.text_input("Ask about a past report")
if st.button("Ask") and question.strip():
    try:
        st.session_state["chat"] = run_chat(question.strip())
    except Exception as exc:
        st.error(error_text(exc))
# Show the last chat answer.
if st.session_state.get("chat"):
    st.text(st.session_state["chat"].get("text") or "")

# Show the two doctors and which one is free.
st.subheader("Doctors")
st.table(storage.read_rows(storage.DOCTORS_CSV, storage.DOCTOR_COLUMNS))
# The appointment email is sent only after this dropdown is used.
booked = st.session_state.get("appointment_set") or {}
needs_visit = bool(run) and str(run.get("appointment") or "").startswith("Yes")
if needs_visit or booked:
    options = ["Morning", "Evening"]
    saved_choice = booked.get("choice") or "Morning"
    st.write((run or {}).get("question") or "Choose Morning or Evening before the appointment email.")
    free_time = st.selectbox(
        "Your free time",
        options,
        index=options.index(saved_choice) if saved_choice in options else 0,
        disabled=bool(booked),
    )
    st.write("Morning is Monday 10 AM. Evening is Friday 5 PM. The email goes to the available doctor.")
    if booked:
        # Keep the button on the page and turn it green after the appointment is set.
        st.markdown(
            """
            <style>
            div[class*="st-key-appointment_set_button"] button {
                background-color: #1b7f3a !important;
                color: white !important;
                border: 1px solid #14612c !important;
                opacity: 1 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.button("Appointment already set", disabled=True, key="appointment_set_button")
        st.write(
            f"{booked.get('doctor', '')} at {booked.get('slot', '')}. "
            "That email includes the doctor name, the time from the dropdown, and the report summary. "
            "The same summary was also saved for you."
        )
    elif st.button("Send appointment email"):
        try:
            info = send_appointment_emails(free_time, (run or {}).get("summary") or "")
        except Exception as exc:
            st.error(error_text(exc))
        else:
            updated = dict(run)
            updated["appointment"] = f"Yes. Appointment set with {info['doctor']} at {info['slot']}."
            st.session_state["run"] = updated
            st.session_state["appointment_set"] = info
            st.rerun()
elif run and str(run.get("appointment") or "").startswith("No"):
    st.write("No appointment email. Both values are inside the sample limits.")
else:
    st.write("When a lab value is above the limit, Morning or Evening appears here before any appointment email.")

# Show every saved lab decision.
st.subheader("Saved decisions")
st.table(storage.read_rows(storage.DECISIONS_CSV, storage.DECISION_COLUMNS))
