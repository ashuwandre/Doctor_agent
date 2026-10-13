# One LangGraph agent. There is no router.
# Steps: read the photo, match the goal to a tool description, call that tool, pause for a time.
# Run: python graph.py report.jpg
# Chat: python graph.py chat "What was the last tac level?"

import asyncio  # run the async graph from a normal function
import base64  # turn image bytes into text for the model
import os  # check the file type and whether the file exists
from datetime import datetime  # make a unique id for each run
from typing import TypedDict  # the fields the graph remembers

from langchain_core.messages import HumanMessage  # send the picture to Azure
from langgraph.checkpoint.memory import MemorySaver  # remember a paused run in this process
from langgraph.graph import END, START, StateGraph  # the graph pieces
from langgraph.types import Command, interrupt  # resume a pause, and pause for a person

import storage  # patient limits used for the on-screen summary
from mcp_client import call_named_tool, chat_model, choose_tool, list_tools, read_json


# Stop a run that keeps calling tools.
MAX_STEPS = 10

# Extra choice for a report only. This is not an MCP tool.
ASK_FOR_TIME = """
One extra choice is allowed, and it is not an MCP tool:
{"tool": "ask_human", "arguments": {}}
Do not pick ask_human yourself. After read_doctors, the Doctors section asks Morning or Evening.
Do not call send_doctor_email until a result line starts with "Human is free at:".
"""


class AgentState(TypedDict):
    # The job the model compares with each tool description.
    goal: str
    # Uploaded file. Empty during chat.
    report_path: str
    # Every tool name and description, as text.
    cards: str
    # Allowed tool names from the server.
    tool_names: list[str]
    # Tool results so far.
    notes: list[str]
    # How many choose steps have run.
    steps: int
    # The tool the model just picked.
    tool_name: str
    # Arguments for that tool.
    tool_args: dict
    # The final text shown on the screen.
    answer: str
    # Short lab summary shown as soon as Check report finishes.
    summary: str
    # Yes or no: whether an appointment is needed.
    appointment: str


def _blank(goal: str, report_path: str) -> AgentState:
    # Start one run with empty notes.
    return {
        "goal": goal,
        "report_path": report_path,
        "cards": "",
        "tool_names": [],
        "notes": [],
        "steps": 0,
        "tool_name": "",
        "tool_args": {},
        "answer": "",
        "summary": "",
        "appointment": "",
    }


def _pictures(path: str) -> list[tuple[str, str]]:
    # Return pairs of (mime type, base64 text) for a photo or a PDF.
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")
        return [(mime, encoded)]

    if ext == ".pdf":
        # Open the PDF and turn the first three pages into images.
        import pymupdf

        document = pymupdf.open(path)
        pictures = []
        for index, page in enumerate(document):
            if index >= 3:
                break
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            pictures.append(("image/png", encoded))
        document.close()
        if not pictures:
            raise ValueError("The PDF has no pages.")
        return pictures

    raise ValueError("Upload a txt, jpg, jpeg, png, or pdf file.")


def _read_text_report(path: str) -> dict:
    # Read date, tac, creatinine, and note from lines like "Tac: 9.2".
    found = {"date": "", "tac": "", "creatinine": "", "note": ""}
    with open(path, encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "date":
                found["date"] = value
            elif key in ("tac", "tacrolimus"):
                found["tac"] = value
            elif key in ("creatinine", "creatine"):
                found["creatinine"] = value
            elif key == "note":
                found["note"] = value
    return found


def _as_number(text: str) -> float | None:
    # Turn a lab cell into a number. Empty or words stay empty.
    try:
        return float(str(text).strip())
    except ValueError:
        return None


def _lab_summary(found: dict) -> tuple[str, str]:
    # Compare the report with the sample limits and say if a visit is needed.
    patient = storage.read_patient()
    tac = _as_number(found.get("tac", ""))
    creatinine = _as_number(found.get("creatinine", ""))
    tac_max = _as_number(patient.get("tac_max", ""))
    creatinine_max = _as_number(patient.get("creatinine_max", ""))
    tac_high = tac is not None and tac_max is not None and tac > tac_max
    creatinine_high = (
        creatinine is not None and creatinine_max is not None and creatinine > creatinine_max
    )
    lines = [
        f"Date: {found.get('date', '') or 'not shown'}",
        f"Tac: {found.get('tac', '')} (sample limit {patient.get('tac_min', '')} to {patient.get('tac_max', '')})",
        f"Creatinine: {found.get('creatinine', '')} (sample max {patient.get('creatinine_max', '')})",
    ]
    if found.get("note"):
        lines.append(f"Note: {found.get('note', '')}")
    if tac_high:
        lines.append("Tac is above the sample limit.")
    if creatinine_high:
        lines.append("Creatinine is above the sample limit.")
    if tac is not None and creatinine is not None and not tac_high and not creatinine_high:
        lines.append("Both values are inside the sample limits.")
    summary = "\n".join(lines)
    if tac_high or creatinine_high:
        appointment = "Yes. An appointment is needed. Choose Morning or Evening in the Doctors section."
    else:
        appointment = "No. An appointment is not needed."
    return summary, appointment


def _report_goal(found: dict) -> str:
    # Build the goal from the numbers read off the report.
    return f"""A lab report was read.
Date: {found.get("date", "")}
Tac: {found.get("tac", "")}
Creatinine: {found.get("creatinine", "")}
Note: {found.get("note", "")}

Finish this goal by matching tool descriptions. Do these parts if a description fits:
- Read the limits her doctor saved.
- Look up general Wikipedia background for tacrolimus and creatinine.
- Email a short summary. Say this is a family summary, not a medical instruction.
  The high or low check uses only the numbers in the patient file. Blank limits mean the value is not high.
- Save the lab decision once.
- Call send_email only once for the family summary.
- If tac is above tac_max or creatinine is above creatinine_max:
  1. Call read_doctors.
  2. Stop after read_doctors. The Doctors section asks Morning or Evening before any appointment email.
  3. After a line that starts with "Human is free at:" is in the results, call send_doctor_email once.
     Morning means Monday 10 AM. Evening means Friday 5 PM.
     Choose the doctor whose available cell is yes and whose times include that slot.
     Do not email a doctor whose available cell is no.
     Do not call send_doctor_email before that human line exists.
  4. Then choose done.
- If both values are inside the limits, do not ask for a time and do not email a doctor.
- When those parts are done, choose done.
"""


def read_report(state: AgentState) -> dict:
    # Chat has no file, so skip the picture step.
    path = (state.get("report_path") or "").strip()
    if not path:
        return {"tool_name": ""}
    if not os.path.exists(path):
        return {"tool_name": "done", "notes": [f"Report file was not found: {path}"]}

    # A text file already has the numbers, so no picture call is needed.
    if path.lower().endswith(".txt"):
        found = _read_text_report(path)
        tac = found["tac"]
        creatinine = found["creatinine"]
        if not tac or not creatinine:
            return {
                "tool_name": "done",
                "notes": ["The text file needs a Tac line and a Creatinine line."],
            }
        summary, appointment = _lab_summary(found)
        return {
            "tool_name": "",
            "goal": _report_goal(found),
            "summary": summary,
            "appointment": appointment,
            "notes": [summary, appointment],
        }

    try:
        pictures = _pictures(path)
    except Exception as exc:
        return {"tool_name": "done", "notes": [f"Could not read the report file: {exc}"]}

    # Ask the model to copy tac and creatinine out of the picture.
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Read this lab report image. Reply with JSON only: "
                '{"date": "", "tac": "", "creatinine": "", "note": ""}. '
                "tac and creatinine must be the numbers you can see, with no units. "
                "Leave a value empty if you cannot see it. "
                "Creatine on the page means creatinine."
            ),
        }
    ]
    for mime, encoded in pictures:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }
        )

    reply = chat_model().invoke([HumanMessage(content=content)])
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    try:
        found = read_json(text)
    except Exception as exc:
        return {"tool_name": "done", "notes": [f"Could not read numbers from the report: {exc}"]}

    # Stop when either number is missing. Do not email or book a visit.
    tac = str(found.get("tac", "")).strip()
    creatinine = str(found.get("creatinine", "")).strip()
    if not tac or not creatinine:
        return {
            "tool_name": "done",
            "notes": [
                "Tac or creatinine was not visible on the report. "
                "No email was sent and no appointment was booked."
            ],
        }

    # Continue. The next step compares this goal with the tool descriptions.
    summary, appointment = _lab_summary(found)
    return {
        "tool_name": "",
        "goal": _report_goal(found),
        "summary": summary,
        "appointment": appointment,
        "notes": [summary, appointment],
    }


async def load_tools(state: AgentState) -> dict:
    # Read every tool description from the MCP server.
    cards, names = await list_tools()
    return {"cards": cards, "tool_names": names}


def choose(state: AgentState) -> dict:
    # Count this step and stop after MAX_STEPS.
    steps = int(state.get("steps") or 0) + 1
    notes = list(state.get("notes") or [])
    if steps > MAX_STEPS:
        notes.append("Agent: stopped after the step limit.")
        return {"steps": steps, "tool_name": "done", "tool_args": {}, "notes": notes}

    # A report may ask the person for a time. Chat does not.
    extra = ASK_FOR_TIME if state.get("report_path") else ""
    choice = choose_tool(state.get("goal") or "", state.get("cards") or "", notes, extra)
    name = str(choice.get("tool", "")).strip()
    arguments = choice.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    # Reject a tool name that is not on the server.
    allowed = set(state.get("tool_names") or [])
    allowed.update({"done", "ask_human"})
    if name not in allowed:
        notes.append(f"Agent picked unknown tool '{name}'.")
        return {"steps": steps, "tool_name": "done", "tool_args": {}, "notes": notes}

    # The appointment email waits until the Doctors dropdown has a choice.
    has_time = any(line.startswith("Human is free at:") for line in notes)
    if name == "send_doctor_email" and state.get("report_path") and not has_time:
        notes.append("Agent: the Doctors dropdown must choose Morning or Evening before the appointment email.")
        return {"steps": steps, "tool_name": "ask_human", "tool_args": {}, "notes": notes}

    return {"steps": steps, "tool_name": name, "tool_args": arguments, "notes": notes}


async def call_tool(state: AgentState) -> dict:
    # Call the MCP tool the model just picked.
    name = state.get("tool_name") or ""
    arguments = state.get("tool_args") or {}
    text = await call_named_tool(name, arguments)
    notes = list(state.get("notes") or [])
    notes.append(f"{name}: {text}")
    return {"notes": notes}


def ask_human(state: AgentState) -> dict:
    # Pause here. On resume, interrupt returns the time the person typed.
    time_text = interrupt(
        "Dr Patil is available in the morning (Monday 10 AM) and the evening (Friday 5 PM). "
        "Dr Shah is not available. Choose Morning or Evening."
    )
    notes = list(state.get("notes") or [])
    notes.append(f"Human is free at: {time_text}")
    return {"notes": notes}


def finish(state: AgentState) -> dict:
    # Join the notes into the answer shown on the screen.
    notes = [line for line in (state.get("notes") or []) if line]
    text = "\n".join(notes).strip() or "No result."
    return {"answer": text}


def _after_read(state: AgentState) -> str:
    # A missing number ends the run. Otherwise choose a tool.
    if state.get("tool_name") == "done":
        return "finish"
    return "choose"


def _after_tool(state: AgentState) -> str:
    # After the doctor list, pause in the Doctors section before any appointment email.
    notes = state.get("notes") or []
    has_time = any(str(line).startswith("Human is free at:") for line in notes)
    if state.get("report_path") and state.get("tool_name") == "read_doctors" and not has_time:
        return "ask_human"
    return "choose"


def _after_choose(state: AgentState) -> str:
    # Follow the one model's choice. This function is not a second agent.
    name = state.get("tool_name") or ""
    if name == "done":
        return "finish"
    if name == "ask_human":
        return "ask_human"
    return "call_tool"


def build_graph():
    # Name each step and connect them.
    graph = StateGraph(AgentState)
    graph.add_node("load_tools", load_tools)
    graph.add_node("read_report", read_report)
    graph.add_node("choose", choose)
    graph.add_node("call_tool", call_tool)
    graph.add_node("ask_human", ask_human)
    graph.add_node("finish", finish)

    graph.add_edge(START, "load_tools")
    graph.add_edge("load_tools", "read_report")
    graph.add_conditional_edges("read_report", _after_read, {"choose": "choose", "finish": "finish"})
    graph.add_conditional_edges(
        "choose",
        _after_choose,
        {"call_tool": "call_tool", "ask_human": "ask_human", "finish": "finish"},
    )
    # After the doctor list, pause for Morning or Evening. Otherwise choose the next tool.
    graph.add_conditional_edges(
        "call_tool",
        _after_tool,
        {"ask_human": "ask_human", "choose": "choose"},
    )
    graph.add_edge("ask_human", "choose")
    graph.add_edge("finish", END)
    # MemorySaver keeps the pause in this process.
    return graph.compile(checkpointer=MemorySaver())


# One graph for the whole process, so a resume can find the paused run.
GRAPH = build_graph()


def _config(thread_id: str) -> dict:
    # LangGraph uses this id to find the saved run.
    return {"configurable": {"thread_id": thread_id}}


def _pack(thread_id: str) -> dict:
    # Read the saved state after the graph stops or pauses.
    snapshot = GRAPH.get_state(_config(thread_id))
    values = snapshot.values or {}
    text = values.get("answer") or "\n".join(values.get("notes") or [])
    packed = {
        "thread_id": thread_id,
        "question": "",
        "text": text,
        "summary": values.get("summary") or "",
        "appointment": values.get("appointment") or "",
    }
    if snapshot.interrupts:
        packed["status"] = "need_time"
        packed["question"] = str(snapshot.interrupts[0].value)
        return packed
    packed["status"] = "done"
    return packed


def run_report(path: str, thread_id: str = "") -> dict:
    # Make an id when the caller did not pass one.
    if not thread_id:
        thread_id = datetime.now().strftime("report-%Y%m%d%H%M%S")
    # Start from an empty goal. read_report fills the goal from the picture.
    asyncio.run(GRAPH.ainvoke(_blank("", path), _config(thread_id)))
    return _pack(thread_id)


def continue_with_time(thread_id: str, time_text: str) -> dict:
    # Continue the paused run and give it the typed time.
    asyncio.run(GRAPH.ainvoke(Command(resume=time_text), _config(thread_id)))
    return _pack(thread_id)


def run_chat(message: str, thread_id: str = "") -> dict:
    # Chat has a goal and no picture.
    if not thread_id:
        thread_id = datetime.now().strftime("chat-%Y%m%d%H%M%S")
    goal = f"""The user said: {message}

Match tool descriptions to this message.
If they ask about a past report, past decision, or earlier chat, read history.
Save their line with save_chat using role user.
Save your short answer with save_chat using role assistant.
Then choose done.
Do not email a lab summary unless they asked you to send one.
"""
    asyncio.run(GRAPH.ainvoke(_blank(goal, ""), _config(thread_id)))
    return _pack(thread_id)


# Command line: a report path, or the word chat plus a question.
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "chat":
        result = run_chat(" ".join(sys.argv[2:]))
    elif len(sys.argv) == 2:
        result = run_report(sys.argv[1])
    else:
        raise SystemExit('Use: python graph.py report.jpg   or   python graph.py chat "question"')

    print(result["status"])
    if result["question"]:
        print(result["question"])
        print("thread_id:", result["thread_id"])
    print(result["text"])
