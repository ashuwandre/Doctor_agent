"""
Simple Doctor Report Assistant
Run:
    streamlit run app.py

This is an interview/demo project:
- Accepts a TEXT lab report only.
- Extracts Tac and Creatinine values.
- Uses one LangChain agent and a simple Python tool.
- Uses LangGraph to run two steps.
- Uses Streamlit for human review before preparing an appointment request.

Important:
The threshold values are demo values only, NOT medical advice.
Use the patient's clinician-provided limits in a real application.
This demo does NOT send real emails or store uploaded reports.
"""

import os
import re
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

load_dotenv()

MODEL_NAME = "gpt-4o-mini"

# Demo values only. Replace with clinician-provided limits.
DEFAULT_TAC_MAX = 8.0
DEFAULT_CREATININE_MAX = 1.5


# ---------------------------------------------------------
# 2. Simple tool
# ---------------------------------------------------------

@tool
def check_lab_values(
    tac: float,
    creatinine: float,
    tac_max: float,
    creatinine_max: float,
) -> str:
    """Compare Tac and Creatinine results with the sample limits provided."""

    messages = []

    if tac > tac_max:
        messages.append(
            f"Tac value {tac} is above the sample maximum {tac_max}."
        )
    else:
        messages.append(
            f"Tac value {tac} is within the sample maximum {tac_max}."
        )

    if creatinine > creatinine_max:
        messages.append(
            f"Creatinine value {creatinine} is above the sample maximum {creatinine_max}."
        )
    else:
        messages.append(
            f"Creatinine value {creatinine} is within the sample maximum {creatinine_max}."
        )

    if tac > tac_max or creatinine > creatinine_max:
        messages.append(
            "Please contact the treating clinical team for interpretation."
        )
    else:
        messages.append(
            "Both values are within these demo limits. This does not confirm that the results are medically normal."
        )

    return "\n".join(messages)


# ---------------------------------------------------------
# 3. LangGraph state
# ---------------------------------------------------------

class ReportState(TypedDict):
    report_text: str
    tac: float
    creatinine: float
    tac_max: float
    creatinine_max: float
    summary: str


# ---------------------------------------------------------
# 4. Node 1: Extract values from the text report
# ---------------------------------------------------------

def extract_report_node(state: ReportState) -> dict:
    """Read simple lines such as Tac: 7.2 and Creatinine: 1.1."""

    report = state["report_text"]

    tac_match = re.search(
        r"(?im)^\s*tac(?:rolimus)?\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        report,
    )
    creatinine_match = re.search(
        r"(?im)^\s*creatinine\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        report,
    )

    if not tac_match or not creatinine_match:
        raise ValueError(
            "The text report must contain lines like 'Tac: 7.2' "
            "and 'Creatinine: 1.1'."
        )

    return {
        "tac": float(tac_match.group(1)),
        "creatinine": float(creatinine_match.group(1)),
    }


# ---------------------------------------------------------
# 5. Create the LangChain agent
# ---------------------------------------------------------

def make_agent():
    """Create one agent and give it the check_lab_values tool."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )

    return create_agent(
        model=llm,
        tools=[check_lab_values],
        system_prompt=(
            "You are a careful report-summary assistant. "
            "You must use the check_lab_values tool to compare the values. "
            "Explain the tool result in simple language. "
            "Do not diagnose, recommend medication changes, or give treatment advice. "
            "Say that demo limits must be confirmed with the treating clinician."
        ),
    )


# ---------------------------------------------------------
# 6. Node 2: Ask the agent to use the tool and explain result
# ---------------------------------------------------------

def analyze_report_node(state: ReportState) -> dict:
    agent = make_agent()

    prompt = f"""
    Check this report using the check_lab_values tool.

    Tac value: {state['tac']}
    Creatinine value: {state['creatinine']}

    Demo Tac maximum: {state['tac_max']}
    Demo Creatinine maximum: {state['creatinine_max']}

    Explain the result in simple language.
    """

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    answer = result["messages"][-1].content

    return {"summary": answer}


# ---------------------------------------------------------
# 7. Build the LangGraph workflow
# ---------------------------------------------------------

def build_graph():
    builder = StateGraph(ReportState)

    builder.add_node("extract_report", extract_report_node)
    builder.add_node("analyze_report", analyze_report_node)

    builder.add_edge(START, "extract_report")
    builder.add_edge("extract_report", "analyze_report")
    builder.add_edge("analyze_report", END)

    return builder.compile()


graph = build_graph()


# ---------------------------------------------------------
# 8. Simple doctor availability tool
# ---------------------------------------------------------

@tool
def get_available_doctor(slot: str) -> str:
    """Find the demo doctor available for Morning or Evening."""

    if slot.lower() == "morning":
        return "Demo: Dr Patil is available Monday at 10 AM."

    if slot.lower() == "evening":
        return "Demo: Dr Patil is available Friday at 5 PM."

    return "No matching demo appointment slot was found."


# ---------------------------------------------------------
# 9. Streamlit user interface
# ---------------------------------------------------------

st.title("Simple Doctor Report Assistant")

st.warning(
    "Learning demo only. The limits below are sample values, not medical "
    "instructions. Confirm all limits and results with the treating clinician."
)

st.write(
    "Upload a text report containing lines such as "
    "'Tac: 7.2' and 'Creatinine: 1.1'."
)

tac_max = st.number_input(
    "Demo Tac maximum",
    min_value=0.1,
    value=DEFAULT_TAC_MAX,
    step=0.1,
)

creatinine_max = st.number_input(
    "Demo Creatinine maximum",
    min_value=0.1,
    value=DEFAULT_CREATININE_MAX,
    step=0.1,
)

uploaded_file = st.file_uploader(
    "Upload lab report (.txt)",
    type=["txt"],
)

if uploaded_file is not None and st.button("Check report"):
    try:
        report_text = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "report_text": report_text,
            "tac": 0.0,
            "creatinine": 0.0,
            "tac_max": float(tac_max),
            "creatinine_max": float(creatinine_max),
            "summary": "",
        }

        with st.spinner("Reading and checking the report..."):
            result = graph.invoke(initial_state)

        st.session_state["report_result"] = result
        st.session_state["appointment_prepared"] = ""

    except Exception as exc:
        st.error(str(exc))


# ---------------------------------------------------------
# 10. Show report result and human review
# ---------------------------------------------------------

report_result = st.session_state.get("report_result")

if report_result:
    st.subheader("Extracted values")
    st.write(f"Tac: {report_result['tac']}")
    st.write(f"Creatinine: {report_result['creatinine']}")

    st.subheader("Agent summary")
    st.write(report_result["summary"])

    needs_clinician_review = (
        report_result["tac"] > report_result["tac_max"]
        or report_result["creatinine"] > report_result["creatinine_max"]
    )

    if needs_clinician_review:
        st.subheader("Human review")

        st.write(
            "At least one value is above the demo limit. "
            "Please contact the treating clinical team. "
            "Choose a demo slot only if the family wants to prepare an appointment request."
        )

        slot = st.selectbox(
            "Choose a preferred demo appointment time",
            ["Morning", "Evening"],
        )

        if st.button("Prepare appointment request"):
            doctor_info = get_available_doctor.invoke({"slot": slot})

            st.session_state["appointment_prepared"] = (
                f"{doctor_info}\n\n"
                "Appointment request prepared for human review. "
                "No email has been sent."
            )

        if st.session_state.get("appointment_prepared"):
            st.success(st.session_state["appointment_prepared"])

    else:
        st.info(
            "Both values are within the entered demo limits. "
            "This is not a medical assessment."
        )

st.caption(
    "Demo limitations: text reports only, sample thresholds, one agent, "
    "two LangGraph nodes, and no real appointment booking or email sending."
)
