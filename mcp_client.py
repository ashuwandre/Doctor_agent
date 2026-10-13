# MCP client. It starts mcp_server.py, reads tool descriptions, and calls one tool.
# The graph decides which tool. This file does not decide.
# Run: python mcp_client.py

import asyncio  # run the async session from the command line
import json  # read the JSON the model returns
import os  # find this folder and the .env file
import sys  # the Python program used to start the server
from contextlib import asynccontextmanager  # open and close the server session

from dotenv import load_dotenv  # read .env into the process
from mcp import ClientSession  # talk to the MCP server
from mcp.client.stdio import StdioServerParameters, stdio_client  # start the server on a pipe


# Folder that contains this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load Azure and SMTP values from .env.
load_dotenv(os.path.join(BASE_DIR, ".env"))

# How to start mcp_server.py. Its stdout is the MCP pipe, so the server must not print.
SERVER = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(BASE_DIR, "mcp_server.py")],
    cwd=BASE_DIR,
)


@asynccontextmanager
async def open_session():
    # Start the server and keep the pipe open for this block.
    async with stdio_client(SERVER) as (read, write):
        # Wrap the pipe in an MCP session.
        async with ClientSession(read, write) as session:
            # Handshake before any tool call.
            await session.initialize()
            # Hand the ready session to the caller.
            yield session


def chat_model():
    # Accept either name. The .env from the other project uses DEPLOYMENT_NAME.
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        or os.getenv("AZURE_OPENAI_MODEL_NAME", "").strip()
    )
    if not api_key or not deployment:
        raise RuntimeError("Fill AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT_NAME in .env.")

    # Import here so listing tools still works before this package is needed.
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    # The copied endpoint is still the placeholder, and the key is an OpenAI key.
    # Use Azure only when the endpoint is a real Azure address.
    azure_ready = "openai.azure.com" in endpoint and "YOUR-RESOURCE" not in endpoint
    if azure_ready:
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version or "2024-08-01-preview",
            azure_deployment=deployment,
            temperature=0,
        )
    return ChatOpenAI(model=deployment, api_key=api_key, temperature=0)


def tool_card(tool) -> str:
    # Turn one MCP tool into text the model can compare with the goal.
    # Older mcp uses inputSchema. Newer mcp uses input_schema.
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [f"Tool: {tool.name}", "Description:", (tool.description or "").strip(), "Arguments:"]
    if not properties:
        lines.append("- no arguments")
    # Add each argument name and the description the server published.
    for name, info in properties.items():
        need = "required" if name in required else "optional"
        lines.append(f"- {name} ({need}): {info.get('description', '')}")
    return "\n".join(lines)


def result_text(result) -> str:
    # Collect the text blocks from a tool result.
    lines = [getattr(block, "text", "") for block in result.content]
    body = "\n".join(line for line in lines if line).strip() or "(no text returned)"
    # Older mcp uses isError. Newer mcp uses is_error.
    failed = getattr(result, "is_error", None)
    if failed is None:
        failed = getattr(result, "isError", False)
    if failed:
        return "Tool error: " + body
    return body


def read_json(text: str) -> dict:
    # The model sometimes wraps JSON in a markdown fence. Remove that fence.
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    # Keep only the object, in case there is text around it.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("The model did not return a JSON object.")
    return data


def choose_tool(goal: str, cards: str, notes: list[str], extra: str = "") -> dict:
    # Show earlier tool results so the model does not call the same tool forever.
    so_far = "\n".join(notes) if notes else "(nothing yet)"
    # extra is the ask_human choice. It is empty during chat.
    extra_block = ("\n" + extra.strip() + "\n") if extra.strip() else ""
    # The model must match the goal to a tool description. There is no router.
    prompt = f"""You are the one doctor agent. There is no router.

Compare the goal with each tool description.
Call the one tool whose description matches the part of the goal that is still open.
Skip a tool when its description says not to use it for this job.
If the goal is finished, set tool to done.
{extra_block}
Reply with JSON only:
{{"tool": "tool_name_or_done", "arguments": {{}}}}

Goal:
{goal}

Tool results so far:
{so_far}

Tools:
{cards}
"""
    reply = chat_model().invoke(prompt)
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    return read_json(text)


async def list_tools() -> tuple[str, list[str]]:
    # Open the server, read the tool list, then close it.
    async with open_session() as session:
        listed = await session.list_tools()
        cards = [tool_card(tool) for tool in listed.tools]
        names = [tool.name for tool in listed.tools]
        return "\n\n".join(cards), names


async def call_named_tool(name: str, arguments: dict | None = None) -> str:
    # Call the tool the graph already chose.
    async with open_session() as session:
        result = await session.call_tool(name, arguments or {})
        return result_text(result)


# Print the descriptions when this file is run by itself.
if __name__ == "__main__":
    cards, _names = asyncio.run(list_tools())
    print(cards)
