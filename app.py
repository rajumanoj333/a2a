"""Streamlit UI for a Google ADK multi-agent travel planner."""

import asyncio
import os
import warnings
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"\[EXPERIMENTAL\] feature FeatureName\.JSON_SCHEMA_FOR_FUNC_DECL.*",
    category=UserWarning,
)

import streamlit as st
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from travel_agents import build_agents


load_dotenv()
st.set_page_config(page_title="AI Trip Planner", page_icon="🧭", layout="wide")


def load_streamlit_secrets() -> None:
    """Make Streamlit Cloud secrets available to the direct Groq adapter."""
    for name in ("GROQ_API_KEY", "GROQ_MODEL"):
        if os.getenv(name):
            continue
        try:
            value = st.secrets.get(name)
        except FileNotFoundError:
            value = None
        if value:
            os.environ[name] = str(value)


load_streamlit_secrets()


def event_detail(event: Any) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    author = getattr(event, "author", "workflow")
    for call in event.get_function_calls():
        if call.name == "transfer_to_agent":
            details.append(
                {
                    "kind": "delegation",
                    "agent": author,
                    "detail": f"Agent delegation to {call.args or {}}",
                }
            )
            continue
        details.append(
            {
                "kind": "tool",
                "agent": author,
                "detail": f"{call.name}({call.args or {}})",
            }
        )
    for response in event.get_function_responses():
        if response.name == "transfer_to_agent":
            details.append(
                {
                    "kind": "delegation",
                    "agent": author,
                    "detail": "Agent delegation completed",
                }
            )
            continue
        details.append(
            {
                "kind": "tool_result",
                "agent": author,
                "detail": f"{response.name} returned {response.response}",
            }
        )
    metadata = getattr(event, "custom_metadata", {}) or {}
    if any(key.startswith("adk:") for key in metadata):
        details.append(
            {
                "kind": "delegation",
                "agent": author,
                "detail": "ADK agent event",
            }
        )
    if author != "user" and event.content and event.content.parts:
        text = " ".join(
            part.text.strip()
            for part in event.content.parts
            if part.text and part.text.strip()
        )
        if text:
            details.append(
                {
                    "kind": "agent",
                    "agent": author,
                    "detail": text,
                }
            )
    if not details and author != "user":
        details.append({"kind": "agent", "agent": author, "detail": "Agent event"})
    return details


async def run_workflow(prompt: str) -> tuple[str, list[dict[str, str]]]:
    service = InMemorySessionService()
    app_name = "adk_travel_planner"
    user_id = "streamlit_user"
    session = await service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(
        agent=build_agents(),
        app_name=app_name,
        session_service=service,
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    trace: list[dict[str, str]] = [
        {"kind": "orchestrator", "agent": "orchestrator", "detail": "Received user prompt"}
    ]
    final_text = ""
    last_agent_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        trace.extend(event_detail(event))
        if event.is_final_response() and event.content and event.content.parts:
            text = " ".join(
                part.text.strip()
                for part in event.content.parts
                if part.text and part.text.strip()
            )
            if text:
                if event.author == "orchestrator":
                    final_text = text
                last_agent_text = text
    if not final_text:
        final_text = last_agent_text
    return final_text, trace


def main() -> None:
    st.title("🧭 AI Trip Planner")
    st.caption("Google ADK + GroqCloud + Streamlit")
    st.info(
        "The orchestrator delegates to local Travel, Hotel, and Food ADK agents. "
        "Each specialist uses focused deterministic demo tools."
    )
    prompt = st.text_area(
        "What would you like to plan?",
        value="Find a hotel in Goa and restaurants within 2 km of the hotel.",
        height=120,
    )
    if st.button("Plan my trip", type="primary", disabled=not prompt.strip()):
        if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key":
            st.error("Set a valid GROQ_API_KEY in .env before planning a trip.")
            st.stop()
        with st.spinner("Orchestrator is coordinating agents..."):
            try:
                answer, trace = asyncio.run(run_workflow(prompt))
                if not answer.strip():
                    raise RuntimeError(
                        "The ADK workflow completed without a final response. "
                        "Check the agent trace for the last completed handoff."
                    )
            except Exception as exc:
                st.error(f"The ADK workflow failed: {type(exc).__name__}: {exc}")
                st.stop()
        st.subheader("Your plan")
        st.markdown(answer)
        with st.expander("Live agent and tool tracking", expanded=True):
            for item in trace:
                label = item["kind"].replace("_", " ").title()
                if item["kind"] == "delegation":
                    st.markdown(f"**Agent delegation** · `{item['agent']}` · {item['detail']}")
                else:
                    st.markdown(f"- **{label}** · `{item['agent']}` · {item['detail']}")


if __name__ == "__main__":
    main()
