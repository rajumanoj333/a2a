"""Three-agent Google ADK orchestration demo built with Marimo.

Run with: marimo run app.py
Edit with: marimo edit app.py
"""

import asyncio
import os
from datetime import datetime

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        <style>
        :root { --marimo-heading-font: Inter, ui-sans-serif, system-ui, sans-serif; }
        .hero { padding: 2rem 0 1.25rem; }
        .hero h1 { font-size: 2.4rem; letter-spacing: -0.045em; margin-bottom: .5rem; }
        .hero p { max-width: 46rem; color: #52606d; font-size: 1.05rem; }
        .flow { background: #f4f7fb; border: 1px solid #d8e1ec; border-radius: 14px; padding: 1rem 1.25rem; }
        .flow strong { color: #123b61; }
        .note { color: #52606d; font-size: .92rem; }
        </style>
        <div class="hero">
          <h1>Three minds. One answer.</h1>
          <p>A tiny, observable Google ADK workflow: one agent plans, one solves, and one reviews.</p>
        </div>
        <div class="flow"><strong>Prompt</strong> &nbsp;→&nbsp; Coordinator &nbsp;→&nbsp; Specialist &nbsp;→&nbsp; Reviewer &nbsp;→&nbsp; <strong>Answer</strong></div>
        """
    )

    prompt = mo.ui.text_area(
        value="Explain why orchestration is useful in a multi-agent system.",
        label="Give the team a task",
        rows=5,
        full_width=True,
    )
    run = mo.ui.run_button(label="Run the three-agent team")
    mo.vstack([prompt, run])
    return mo, prompt, run


@app.cell
async def _(mo, prompt, run):
    from google.adk.agents import Agent, SequentialAgent
    from google.adk.runners import Runner
    from google.genai import types
    from google.adk.sessions import InMemorySessionService

    MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-2.5-flash")

    coordinator = Agent(
        name="coordinator",
        model=MODEL,
        instruction=(
            "You are the Coordinator. Read the user's task, clarify the goal internally, "
            "and create a concise execution brief for the Specialist. Do not answer the "
            "user yet. Keep the brief practical and include success criteria."
        ),
    )
    specialist = Agent(
        name="specialist",
        model=MODEL,
        instruction=(
            "You are the Specialist. Use the Coordinator's brief from the shared workflow "
            "context. Do the main work and produce a useful draft answer with examples. "
            "Do not discuss the orchestration mechanics."
        ),
    )
    reviewer = Agent(
        name="reviewer",
        model=MODEL,
        instruction=(
            "You are the Reviewer. Inspect the Specialist's draft and return the final answer. "
            "Fix unclear wording, missing steps, or unsupported claims. Start directly with "
            "the answer and keep it easy to understand."
        ),
    )
    team = SequentialAgent(
        name="three_agent_team",
        sub_agents=[coordinator, specialist, reviewer],
    )

    async def run_adk_workflow(user_prompt: str):
        """Run ADK and return the final text plus a trace of agent events."""
        service = InMemorySessionService()
        app_name = "marimo_adk_demo"
        user_id = "demo_user"
        session = await service.create_session(app_name=app_name, user_id=user_id)
        runner = Runner(agent=team, app_name=app_name, session_service=service)
        content = types.Content(role="user", parts=[types.Part(text=user_prompt)])
        trace = []
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            author = getattr(event, "author", "workflow")
            if author not in {item["agent"] for item in trace}:
                trace.append(
                    {
                        "agent": author,
                        "status": "completed",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                )
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text or final_text
        return final_text, trace

    def demo_workflow(user_prompt: str):
        """Credential-free preview for explaining the UI before connecting Gemini."""
        return (
            "Orchestration divides a larger task into clear responsibilities. The Coordinator "
            "creates the plan, the Specialist does the focused work, and the Reviewer checks "
            "the result. This makes the workflow easier to debug, explain, and extend.",
            [
                {"agent": "coordinator", "status": "completed", "time": "demo"},
                {"agent": "specialist", "status": "completed", "time": "demo"},
                {"agent": "reviewer", "status": "completed", "time": "demo"},
            ],
        )

    if not run.value:
        mo.stop(True)

    use_demo = not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
    if use_demo:
        answer, trace = demo_workflow(prompt.value)
        mode_label = "Preview mode — add GOOGLE_API_KEY to call Gemini"
    else:
        try:
            answer, trace = await run_adk_workflow(prompt.value)
            mode_label = f"Live Google ADK • {MODEL}"
        except Exception as exc:
            answer, trace = demo_workflow(prompt.value)
            mode_label = f"Preview fallback • {type(exc).__name__}"

    return answer, mode_label, mo, trace


@app.cell
def _(answer, mode_label, mo, trace):
    rows = []
    labels = {
        "coordinator": ("Coordinator", "Turns the prompt into a plan"),
        "specialist": ("Specialist", "Completes the main task"),
        "reviewer": ("Reviewer", "Checks and finalizes the answer"),
        "three_agent_team": ("Team", "Coordinates the workflow"),
    }
    for item in trace:
        name, purpose = labels.get(item["agent"], (item["agent"].title(), "Workflow event"))
        rows.append(f"**{name}**  ·  `{item['status']}`  ·  {purpose}  ·  {item['time']}")

    tracking = mo.md("\n\n".join(rows) if rows else "No agent events yet.")
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("### Final answer"),
                    mo.md(answer),
                    mo.md(f"<span class='note'>{mode_label}</span>"),
                ],
                gap=1,
            ),
            mo.vstack([mo.md("### Agent tracking"), tracking], gap=1),
        ],
        widths=[2, 1],
        gap=2,
    )


if __name__ == "__main__":
    app.run()
