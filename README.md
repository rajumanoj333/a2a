# AI Trip Planner

A real Google ADK multi-agent demo using the official GroqCloud SDK and Streamlit.

## Architecture

```text
Streamlit
   |
ADK Orchestrator
   | local ADK agent delegation
   +--> Travel Agent --> search_places()
   +--> Hotel Agent  --> search_hotels(), search_restaurants()
   +--> Food Agent   --> search_restaurants()
```

The specialist agents are local ADK sub-agents. The orchestrator delegates to them directly in one process, so there are no A2A servers, network ports, agent cards, or experimental A2A integration warnings. The tracking panel shows agent delegation, tool calls, tool results, and agent responses.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

Set your Groq key in `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

`groq_adk_model.py` is the small compatibility adapter between ADK's model
interface and the official Groq SDK. No LiteLLM, A2A, or OpenAI credentials are used.

Then start Streamlit:

```powershell
streamlit run app.py
```

The current tools return deterministic demo data. Replace those tool bodies with real travel APIs later without changing the agent boundaries.
