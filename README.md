# Three minds. One answer.

A small, easy-to-explain multi-agent demo using [Google ADK](https://google.github.io/adk-docs/) and [Marimo](https://github.com/marimo-team/marimo).

## What it demonstrates

1. **Coordinator** turns the user's prompt into a brief.
2. **Specialist** completes the main task.
3. **Reviewer** checks and returns the final answer.

The Marimo UI shows the final result and a tracking panel with the agents that ran. Without credentials, the app uses a clearly labeled preview mode so the UI can be demoed immediately.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
marimo run app.py
```

For live Gemini calls, copy `.env.example` to `.env` and set `GOOGLE_API_KEY`, or set the environment variable in your shell before launching. Google ADK's local Python setup supports Gemini API key authentication or Google Cloud credentials.

To edit the notebook:

```powershell
marimo edit app.py
```
