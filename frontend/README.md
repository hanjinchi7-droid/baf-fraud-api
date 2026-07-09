# Streamlit front-end

A simple UI (course-taught Streamlit) that calls the FastAPI `/predict` service and shows the
fraud probability, deterministic risk route, and per-instance SHAP risk factors.

## Run it (local)

1. Start the API in one terminal:
   ```powershell
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the UI in a second terminal:
   ```powershell
   python -m streamlit run frontend/streamlit_app.py
   ```
3. A browser opens at `http://localhost:8501`. The form is pre-filled with a real sample
   application — click **Score application**. Take a screenshot for the report.

## Point it at the deployed API instead
In the sidebar, set **API base URL** to your Render URL
(e.g. `https://baf-fraud-api.onrender.com`) and click **Check /health** first to wake it.

## Optional: deploy the UI too
Push to GitHub and deploy on **Streamlit Community Cloud** (free): set the main file to
`frontend/streamlit_app.py` and it installs `frontend/requirements.txt` automatically. Not required
for the assignment — a local screenshot is enough.
