# FastAPI deployment

Run locally from the project root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, check `GET /health`, then send an
application JSON object to `POST /predict`.

For Render, use `pip install -r requirements.txt` as the build command and
`uvicorn api.main:app --host 0.0.0.0 --port $PORT` as the start command.
