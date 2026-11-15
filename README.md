# ATHR - Mars Exploration Platform with a Grounded AI Assistant

NASA Space Apps Challenge 2026 project.

ATHR is an interactive Mars globe (3D / 2D). Select a point or search a place name, and ATHR shows real NASA data for that location: elevation, a local slope estimate, nearby Perseverance waypoints and the rover route. An AI assistant explains the selected location in **Arabic or English**, using only the data the backend retrieved, and lists its sources.

**Live demo:** [https://athr-nasa-1.onrender.com]  
**API:** https://athr-nasa.onrender.com (free tier: the first request after idle time can take about a minute)

## NASA data used
- **MGS MOLA MEGDR 32ppd** (NASA PDS Geosciences): elevation and local slope
- **Mars 2020 Waypoints** (NASA/JPL MMGIS): Perseverance route
- **Mars Trek** (NASA/JPL): surface imagery and place-name search

## How the AI assistant stays grounded
1. The frontend sends the question and the selected point to `POST /api/assistant`.
2. The backend reads elevation, slope and waypoints from the NASA sources itself.
3. The language model receives only those verified facts and explains them.
4. The server attaches the sources and a confidence level. Missing data is reported as missing, and terrain safety stays "undetermined" without validated thresholds.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the AI settings (see below)
uvicorn main:app --port 8000
```
Open the frontend (`index.html`) in a browser. The assistant needs an AI key; **without one, the rest of the platform still works** and the assistant reports that it is not configured.

`.env` settings:
```
ATHR_AI_PROVIDER=openai          # anthropic, or openai (any OpenAI-compatible API)
ATHR_AI_API_KEY=your-key
ATHR_AI_MODEL=model-id
ATHR_AI_BASE_URL=provider-url    # optional
ATHR_CORS_ORIGINS=frontend-url   # when the frontend is hosted elsewhere
```

## Limitations
- Slope is a local estimate from ~1.85 km pixels; it says nothing about rock-scale hazards.
- Hazard classification is pending scientifically validated thresholds.

## AI use disclosure
Parts of the code were written with AI assistance. The in-app assistant uses a large language model constrained to NASA data retrieved by the backend.

## Built with
CesiumJS, Python, FastAPI, HTTPX, Pydantic, Render.
