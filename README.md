# Lingua — Local Web App

A professional web frontend for your NLP semester project. FastAPI backend + custom HTML/CSS/JS frontend, runs locally on your machine.

## What's inside

```
lingua/
├── app.py                          # FastAPI backend (Python)
├── requirements.txt                # Python dependencies
├── static/
│   └── index.html                  # Frontend (HTML + CSS + JS in one file)
├── export_models_from_colab.py     # Snippet to run in Colab once
├── model.pkl                       # YOU GENERATE THIS from Colab (see below)
└── README.md                       # This file
```

## Setup (one-time, ~5 minutes)

### Step 1 — Export the model from your Colab notebook

1. Open your Colab notebook.
2. Make sure **Run all** has been done (so all your models are trained in memory).
3. Open `export_models_from_colab.py` from this folder, copy its contents.
4. Paste as a **new cell at the end** of your notebook and run it.
5. A file called `model.pkl` will download to your computer.
6. **Move `model.pkl` into this `lingua/` folder** (next to `app.py`).

### Step 2 — Install Python dependencies

Open VS Code or PyCharm in the `lingua/` folder. In the terminal:

```bash
pip install -r requirements.txt
```

(If you have multiple Python versions, use `python3 -m pip install -r requirements.txt`.)

### Step 3 — Run the server

```bash
python app.py
```

You should see:

```
Loading model from .../model.pkl ...
  vocabulary size  : 4,387
  n-gram orders    : 1 .. 6
  pos model vocab  : ...
  neg model vocab  : ...
Model loaded.

==================================================
 Lingua server starting...
 Open http://localhost:8000 in your browser
==================================================
```

### Step 4 — Open in browser

Go to **http://localhost:8000**

That's it. Four tabs:
- **01 Auto-Complete** — live predictions across 5 n-gram models as you type
- **02 Sentiment** — hybrid classifier (bigram + negation + lexicon)
- **03 Perplexity** — side-by-side sentence comparison
- **04 About** — project metadata

## For the viva demo

1. Run `python app.py` before the viva starts.
2. Open the browser to `http://localhost:8000`.
3. Full-screen the browser tab.
4. Walk through the tabs using the example chips (one-click demos).

## Tech stack

- **Backend:** Python 3 · FastAPI · uvicorn · NLTK
- **Frontend:** HTML5 · CSS (custom, no framework) · vanilla JavaScript
- **Typography:** Fraunces (display) · Manrope (body) · JetBrains Mono (code)
- **Theme:** Dark editorial with warm amber accents

## Troubleshooting

| Problem | Solution |
|---|---|
| `model.pkl not found` | Run Step 1 again, make sure the file is in the `lingua/` folder |
| `Port 8000 already in use` | Change the port in `app.py` (last line) — e.g. `port=8001` |
| `ModuleNotFoundError: fastapi` | Run `pip install -r requirements.txt` |
| API returns 500 errors | Check the terminal — usually a missing variable in `model.pkl` |
| Frontend looks broken | Hard-refresh (Ctrl+Shift+R) to clear browser cache |

## What to say in the viva if asked about this

*"I built a custom web frontend using FastAPI for the backend API and vanilla HTML/CSS/JS for the frontend. The trained n-gram models are serialized with pickle and loaded once at server startup. The frontend calls three REST endpoints — `/api/autocomplete`, `/api/sentiment`, `/api/perplexity` — and renders results in a typography-driven editorial layout. No external UI frameworks; the design is custom-built."*

---

Built by Jawwad Hussain · NLP Semester Project · Air University · May 2026
