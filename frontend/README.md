# SOAR-RAG frontend

React + Vite console for the SOAR-RAG backend (`app/api.py`). Shows a
baseline-RAG vs SOAR-RAG answer side by side, and — the actual point of the
project — a per-iteration attention trace: for every retrieved document, a
bar showing how much of the model's attention it actually received while
generating the answer.

## Run locally
```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_URL to wherever app/api.py is running
npm run dev
```
Opens on http://localhost:5173. The backend needs CORS enabled for it
(already set in `app/api.py`) and must be reachable at `VITE_API_URL`.

Since the backend needs a GPU to load the generator model, the typical setup
during development is: run `app/api.py` in the Colab notebook (exposed via
`ngrok` or Colab's port forwarding), point `VITE_API_URL` at that public URL,
and run the frontend locally.

## What it calls
- `POST /ingest` — send the corpus, builds the FAISS index + loads the model
- `POST /compare` — runs both `BaselineRAGPipeline` and `SOARRAGPipeline` on
  the same query and returns both traces, which is what this UI renders

## Build for deployment
```bash
npm run build
```
Outputs static files to `dist/` — deployable to Vercel/Netlify/GitHub Pages
like any static site, same as your other RAG project's frontend.
