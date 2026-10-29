const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const ingest = (texts) => post("/ingest", { texts });
export const compare = (query) => post("/compare", { query, verbose: false });
export const query = (q) => post("/query", { query: q, verbose: false });
export const health = () =>
  fetch(`${BASE_URL}/health`).then((r) => r.json());
