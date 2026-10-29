import React, { useState, useCallback } from "react";
import { ingest, compare } from "./api";
import AttentionTrace from "./AttentionTrace.jsx";
import Chat from "./Chat.jsx";

const DEMO_CORPUS = [
  "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, completed in 1889 for the World's Fair.",
  "Paris is the capital and most populous city of France, known for its art, fashion, gastronomy, and culture.",
  "The Great Wall of China is a series of fortifications built across the historical northern borders of China.",
  "Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy.",
  "Mount Everest, located in the Mahalangur Himal sub-range of the Himalayas, is Earth's highest mountain above sea level.",
  "The Amazon rainforest is a moist broadleaf tropical rainforest that covers most of the Amazon basin in South America.",
];

export default function App() {
  const [corpusText, setCorpusText] = useState(DEMO_CORPUS.join("\n\n"));
  const [query, setQuery] = useState("When and where was the Eiffel Tower completed?");
  const [stage, setStage] = useState("idle");
  const [chunkCount, setChunkCount] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [view, setView] = useState("chat"); // "chat" | "compare"
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatKey, setChatKey] = useState(0); // bump to reset Chat's message history

  const handleIngest = useCallback(async () => {
    setStage("ingesting");
    setErrorMsg("");
    try {
      const texts = corpusText.split(/\n\s*\n/).map((t) => t.trim()).filter(Boolean);
      const res = await ingest(texts);
      setChunkCount(res.num_chunks);
      setStage("ready");
    } catch (e) {
      setErrorMsg(e.message + " — is the backend running on the configured VITE_API_URL?");
      setStage("error");
    }
  }, [corpusText]);

  const handleRun = useCallback(async () => {
    setStage("running");
    setErrorMsg("");
    try {
      const res = await compare(query);
      setResult(res);
      setStage("done");
    } catch (e) {
      setErrorMsg(e.message);
      setStage("error");
    }
  }, [query]);

  const kbReady = stage === "ready" || stage === "done";

  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="sidebar">
        <div className="sidebar-top">
          <button
            className="icon-btn sidebar-toggle"
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? "«" : "»"}
          </button>
          {sidebarOpen && (
            <div className="brand">
              <span className="brand-mark">◈</span>
              <div>
                <div className="brand-title">SOAR&#8209;RAG</div>
                <div className="brand-sub">attention&#8209;guided retrieval</div>
              </div>
            </div>
          )}
        </div>

        {sidebarOpen && (
          <>
            <button
              className="btn btn-primary new-chat-btn"
              onClick={() => { setView("chat"); setChatKey((k) => k + 1); }}
            >
              + New chat
            </button>

            <nav className="side-nav">
              <button
                className={`side-nav-item ${view === "chat" ? "active" : ""}`}
                onClick={() => setView("chat")}
              >
                <span className="side-nav-icon">◆</span> Chat
              </button>
              <button
                className={`side-nav-item ${view === "compare" ? "active" : ""}`}
                onClick={() => setView("compare")}
              >
                <span className="side-nav-icon">⇄</span> Compare baseline vs SOAR
              </button>
            </nav>

            <div className="sidebar-divider" />

            <div className="kb-section">
              <div className="kb-section-title">Knowledge base</div>
              <p className="kb-hint">
                One document per blank-line-separated block. The chat only
                answers from what's built here.
              </p>
              <textarea
                className="corpus-input"
                value={corpusText}
                onChange={(e) => setCorpusText(e.target.value)}
                rows={8}
                spellCheck={false}
              />
              <button
                className="btn btn-secondary kb-build-btn"
                onClick={handleIngest}
                disabled={stage === "ingesting"}
              >
                {stage === "ingesting" ? "Building index…" : "Build knowledge base"}
              </button>
              {errorMsg && <div className="error-box">{errorMsg}</div>}
            </div>

            <div className="sidebar-bottom">
              <StageBadge stage={stage} chunkCount={chunkCount} />
            </div>
          </>
        )}
      </aside>

      <main className="main-area">
        {view === "chat" ? (
          <Chat key={chatKey} kbReady={kbReady} />
        ) : (
          <div className="compare-view">
            <div className="compare-header">
              <h2 className="panel-title">Compare baseline vs SOAR&#8209;RAG</h2>
              <p className="panel-hint">
                Runs the same query through a plain single-pass retriever and
                through the full attention-feedback loop, side by side.
              </p>
            </div>
            <div className="compare-query-row">
              <input
                className="query-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask something answerable from the knowledge base…"
                onKeyDown={(e) => e.key === "Enter" && handleRun()}
              />
              <button
                className="btn btn-primary"
                onClick={handleRun}
                disabled={stage === "running" || !kbReady}
              >
                {stage === "running" ? "Running loop…" : "Run comparison"}
              </button>
            </div>
            {!kbReady && (
              <div className="empty-state">
                <span className="empty-glyph">↳</span>
                Build the knowledge base in the sidebar first.
              </div>
            )}
            {kbReady && !result && stage !== "running" && (
              <div className="empty-state">
                <span className="empty-glyph">↳</span>
                Run a query to see the attention feedback loop trace here.
              </div>
            )}
            {stage === "running" && <div className="empty-state">Generating with attention capture…</div>}
            {result && <AttentionTrace result={result} />}
          </div>
        )}
      </main>
    </div>
  );
}

function StageBadge({ stage, chunkCount }) {
  const labels = {
    idle: "not connected",
    ingesting: "indexing…",
    ready: `ready · ${chunkCount ?? 0} chunks`,
    running: "generating…",
    done: `ready · ${chunkCount ?? 0} chunks`,
    error: "error",
  };
  return <div className={`stage-badge stage-${stage}`}>{labels[stage]}</div>;
}
