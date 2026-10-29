import React, { useState, useRef, useEffect } from "react";
import { query as runQuery } from "./api";

// Thresholds for the honest-confidence banner. These map directly onto
// ConfidenceReport.overall from src/confidence.py (Eq. 14) -- nothing here
// is invented client-side, it's the same number the pipeline used to decide
// whether to keep iterating.
function confidenceTier(c) {
  if (c >= 0.75) return { label: "High confidence", tone: "ok" };
  if (c >= 0.5) return { label: "Moderate confidence", tone: "warn" };
  return { label: "Low confidence — verify before relying on this", tone: "danger" };
}

export default function Chat({ kbReady = true }) {
  const [messages, setMessages] = useState([]); // {role, text, meta}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const res = await runQuery(q);
      if (res.error) throw new Error(res.error);
      const lastIter = res.trace[res.trace.length - 1];
      const scores = lastIter?.doc_attention_scores || {};
      const topDocIds = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([id]) => Number(id));
      const citations = topDocIds
        .map((id) => lastIter.retrieved_texts?.[lastIter.retrieved_doc_ids.indexOf(id)])
        .filter(Boolean);

      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.response,
          meta: {
            confidence: lastIter?.confidence ?? null,
            iterations: res.trace.length,
            converged: res.converged_reason,
            citations,
          },
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "", meta: { error: e.message + " — did you build the knowledge base and start the backend?" } },
      ]);
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="chat-shell">
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-column">
          {messages.length === 0 && (
            <div className="chat-empty">
              <span className="empty-glyph-lg">◈</span>
              <div className="chat-empty-title">SOAR&#8209;RAG</div>
              <div className="chat-empty-sub">
                {kbReady
                  ? "Ask a question grounded in the knowledge base you built. Every answer is scored for confidence and cites which documents it actually drew on."
                  : "Build a knowledge base in the sidebar first — answers here are always grounded in it, nothing is invented."}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <ChatBubble key={i} msg={m} />
          ))}
          {busy && (
            <div className="chat-row assistant">
              <div className="chat-bubble assistant typing">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="chat-composer-outer">
        <div className="chat-composer">
          <textarea
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={kbReady ? "Ask something grounded in the knowledge base…" : "Build a knowledge base in the sidebar to start chatting…"}
            rows={1}
          />
          <button className="btn btn-primary send-btn" onClick={send} disabled={busy || !input.trim()} title="Send">
            ↑
          </button>
        </div>
        <div className="chat-disclaimer">SOAR-RAG answers only from the documents in your knowledge base, and can make mistakes — check the citations.</div>
      </div>
    </div>
  );
}

function ChatBubble({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="chat-row user">
        <div className="chat-bubble user">{msg.text}</div>
      </div>
    );
  }

  if (msg.meta?.error) {
    return (
      <div className="chat-row assistant">
        <div className="chat-bubble assistant error-bubble">{msg.meta.error}</div>
      </div>
    );
  }

  const tier = confidenceTier(msg.meta.confidence ?? 0);
  return (
    <div className="chat-row assistant">
      <div className="chat-bubble assistant">
        <p className="chat-text">{msg.text || "(empty response)"}</p>

        <div className={`confidence-pill tone-${tier.tone}`}>
          {tier.label}
          {msg.meta.confidence != null && (
            <span className="confidence-num"> · {(msg.meta.confidence * 100).toFixed(0)}%</span>
          )}
        </div>

        <div className="chat-footnote">
          {msg.meta.iterations} feedback iteration{msg.meta.iterations === 1 ? "" : "s"} · converged: {msg.meta.converged?.replaceAll("_", " ")}
        </div>

        {msg.meta.citations?.length > 0 && (
          <details className="citations">
            <summary>Sources this answer drew on ({msg.meta.citations.length})</summary>
            {msg.meta.citations.map((c, idx) => (
              <div className="citation-item" key={idx}>
                <span className="citation-idx">[{idx + 1}]</span> {c}
              </div>
            ))}
          </details>
        )}
      </div>
    </div>
  );
}
