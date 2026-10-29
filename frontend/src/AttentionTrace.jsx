import React, { useState } from "react";

export default function AttentionTrace({ result }) {
  const { baseline, soar } = result;
  return (
    <div className="trace">
      <div className="answer-row">
        <AnswerCard label="Baseline RAG" sub="single retrieval, no feedback" data={baseline} tone="cool" />
        <AnswerCard label="SOAR-RAG" sub={`converged: ${soar.converged_reason.replaceAll("_", " ")}`} data={soar} tone="warm" />
      </div>

      <div className="stat-row">
        <Stat label="Iterations" value={soar.trace.length} />
        <Stat label="Context compression" value={`${(soar.context_compression_ratio * 100).toFixed(0)}%`} />
        <Stat label="SOAR latency" value={`${soar.total_latency_s.toFixed(2)}s`} />
        <Stat label="Baseline latency" value={`${baseline.total_latency_s.toFixed(2)}s`} />
      </div>

      <h3 className="panel-title panel-title-spaced">Attention feedback trace</h3>
      <p className="panel-hint">
        Each row is a retrieved document; bar length is the share of the model's
        attention it received while generating the answer.
      </p>
      {soar.trace.map((iter) => (
        <IterationCard key={iter.iteration} iter={iter} />
      ))}
    </div>
  );
}

function AnswerCard({ label, sub, data, tone }) {
  return (
    <div className={`answer-card tone-${tone}`}>
      <div className="answer-card-head">
        <span className="answer-card-label">{label}</span>
        <span className="answer-card-sub">{sub}</span>
      </div>
      <p className="answer-text">{data.response || "(empty response)"}</p>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function IterationCard({ iter }) {
  const [open, setOpen] = useState(iter.iteration === 1);
  const scores = iter.doc_attention_scores || {};
  const maxScore = Math.max(...Object.values(scores), 1e-9);

  return (
    <div className="iter-card">
      <button className="iter-head" onClick={() => setOpen((o) => !o)}>
        <span className="iter-num">iter {iter.iteration}</span>
        <span className="iter-meta">
          k={iter.top_k} · conf={iter.confidence.toFixed(2)} · entropy={iter.attention_entropy.toFixed(2)}
        </span>
        <span className="iter-chevron">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="iter-body">
          {iter.retrieved_doc_ids.map((docId, idx) => {
            const score = scores[docId] ?? 0;
            const pct = Math.max(4, (score / maxScore) * 100);
            const text = iter.retrieved_texts?.[idx] || "";
            return (
              <div className="doc-row" key={docId}>
                <div className="doc-bar-track">
                  <div className="doc-bar-fill" style={{ width: `${pct}%` }} />
                </div>
                <div className="doc-text">{text.length > 140 ? text.slice(0, 140) + "…" : text}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
