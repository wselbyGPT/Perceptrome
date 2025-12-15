import { useEffect, useMemo, useState } from "react";

const API_BASE = "/api";

function joinUrl(base, path) {
  const b = base.replace(/\/+$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    // ignore
  }
  return { ok: res.ok, status: res.status, text, json };
}

export default function App() {
  const [openapi, setOpenapi] = useState(null);
  const [status, setStatus] = useState({ state: "idle", msg: "" });

  const apiDocsUrl = useMemo(() => `${API_BASE}/docs`, []);
  const openapiUrl = useMemo(() => joinUrl(API_BASE, "/openapi.json"), []);

  useEffect(() => {
    let alive = true;
    (async () => {
      setStatus({ state: "loading", msg: "Loading OpenAPI…" });
      const r = await fetchJson(openapiUrl);

      if (!alive) return;

      if (!r.ok || !r.json) {
        setOpenapi(null);
        setStatus({
          state: "error",
          msg: `API not reachable at ${openapiUrl} (HTTP ${r.status})`
        });
        return;
      }

      setOpenapi(r.json);
      setStatus({ state: "ok", msg: "API OK" });
    })();

    return () => {
      alive = false;
    };
  }, [openapiUrl]);

  const info = openapi?.info || {};
  const paths = openapi?.paths ? Object.keys(openapi.paths) : [];
  const sample = paths.slice(0, 14);

  return (
    <div className="wrap">
      <header className="top">
        <div>
          <div className="kicker">perceptrome</div>
          <h1 className="title">Sequence intelligence, end-to-end.</h1>
          <p className="sub">
            React UI served by nginx • FastAPI behind <code>/api</code>
          </p>
        </div>

        <div className="actions">
          <a className="btn" href={apiDocsUrl} target="_blank" rel="noreferrer">
            Open API Docs
          </a>
          <button
            className="btn ghost"
            onClick={() => window.location.reload()}
            type="button"
          >
            Refresh
          </button>
        </div>
      </header>

      <section className="grid">
        <div className="card">
          <div className="cardTitle">Backend status</div>
          <div className="pillRow">
            <span className={`pill ${status.state}`}>{status.msg || "—"}</span>
            <span className="pill neutral">
              {info.title ? info.title : "OpenAPI"}
              {info.version ? ` • v${info.version}` : ""}
            </span>
            <span className="pill neutral">{paths.length} routes</span>
          </div>

          <div className="muted">
            Source: <code>{openapiUrl}</code>
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Quick links</div>
          <div className="links">
            <a href="/" className="link">
              /
            </a>
            <a href={`${API_BASE}/docs`} className="link">
              /api/docs
            </a>
            <a href={`${API_BASE}/openapi.json`} className="link">
              /api/openapi.json
            </a>
          </div>

          <div className="muted" style={{ marginTop: 10 }}>
            Tip: keep frontend calls relative (e.g. <code>/api/…</code>) so prod
            stays same-origin.
          </div>
        </div>

        <div className="card span2">
          <div className="cardTitle">Routes (first {sample.length})</div>
          {sample.length === 0 ? (
            <div className="muted">No routes detected yet.</div>
          ) : (
            <div className="routes">
              {sample.map((p) => (
                <div key={p} className="route">
                  <code>{p}</code>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <footer className="foot">
        <span className="muted">
          Deployed static build • no JS source shipped • just <code>dist/</code>
        </span>
      </footer>
    </div>
  );
}
