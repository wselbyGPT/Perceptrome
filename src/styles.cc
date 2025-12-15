:root {
  color-scheme: dark;
  --bg: #070a10;
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.04);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --edge: rgba(255,255,255,0.12);
  --good: rgba(90, 255, 160, 0.18);
  --bad: rgba(255, 90, 120, 0.18);
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  background:
    radial-gradient(900px 500px at 20% 10%, rgba(120, 80, 255, 0.14), transparent 60%),
    radial-gradient(900px 500px at 80% 20%, rgba(80, 255, 210, 0.12), transparent 60%),
    var(--bg);
  color: var(--text);
}

.wrap { max-width: 1040px; margin: 0 auto; padding: 28px 18px 36px; }

.top {
  display: flex;
  gap: 18px;
  align-items: flex-start;
  justify-content: space-between;
  padding: 18px;
  border: 1px solid var(--edge);
  border-radius: 18px;
  background: linear-gradient(180deg, var(--panel), var(--panel2));
}

.kicker {
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
}

.title { margin: 8px 0 6px; font-size: 28px; line-height: 1.15; }
.sub { margin: 0; color: var(--muted); }

.actions { display: flex; gap: 10px; }

.btn {
  appearance: none;
  border: 1px solid var(--edge);
  background: rgba(255,255,255,0.08);
  color: var(--text);
  padding: 10px 12px;
  border-radius: 14px;
  text-decoration: none;
  font-weight: 600;
  cursor: pointer;
}
.btn:hover { background: rgba(255,255,255,0.11); }
.btn.ghost { background: transparent; }

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 14px;
}

.card {
  border: 1px solid var(--edge);
  border-radius: 18px;
  padding: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.035));
}
.cardTitle {
  font-weight: 800;
  margin-bottom: 10px;
  letter-spacing: 0.01em;
}
.span2 { grid-column: span 2; }

.pillRow { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.pill {
  border: 1px solid var(--edge);
  border-radius: 999px;
  padding: 6px 10px;
  font-weight: 700;
  font-size: 12px;
}
.pill.ok { background: var(--good); }
.pill.error { background: var(--bad); }
.pill.loading { background: rgba(120, 80, 255, 0.16); }
.pill.neutral { background: rgba(255,255,255,0.06); }

.muted { color: var(--muted); font-size: 13px; }

.links { display: flex; flex-direction: column; gap: 8px; }
.link {
  color: rgba(160, 220, 255, 0.92);
  text-decoration: none;
  padding: 8px 10px;
  border-radius: 12px;
  border: 1px solid rgba(160,220,255,0.18);
  background: rgba(160,220,255,0.06);
}
.link:hover { background: rgba(160,220,255,0.10); }

.routes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.route {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(0,0,0,0.15);
  border-radius: 12px;
  padding: 10px;
  overflow: hidden;
  text-overflow: ellipsis;
}

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.95em;
}

.foot { margin-top: 12px; padding: 0 6px; }
