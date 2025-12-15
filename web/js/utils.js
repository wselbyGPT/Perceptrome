export const U = (() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $all = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);

    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = String(v ?? "");
      else if (k === "text") n.textContent = String(v ?? "");
      else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null) n.setAttribute(k, String(v));
    }

    for (const kid of kids) {
      if (kid === null || kid === undefined || kid === false) continue;

      if (typeof kid === "string" || typeof kid === "number" || typeof kid === "boolean") {
        n.appendChild(document.createTextNode(String(kid)));
      } else {
        n.appendChild(kid);
      }
    }

    return n;
  }

  const esc = (s) =>
    String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  function fmtBytes(b) {
    b = Number(b || 0);
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (b >= 1024 && i < u.length - 1) {
      b /= 1024;
      i++;
    }
    return `${b.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
  }

  function fmtIso(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return String(iso);
      return d.toLocaleString();
    } catch {
      return String(iso);
    }
  }

  const pct = (x) => {
    if (x === null || x === undefined) return null;
    const n = Number(x);
    return Number.isFinite(n) ? n : null;
  };

  // Fixed: remove extra ')' and normalize numeric inputs
  const clamp = (v, a, b) => {
    const vv = Number(v);
    const aa = Number(a);
    const bb = Number(b);
    if (!Number.isFinite(vv) || !Number.isFinite(aa) || !Number.isFinite(bb)) return vv;
    const lo = Math.min(aa, bb);
    const hi = Math.max(aa, bb);
    return Math.max(lo, Math.min(hi, vv));
  };

  function debounce(fn, ms = 150) {
    let t = null;
    return function (...args) {
      if (t) clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  return { $, $all, el, esc, fmtBytes, fmtIso, pct, clamp, debounce };
})();
