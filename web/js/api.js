import { API_PREFIX } from "./config.js";

export const api = (() => {
  function url(path) {
    if (!path) path = "/";
    if (!path.startsWith("/")) path = "/" + path;
    if (path.startsWith(API_PREFIX + "/")) return path;
    return API_PREFIX + path;
  }

  async function request(path, opts = {}) {
    const full = url(path);
    const res = await fetch(full, opts);
    const ctype = (res.headers.get("content-type") || "").toLowerCase();

    if (!res.ok) {
      let bodyText = "";
      try { bodyText = await res.text(); } catch {}
      const msg = bodyText ? bodyText.slice(0, 600) : `${res.status} ${res.statusText}`;
      throw new Error(`${res.status} ${res.statusText}: ${msg}`);
    }

    if (ctype.includes("application/json")) {
      const json = await res.json();
      return { res, json, text: null, ctype };
    }

    const text = await res.text();
    const trimmed = (text || "").trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const json = JSON.parse(trimmed);
        return { res, json, text, ctype };
      } catch {}
    }
    return { res, json: null, text, ctype };
  }

  const get = (path) => request(path, { method: "GET" });
  const post = (path, obj) =>
    request(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(obj ?? {}),
    });

  async function tryPaths(method, paths, payload) {
    let lastErr = null;
    for (const p of paths || []) {
      try {
        if (method === "GET") return await get(p);
        if (method === "POST") return await post(p, payload);
        return await request(p, {
          method,
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload ?? {}),
        });
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error("No endpoints available");
  }

  async function endpointExists(fullUrl) {
    try {
      let r = await fetch(fullUrl, { method: "HEAD" });
      if (r.ok) return true;
      r = await fetch(fullUrl, { method: "GET", headers: { Range: "bytes=0-0" } });
      return r.ok;
    } catch {
      return false;
    }
  }

  return { url, request, get, post, tryPaths, endpointExists };
})();
