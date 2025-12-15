export const Store = (() => {
  function get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  }
  function set(key, value) {
    try { localStorage.setItem(key, value); } catch {}
  }
  function getJSON(key, fallback) {
    const s = get(key);
    if (!s) return fallback;
    try { return JSON.parse(s); } catch { return fallback; }
  }
  function setJSON(key, obj) {
    set(key, JSON.stringify(obj ?? null));
  }
  return { get, set, getJSON, setJSON };
})();
