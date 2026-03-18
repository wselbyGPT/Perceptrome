export function buildNextUrl(): string {
  return encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
}

export function makeWebSocketUrl(): string {
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${window.location.host}/ws`;
}

export function randomTempPassword(length = 18): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%^&*";
  let out = "";
  const arr = new Uint32Array(length);
  crypto.getRandomValues(arr);
  for (let i = 0; i < length; i += 1) {
    out += chars[arr[i] % chars.length];
  }
  return out;
}
