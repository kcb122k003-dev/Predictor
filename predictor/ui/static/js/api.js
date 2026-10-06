// JSON API client. Every call goes to this computer (127.0.0.1); nothing leaves the machine.

async function request(method, url, body, { raw = false } = {}) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  if (raw) return res;
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res.text();
}

export const api = {
  get: (url) => request("GET", url),
  post: (url, body) => request("POST", url, body ?? {}),
  put: (url, body) => request("PUT", url, body ?? {}),
  patch: (url, body) => request("PATCH", url, body ?? {}),
  del: (url) => request("DELETE", url),
  upload: (url, form) => request("POST", url, form),
};
