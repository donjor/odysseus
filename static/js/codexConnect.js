// static/js/codexConnect.js — ChatGPT-subscription (Codex / OAuth) connect UI
//
// Owner-scoped, self-service: any signed-in user links their own ChatGPT Plus/Pro
// subscription via the device-code flow. The OAuth dance runs server-side (Odysseus
// is a remote host), so this module only ever shows the user-facing `user_code` +
// `verification_uri` and polls connection STATE — never any token material.
//
// Backend (see routes/codex_oauth_routes.py), all under /api/providers/openai-codex:
//   POST  /connect                  -> {attempt_id, endpoint_id, user_code, verification_uri, interval, expires_at}
//   GET   /connect/{attempt_id}     -> {status, user_code, verification_uri, expires_at, ...}
//   POST  /connect/{attempt_id}/cancel
//   GET   /status                   -> {connected:[...], pending:[...]}
//   POST  /disconnect/{endpoint_id}

import uiModule from './ui.js';

const BASE = '/api/providers/openai-codex';

function el(id) { return document.getElementById(id); }
function esc(s) { return uiModule.esc(String(s == null ? '' : s)); }

async function api(path, opts = {}) {
  const res = await fetch(BASE + path, { credentials: 'same-origin', ...opts });
  let body = null;
  try { body = await res.json(); } catch (_) { /* empty / non-JSON */ }
  if (!res.ok) {
    const detail = (body && (body.detail || body.error)) || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return body || {};
}

// ─── Module state for the in-flight device-code attempt ────────────────────
let tick = null;        // 1s interval handle
let attempt = null;     // {id, intervalSec, expiresMs, sinceLastPoll}
let initialized = false;

function setMsg(html, kind = '') {
  const m = el('codex-msg');
  if (!m) return;
  m.className = kind ? `codex-msg codex-msg-${kind}` : 'codex-msg';
  m.innerHTML = html;
}

function clearMsg() { setMsg(''); }

// ─── Render: connected accounts ────────────────────────────────────────────
function renderConnected(connected) {
  const wrap = el('codex-connected');
  if (!wrap) return;
  if (!connected || connected.length === 0) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = connected.map(c => {
    const ok = c.enabled && c.status === 'active';
    const state = c.last_error
      ? `<span class="codex-state codex-state-err" title="${esc(c.last_error)}">needs attention</span>`
      : ok
        ? '<span class="codex-state codex-state-ok">connected</span>'
        : `<span class="codex-state">${esc(c.status || 'inactive')}</span>`;
    return `
      <div class="codex-account">
        <span class="codex-dot ${ok ? 'codex-dot-on' : ''}"></span>
        <span class="codex-account-name">${esc(c.name || 'ChatGPT (Codex)')}</span>
        ${state}
        <button class="admin-btn-delete codex-disc" data-codex-action="disconnect" data-ep="${esc(c.endpoint_id)}">Disconnect</button>
      </div>`;
  }).join('');
}

// ─── Render: idle (no active attempt) ──────────────────────────────────────
function renderIdle() {
  const flow = el('codex-flow');
  if (!flow) return;
  flow.innerHTML = `
    <div class="codex-actions">
      <button class="admin-btn-add" data-codex-action="connect">Connect ChatGPT</button>
    </div>`;
}

// ─── Render: pending device-code ───────────────────────────────────────────
function renderPending(info) {
  const flow = el('codex-flow');
  if (!flow) return;
  flow.innerHTML = `
    <div class="codex-pending">
      <div class="codex-step">1. Open the ChatGPT login page:</div>
      <a class="admin-btn-add codex-open-btn" href="${esc(info.verification_uri)}" target="_blank" rel="noopener noreferrer" data-codex-action="open">Open login page ↗</a>
      <div class="codex-step">2. Enter this code when asked:</div>
      <div class="codex-code-row">
        <code class="codex-code" id="codex-usercode">${esc(info.user_code || '')}</code>
        <button class="admin-btn-sm" data-codex-action="copy" title="Copy code">Copy</button>
      </div>
      <div class="codex-wait"><span class="admin-spinner"></span><span id="codex-countdown">Waiting for you to authorize…</span></div>
      <button class="admin-btn-sm" data-codex-action="cancel">Cancel</button>
    </div>`;
}

function fmtRemaining(ms) {
  if (ms <= 0) return 'expiring…';
  const s = Math.round(ms / 1000);
  const m = Math.floor(s / 60);
  return `Waiting for you to authorize… (${m}:${String(s % 60).padStart(2, '0')} left)`;
}

// ─── Poll loop (single 1s ticker drives countdown + interval-paced poll) ────
function stopFlow() {
  if (tick) { clearInterval(tick); tick = null; }
  attempt = null;
}

function beginFlow(info) {
  stopFlow();
  const expiresMs = info.expires_at ? Date.parse(info.expires_at) : (Date.now() + 600000);
  attempt = {
    id: info.attempt_id,
    intervalSec: Math.max(2, info.interval || 5),
    expiresMs,
    sinceLastPoll: 0,
  };
  renderPending(info);
  tick = setInterval(onTick, 1000);
}

async function onTick() {
  if (!attempt) return;
  const remaining = attempt.expiresMs - Date.now();
  const cd = el('codex-countdown');
  if (cd) cd.textContent = fmtRemaining(remaining);
  if (remaining <= 0) {
    const id = attempt.id;
    stopFlow();
    try { await api(`/connect/${id}/cancel`, { method: 'POST' }); } catch (_) {}
    renderIdle();
    setMsg('Login code expired. Try connecting again.', 'warn');
    loadStatus();
    return;
  }
  attempt.sinceLastPoll += 1;
  if (attempt.sinceLastPoll < attempt.intervalSec) return;
  attempt.sinceLastPoll = 0;

  let st;
  try {
    st = await api(`/connect/${attempt.id}`);
  } catch (e) {
    // Transient poll error — keep waiting; surface only if it persists.
    return;
  }
  if (!attempt) return; // cancelled while awaiting
  switch (st.status) {
    case 'authorized':
      stopFlow();
      renderIdle();
      setMsg('ChatGPT connected. Its models are now available in the model picker.', 'ok');
      loadStatus();
      break;
    case 'expired':
      stopFlow();
      renderIdle();
      setMsg('Login code expired. Try connecting again.', 'warn');
      loadStatus();
      break;
    case 'cancelled':
      stopFlow();
      renderIdle();
      loadStatus();
      break;
    case 'error':
      stopFlow();
      renderIdle();
      setMsg('Login failed. Please try again.', 'err');
      loadStatus();
      break;
    // 'pending' → keep ticking
  }
}

// ─── Actions ───────────────────────────────────────────────────────────────
async function startConnect(btn) {
  clearMsg();
  if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
  try {
    const info = await api('/connect', { method: 'POST' });
    beginFlow(info);
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = 'Connect ChatGPT'; }
    setMsg(`Could not start login: ${esc(e.message)}`, 'err');
  }
}

async function cancelAttempt() {
  if (!attempt) { renderIdle(); return; }
  const id = attempt.id;
  stopFlow();
  renderIdle();
  try { await api(`/connect/${id}/cancel`, { method: 'POST' }); } catch (_) {}
  loadStatus();
}

async function disconnect(endpointId, btn) {
  if (!endpointId) return;
  if (btn) { btn.disabled = true; btn.textContent = 'Removing…'; }
  try {
    await api(`/disconnect/${endpointId}`, { method: 'POST' });
    setMsg('Disconnected.', '');
    loadStatus();
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = 'Disconnect'; }
    setMsg(`Could not disconnect: ${esc(e.message)}`, 'err');
  }
}

function copyCode(btn) {
  const code = el('codex-usercode');
  if (!code) return;
  const text = code.textContent || '';
  const flash = (label) => { if (btn) { const p = btn.textContent; btn.textContent = label; setTimeout(() => { btn.textContent = p; }, 1200); } };
  // navigator.clipboard needs a secure context (HTTPS / localhost). Odysseus is
  // commonly served over plain HTTP on a LAN IP, where it's undefined — fall
  // back to a hidden-textarea execCommand copy so the button still works.
  const legacyCopy = () => {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-1000px';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      flash(ok ? 'Copied' : 'Copy failed');
    } catch (_) { flash('Copy failed'); }
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => flash('Copied')).catch(legacyCopy);
  } else {
    legacyCopy();
  }
}

// ─── Status load (also resumes an interrupted pending attempt) ─────────────
async function loadStatus() {
  const wrap = el('codex-connected');
  try {
    const data = await api('/status');
    renderConnected(data.connected || []);
    // If a device-code login is mid-flight and we aren't already tracking it
    // (e.g. modal reopened), resume the poll so the user sees it complete.
    const pend = (data.pending || [])[0];
    if (pend && pend.status === 'pending' && !attempt) {
      beginFlow({
        attempt_id: pend.attempt_id,
        user_code: pend.user_code,
        verification_uri: pend.verification_uri,
        expires_at: pend.expires_at,
        interval: 5,
      });
    } else if (!attempt) {
      renderIdle();
    }
  } catch (e) {
    // Owner-scoped endpoint; on 401/unknown just show the idle action.
    if (wrap) wrap.innerHTML = '';
    if (!attempt) renderIdle();
  }
}

// ─── Public API ────────────────────────────────────────────────────────────
export function initCodexConnect() {
  if (initialized) return;
  const card = el('codex-card');
  if (!card) return;
  initialized = true;
  renderIdle();
  // One delegated click handler for the whole card.
  card.addEventListener('click', (e) => {
    const t = e.target.closest('[data-codex-action]');
    if (!t) return;
    const action = t.dataset.codexAction;
    if (action === 'open') return; // real <a>, let it navigate
    e.preventDefault();
    switch (action) {
      case 'connect': startConnect(t); break;
      case 'cancel': cancelAttempt(); break;
      case 'copy': copyCode(t); break;
      case 'disconnect': disconnect(t.dataset.ep, t); break;
    }
  });
}

export function refreshCodexStatus() {
  if (!el('codex-card')) return;
  loadStatus();
}

const codexConnectModule = { initCodexConnect, refreshCodexStatus };
export default codexConnectModule;
