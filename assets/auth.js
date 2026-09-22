/**
 * SkinAnalytica — assets/auth.js
 * Client-supplied API key handling — the same model as Swagger's own
 * "Authorize" button. The key is never embedded in this file or shipped
 * in any page source; it's typed in by whoever is using the page and held
 * only in that browser's localStorage. Nothing server-side is exposed by
 * committing this file.
 */
const SA_AUTH = {
  KEY_NAME: 'skinanalytica_api_key',

  get() {
    try { return localStorage.getItem(this.KEY_NAME) || ''; } catch { return ''; }
  },
  set(key) {
    try {
      if (key) localStorage.setItem(this.KEY_NAME, key);
      else localStorage.removeItem(this.KEY_NAME);
    } catch {}
  },
  headers() {
    const k = this.get();
    return k ? { 'X-API-Key': k } : {};
  },

  /** Mounts a small key-status control into the given element. Shows a
   * masked preview if a key is stored, an "Add key" prompt otherwise. */
  mountWidget(el) {
    const render = () => {
      const k = this.get();
      el.innerHTML = k
        ? `<button class="sa-auth-btn sa-auth-set" type="button" aria-label="API key is set, starting with ${k.slice(0,4)}. Click to update or remove it.">🔑 ${k.slice(0,4)}••••</button>`
        : `<button class="sa-auth-btn sa-auth-unset" type="button" aria-label="No API key set. Click to add one.">🔑 Add API key</button>`;
      el.querySelector('button').addEventListener('click', () => {
        const cur = this.get();
        const next = prompt(
          cur ? 'Update your SkinAnalytica API key (leave blank to remove):' : 'Enter your SkinAnalytica API key:\n\nThis is stored only in this browser (localStorage) and sent as the X-API-Key header — it is never embedded in this page or sent anywhere else.',
          cur
        );
        if (next === null) return; // cancelled
        this.set(next.trim());
        render();
      });
    };
    render();
  }
};
