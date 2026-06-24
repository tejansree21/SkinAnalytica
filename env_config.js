/**
 * SkinAnalytica — env_config.js
 * Auto-detects environment and sets correct API URLs.
 * Include this FIRST in skinanalytica_shell.html before any page loads.
 */

(function() {
  const hostname = window.location.hostname;

  // Determine environment
  const isLocal   = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '';
  const isVercel  = hostname.includes('vercel.app');
  const isHF      = hostname.includes('hf.space');

  let API_BASE, ASSISTANT_BASE, DELIVERY_BASE;

  if (isLocal) {
    API_BASE       = 'http://localhost:8001';
    ASSISTANT_BASE = 'http://localhost:8002';
    DELIVERY_BASE  = 'http://localhost:8003';
  } else if (isVercel || isHF) {
    // Production — Render service URLs
    API_BASE       = 'https://skinanalytica-api.onrender.com';
    ASSISTANT_BASE = 'https://skinanalytica-assistant.onrender.com';
    DELIVERY_BASE  = 'https://skinanalytica-delivery.onrender.com';
  } else {
    // Fallback — same origin or custom domain
    API_BASE       = window.SKINANALYTICA_API_URL       || 'http://localhost:8001';
    ASSISTANT_BASE = window.SKINANALYTICA_ASSISTANT_URL || 'http://localhost:8002';
    DELIVERY_BASE  = window.SKINANALYTICA_DELIVERY_URL  || 'http://localhost:8003';
  }

  // Expose globally so all pages can use window.SA_CONFIG
  window.SA_CONFIG = {
    API_BASE,
    ASSISTANT_BASE,
    DELIVERY_BASE,
    ENV: isLocal ? 'local' : isVercel ? 'vercel' : isHF ? 'hf' : 'unknown',
  };

  console.log('[SkinAnalytica] Config:', window.SA_CONFIG);
})();
