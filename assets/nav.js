/**
 * SkinAnalytica -- assets/nav.js
 * Mobile nav toggle. Below 720px theme.css hides .nav-links entirely with no
 * other way to reach the other two pages -- this wires the hamburger button
 * present on all three pages to reveal it as a dropdown.
 */
(function () {
  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');
  if (!toggle || !links) return;

  function close() {
    links.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
  }
  function open() {
    links.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
  }

  toggle.addEventListener('click', () => {
    links.classList.contains('open') ? close() : open();
  });
  links.addEventListener('click', (e) => {
    if (e.target.tagName === 'A') close();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close();
  });
  document.addEventListener('click', (e) => {
    if (!links.contains(e.target) && !toggle.contains(e.target)) close();
  });
})();
