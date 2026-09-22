'use strict';
/**
 * SkinAnalytica -- tests_js/test_nav.js
 * Tests assets/nav.js's mobile-nav toggle logic -- the only way to reach
 * the other pages below 720px (theme.css hides .nav-links entirely), so a
 * broken toggle is a real navigation dead-end, not a cosmetic bug. Loaded
 * via node:vm against the real file, not a reimplementation.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

const { FakeElement, makeFakeDocument } = require('./dom_stub');

const NAV_SRC = fs.readFileSync(path.join(__dirname, '..', 'assets', 'nav.js'), 'utf8');

function loadNav({ withElements = true } = {}) {
  const toggle = new FakeElement('button');
  const links = new FakeElement('div');
  const document = makeFakeDocument(withElements ? { navToggle: toggle, navLinks: links } : {});
  const context = { document };
  vm.createContext(context);
  vm.runInContext(NAV_SRC, context);
  return { toggle, links, document };
}

test('missing navToggle/navLinks elements: script returns early, does not throw', () => {
  assert.doesNotThrow(() => loadNav({ withElements: false }));
});

test('clicking the toggle when closed opens the menu', () => {
  const { toggle, links } = loadNav();
  toggle.dispatchEvent('click', {});
  assert.equal(links.classList.contains('open'), true);
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
});

test('clicking the toggle again when open closes the menu', () => {
  const { toggle, links } = loadNav();
  toggle.dispatchEvent('click', {}); // open
  toggle.dispatchEvent('click', {}); // close
  assert.equal(links.classList.contains('open'), false);
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
});

test('pressing Escape closes an open menu', () => {
  const { toggle, links, document } = loadNav();
  toggle.dispatchEvent('click', {}); // open
  document.dispatchEvent('keydown', { key: 'Escape' });
  assert.equal(links.classList.contains('open'), false);
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
});

test('pressing a non-Escape key does not close the menu', () => {
  const { toggle, links, document } = loadNav();
  toggle.dispatchEvent('click', {}); // open
  document.dispatchEvent('keydown', { key: 'Enter' });
  assert.equal(links.classList.contains('open'), true);
});

test('clicking a nav link (<a>) closes the menu -- selecting a destination should dismiss it', () => {
  const { toggle, links } = loadNav();
  toggle.dispatchEvent('click', {}); // open
  const linkNode = new FakeElement('a');
  links.dispatchEvent('click', { target: linkNode });
  assert.equal(links.classList.contains('open'), false);
});

test('clicking a non-link inside the menu does not close it', () => {
  const { toggle, links } = loadNav();
  toggle.dispatchEvent('click', {}); // open
  const spanNode = new FakeElement('span');
  links.dispatchEvent('click', { target: spanNode });
  assert.equal(links.classList.contains('open'), true);
});

test('clicking outside both the menu and the toggle closes an open menu', () => {
  const { toggle, links, document } = loadNav();
  toggle.dispatchEvent('click', {}); // open
  const outsideNode = new FakeElement('div');
  document.dispatchEvent('click', { target: outsideNode });
  assert.equal(links.classList.contains('open'), false);
});

test('clicking inside the toggle itself does not trigger the outside-click close path', () => {
  const { toggle, links, document } = loadNav();
  toggle.dispatchEvent('click', {}); // open (via the toggle's own click handler)
  // simulate the bubbled document-level click a real click on the toggle would
  // also produce, targeting the toggle itself
  document.dispatchEvent('click', { target: toggle });
  assert.equal(links.classList.contains('open'), true);
});
