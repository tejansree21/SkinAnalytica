'use strict';
/**
 * SkinAnalytica -- tests_js/test_auth.js
 * Tests assets/auth.js's real logic: the client-held-key model (never
 * embedded in page source, only localStorage) is the actual security
 * property the top-priority auth finding this project fixed relies on --
 * see docs/MODEL_CARD.md finding #13's "closed" update. Loaded via node:vm
 * against the real file, not a reimplementation, so a change to the
 * shipped script is what's actually under test.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

const { FakeElement, makeFakeLocalStorage } = require('./dom_stub');

const AUTH_SRC = fs.readFileSync(path.join(__dirname, '..', 'assets', 'auth.js'), 'utf8');

function loadAuth({ promptReturns = null } = {}) {
  const localStorage = makeFakeLocalStorage();
  const promptCalls = [];
  const context = {
    localStorage,
    prompt: (...args) => { promptCalls.push(args); return promptReturns; },
  };
  vm.createContext(context);
  vm.runInContext(AUTH_SRC, context);
  // auth.js declares `const SA_AUTH = {...}` -- a const/let top-level
  // binding does NOT become a property of the sandbox object the way a
  // `var` would, so it must be read back via a second script in the same
  // context rather than off `context.SA_AUTH` directly.
  const SA_AUTH = vm.runInContext('SA_AUTH', context);
  return { SA_AUTH, localStorage, promptCalls };
}

test('get() returns empty string when nothing stored', () => {
  const { SA_AUTH } = loadAuth();
  assert.equal(SA_AUTH.get(), '');
});

test('set() then get() round-trips a key', () => {
  const { SA_AUTH } = loadAuth();
  SA_AUTH.set('sk-real-key-123');
  assert.equal(SA_AUTH.get(), 'sk-real-key-123');
});

test('set() with an empty string removes the stored key, not stores an empty one', () => {
  const { SA_AUTH, localStorage } = loadAuth();
  SA_AUTH.set('sk-real-key-123');
  SA_AUTH.set('');
  assert.equal(SA_AUTH.get(), '');
  assert.equal(localStorage._store.has(SA_AUTH.KEY_NAME), false);
});

test('headers() includes X-API-Key only when a key is set', () => {
  // headers() objects are created inside the vm sandbox realm, so they have
  // a different Object.prototype than this test file's own realm --
  // deepStrictEqual would fail on that cross-realm prototype mismatch even
  // though the contents are identical. Compare via JSON instead.
  const { SA_AUTH } = loadAuth();
  assert.equal(JSON.stringify(SA_AUTH.headers()), JSON.stringify({}));
  SA_AUTH.set('sk-real-key-123');
  assert.equal(JSON.stringify(SA_AUTH.headers()), JSON.stringify({ 'X-API-Key': 'sk-real-key-123' }));
});

test('get() fails safe (empty string, not a throw) if localStorage.getItem throws', () => {
  const context = {
    localStorage: { getItem() { throw new Error('storage disabled'); } },
    prompt: () => null,
  };
  vm.createContext(context);
  vm.runInContext(AUTH_SRC, context);
  const SA_AUTH = vm.runInContext('SA_AUTH', context);
  assert.equal(SA_AUTH.get(), '');
});

test('mountWidget renders the "add key" prompt state when no key is set', () => {
  const { SA_AUTH } = loadAuth();
  const el = new FakeElement('div');
  SA_AUTH.mountWidget(el);
  assert.match(el.innerHTML, /Add API key/);
  assert.doesNotMatch(el.innerHTML, /sk-real-key/);
});

test('mountWidget masks the stored key to its first 4 characters, never shows the full key', () => {
  const { SA_AUTH } = loadAuth();
  SA_AUTH.set('sk-supersecretvalue');
  const el = new FakeElement('div');
  SA_AUTH.mountWidget(el);
  assert.match(el.innerHTML, /sk-s/);
  assert.doesNotMatch(el.innerHTML, /supersecretvalue/);
});

test('mountWidget: clicking the button with a null prompt result (cancel) leaves the key unchanged', () => {
  const { SA_AUTH, el } = (() => {
    const loaded = loadAuth({ promptReturns: null });
    const el = new FakeElement('div');
    loaded.SA_AUTH.set('sk-original');
    loaded.SA_AUTH.mountWidget(el);
    return { SA_AUTH: loaded.SA_AUTH, el };
  })();
  el.querySelector('button').dispatchEvent('click', {});
  assert.equal(SA_AUTH.get(), 'sk-original');
});

test('mountWidget: submitting a new key via prompt updates storage and re-renders', () => {
  const loaded = loadAuth({ promptReturns: '  sk-new-key  ' });
  const el = new FakeElement('div');
  loaded.SA_AUTH.mountWidget(el);
  el.querySelector('button').dispatchEvent('click', {});
  // trimmed before storing -- a pasted key with surrounding whitespace must not silently fail auth
  assert.equal(loaded.SA_AUTH.get(), 'sk-new-key');
  assert.match(el.innerHTML, /sk-n/);
});

test('mountWidget: submitting an empty prompt result removes the key (matches set() semantics)', () => {
  const loaded = loadAuth({ promptReturns: '' });
  const el = new FakeElement('div');
  loaded.SA_AUTH.set('sk-original');
  loaded.SA_AUTH.mountWidget(el);
  el.querySelector('button').dispatchEvent('click', {});
  assert.equal(loaded.SA_AUTH.get(), '');
  assert.match(el.innerHTML, /Add API key/);
});
