'use strict';
/**
 * SkinAnalytica -- tests_js/dom_stub.js
 * A minimal, hand-rolled DOM/localStorage stub -- deliberately NOT jsdom.
 * assets/auth.js and assets/nav.js are plain browser scripts (no module
 * exports, no build step), and the surface they actually touch (a handful
 * of element methods, localStorage, prompt) is small enough that a real
 * DOM implementation would be a new dependency for no real benefit. Keeps
 * this project's existing "don't add a dependency the problem doesn't
 * need" discipline (see src/inference_utils.py's docstring, or the
 * config/skin_config.yaml comment on why YAML isn't loaded at runtime).
 */

class FakeEventTarget {
  constructor() {
    this._listeners = {};
  }
  addEventListener(type, fn) {
    (this._listeners[type] ||= []).push(fn);
  }
  dispatchEvent(type, evt) {
    for (const fn of this._listeners[type] || []) fn(evt);
  }
}

class FakeElement extends FakeEventTarget {
  constructor(tagName = 'div') {
    super();
    this.tagName = tagName.toUpperCase();
    this._attrs = {};
    this._classes = new Set();
    this.children = [];
    this.classList = {
      add: (c) => this._classes.add(c),
      remove: (c) => this._classes.delete(c),
      contains: (c) => this._classes.has(c),
    };
  }
  setAttribute(name, value) { this._attrs[name] = String(value); }
  getAttribute(name) { return this._attrs[name] ?? null; }
  contains(node) { return node === this || this.children.includes(node); }
  set innerHTML(html) {
    this._innerHTML = html;
    // auth.js only ever needs to find the single <button> it just wrote --
    // no general parsing, just enough to make querySelector('button') work.
    this._button = /<button/.test(html) ? new FakeElement('button') : null;
  }
  get innerHTML() { return this._innerHTML || ''; }
  querySelector(sel) {
    if (sel === 'button') return this._button;
    return null;
  }
}

function makeFakeLocalStorage() {
  const store = new Map();
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    _store: store,
  };
}

function makeFakeDocument(elementsById) {
  const doc = new FakeEventTarget();
  doc.getElementById = (id) => elementsById[id] || null;
  return doc;
}

module.exports = { FakeEventTarget, FakeElement, makeFakeLocalStorage, makeFakeDocument };
