"""
AdverScan theme manager.

Why the old versions only updated on page change
------------------------------------------------
Streamlit's native 3-dots theme switch is handled 100% in the browser. It does
NOT trigger a Python rerun, so any Python-side logic (st.context.theme ->
inject css) only runs on the NEXT rerun, i.e. when you change page.

Approach used here: switch in the browser, no rerun at all
----------------------------------------------------------
* base.css + dark.css + light.css are all injected once (by the script, into a
  single <style id="adverscan-theme-css">). Each theme is scoped under
  <html data-theme="dark|light">, so only one applies at a time.
* A small script watches Streamlit's native theme and sets that attribute
  instantly. To read the native theme without reading our own colours back
  (the old feedback loop), it briefly disables our <style>, reads Streamlit's
  own background colour, and re-enables it, all in one synchronous step, so
  nothing flickers.
* Triggers: class/style change on .stApp (emotion re-themes it), clicks in the
  menu, OS dark-mode change (System option), plus a cheap safety poll.
* Python only supplies the first-paint value (st.context.theme) and an optional
  ?theme=light|dark override.
"""
import json
from pathlib import Path

import streamlit as st

STYLES_DIR = Path(__file__).resolve().parents[1] / "styles"
_STYLE_ID = "adverscan-theme-css"

_JS = r"""
<script>
(function () {
  var W = window.parent || window;          // st.html -> same window; components.html -> parent
  var D = W.document;
  var CSS = __CSS__;
  var INITIAL = "__INITIAL__";
  var FORCE = __FORCE__;                    // true when ?theme= is used
  var STYLE_ID = "__STYLE_ID__";

  var S = W.__adverscanTheme;
  if (S && S.setCss) { S.setCss(CSS); S.refresh(); return; }   // already running: just update css

  var styleEl = D.getElementById(STYLE_ID);
  if (!styleEl) { styleEl = D.createElement("style"); styleEl.id = STYLE_ID; D.head.appendChild(styleEl); }
  styleEl.textContent = CSS;

  var root = D.documentElement;
  if (!root.getAttribute("data-theme")) root.setAttribute("data-theme", INITIAL);   // first paint

  function lum(c) {
    var m = c && c.match(/[\d.]+/g);
    if (!m || m.length < 3) return null;
    if (m.length >= 4 && parseFloat(m[3]) === 0) return null;                        // transparent
    return (parseFloat(m[0]) * 299 + parseFloat(m[1]) * 587 + parseFloat(m[2]) * 114) / 1000;
  }

  // Read STREAMLIT's theme (not ours): disable our sheet, read, re-enable. Synchronous, no paint.
  function detect() {
    var sheet = styleEl.sheet, result = null;
    if (sheet) sheet.disabled = true;
    try {
      var cands = [D.querySelector(".stApp"), D.querySelector('[data-testid="stAppViewContainer"]'), D.body, D.documentElement];
      for (var i = 0; i < cands.length; i++) {
        if (!cands[i]) continue;
        var l = lum(W.getComputedStyle(cands[i]).backgroundColor);
        if (l !== null) { result = l > 140 ? "light" : "dark"; break; }
      }
    } catch (e) { /* ignore */ }
    finally { if (sheet) sheet.disabled = false; }
    return result;
  }

  function apply() {
    var t = FORCE ? INITIAL : detect();
    if (t && root.getAttribute("data-theme") !== t) root.setAttribute("data-theme", t);
  }

  var lastSig = null, app = null, observer = null;
  function signature() {
    var a = D.querySelector(".stApp");
    return a ? (a.className + "|" + (a.getAttribute("style") || "")) : "";
  }
  function check(force) {
    var sig = signature();
    if (force || sig !== lastSig) { lastSig = sig; apply(); }
  }
  function schedule() { [0, 60, 200, 500, 1000].forEach(function (ms) { W.setTimeout(function () { check(true); }, ms); }); }

  function observe() {
    var a = D.querySelector(".stApp");
    if (!a || a === app || !W.MutationObserver) return;
    app = a;
    if (observer) observer.disconnect();
    observer = new W.MutationObserver(function () { check(false); });
    observer.observe(a, { attributes: true, attributeFilter: ["class", "style"] });
  }

  D.addEventListener("click", schedule, true);                       // menu selection (System/Light/Dark)
  if (W.matchMedia) {
    var mq = W.matchMedia("(prefers-color-scheme: dark)");
    (mq.addEventListener ? mq.addEventListener("change", schedule) : mq.addListener(schedule));
  }
  W.setInterval(function () { observe(); check(false); }, 700);       // cheap safety net

  W.__adverscanTheme = {
    setCss: function (c) { styleEl.textContent = c; },
    refresh: function () { observe(); check(true); }
  };
  observe();
  apply();
  schedule();
})();
</script>
"""


def _read(name: str) -> str:
    path = STYLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Theme file not found: {path}")
    return path.read_text(encoding="utf-8")


def _native_theme():
    """'Light'/'Dark' from st.context.theme (accurate at rerun time), else None."""
    ctx = getattr(st, "context", None)
    theme = getattr(ctx, "theme", None) if ctx is not None else None
    if theme is None:
        return None
    try:
        value = theme.get("type") if hasattr(theme, "get") else getattr(theme, "type", None)
    except Exception:
        return None
    return ("Light" if str(value).lower() == "light" else "Dark") if value else None


def _query_override():
    try:
        qp = st.query_params.get("theme")
    except Exception:
        return None
    return ("Light" if str(qp).lower() == "light" else "Dark") if qp else None


def load_theme(active_theme: str = "Dark", force: bool = False) -> None:
    """Install all theme CSS and the browser-side switcher script."""
    initial = "light" if str(active_theme).lower() == "light" else "dark"
    css = "\n".join([_read("base.css"), _read("dark.css"), _read("light.css")])

    script = (
        _JS.replace("__CSS__", json.dumps(css).replace("</", "<\\/"))
        .replace("__INITIAL__", initial)
        .replace("__FORCE__", "true" if force else "false")
        .replace("__STYLE_ID__", _STYLE_ID)
    )

    try:
        st.html(script, unsafe_allow_javascript=True)
    except TypeError:  # older Streamlit: run it in a zero-height iframe instead
        import streamlit.components.v1 as components

        components.html(script, height=0)


def initialize_theme() -> str:
    """Install the theme system; returns the theme as of this rerun."""
    override = _query_override()
    theme = override or _native_theme() or st.session_state.get("theme", "Dark")
    st.session_state["theme"] = theme
    load_theme(theme, force=bool(override))
    return theme