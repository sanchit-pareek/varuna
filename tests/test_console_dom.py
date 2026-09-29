"""Static checks on the console's DOM contract.

The JavaScript looks elements up by id and writes into them. Nothing enforces
that those ids still exist by the time it does, and the failure mode is silent
until a user clicks the button: `resetRun()` cleared `verdict-card.innerHTML`,
which deleted the two elements nested inside it that the renderer needs, and
every subsequent run died with "Cannot set properties of null".
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATIC = ROOT / "app" / "static"


class IdNesting(HTMLParser):
    """Record, for every id in the document, the ids that enclose it."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.ancestors = {}
        self.void = {"br", "hr", "img", "input", "meta", "link", "source"}

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        node = d.get("id")
        if node:
            self.ancestors[node] = list(self.stack)
        if tag not in self.void:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id"):
            self.ancestors[d["id"]] = list(self.stack)

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()


def _parse():
    p = IdNesting()
    p.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return p.ancestors


def _js():
    return (STATIC / "app.js").read_text(encoding="utf-8")


def test_every_id_the_script_looks_up_exists_in_the_markup():
    """`$("thing")` must resolve, or the write into it throws at click time."""
    ancestors = _parse()
    known = set(ancestors)

    # Ids the script creates at runtime rather than finding in the markup.
    created = {m for m in re.findall(r'\.id\s*=\s*"([a-zA-Z0-9_-]+)"', _js())}

    referenced = set(re.findall(r'\$\(\s*"([a-zA-Z0-9_-]+)"\s*\)', _js()))
    referenced |= set(re.findall(r'getElementById\(\s*"([a-zA-Z0-9_-]+)"\s*\)', _js()))

    missing = sorted(referenced - known - created)
    assert not missing, "app.js looks up ids that are not in index.html: %s" % missing


def test_clearing_a_container_never_deletes_an_id_the_script_needs():
    """The exact bug: `verdict-card` encloses `verdict-time` and `verdict`.

    Wiping the parent's innerHTML removed both, and the renderer then wrote into
    null. Anything the script empties must not enclose another id it uses.
    """
    ancestors = _parse()
    js = _js()

    cleared = set(re.findall(
        r'\$\(\s*"([a-zA-Z0-9_-]+)"\s*\)\s*\.innerHTML\s*=\s*""', js))
    for var_pat in (r'var\s+(\w+)\s*=\s*\$\(\s*"([a-zA-Z0-9_-]+)"\s*\)',):
        for var, node in re.findall(var_pat, js):
            if re.search(re.escape(var) + r'\.innerHTML\s*=\s*""', js):
                cleared.add(node)

    referenced = set(re.findall(r'\$\(\s*"([a-zA-Z0-9_-]+)"\s*\)', js))

    offenders = []
    for node, chain in ancestors.items():
        if node not in referenced:
            continue
        for parent in chain:
            if parent and parent in cleared:
                offenders.append((parent, node))

    assert not offenders, (
        "these containers are emptied but enclose an id the script writes into, "
        "so the write will hit null after a reset: %s" % sorted(set(offenders)))


def test_the_reset_restores_every_panel_the_run_fills():
    """Switching scenes must not leave one panel stale and another cleared."""
    js = _js()
    assert "function resetRun()" in js, "no resetRun; scene switching will keep stale state"
    for panel in ("detection", "origin", "suspects", "trace"):
        assert re.search(r'\["%s"' % panel, js), (
            "resetRun does not restore the %s panel" % panel)
    # The finding lives in the right rail now and always occupies its block, so
    # the reset restores its empty state instead of hiding the card.
    assert 'v.appendChild(el("p", "hint", "No run yet."))' in js, (
        "resetRun does not restore the finding panel's empty state")
    assert "renderCase(null)" in js, (
        "resetRun does not clear the case header, run summary and bottom strip")


def test_asset_urls_change_when_the_assets_do(tmp_path, monkeypatch):
    """Stamping the version was not enough to bust a browser cache.

    `app.js?v=1.5.0` stays byte-identical as a URL no matter how much app.js
    changes, so a browser that already had it served the old script against a
    freshly rendered DOM. Silent, and exactly the failure the stamp existed to
    prevent. The stamp is now a hash of the files themselves.
    """
    from app import config, main

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<script src='/static/app.js?v=__V__'>", encoding="utf-8")
    (static / "app.js").write_text("var a = 1;", encoding="utf-8")
    (static / "style.css").write_text("body{}", encoding="utf-8")
    monkeypatch.setattr(config, "STATIC_DIR", static)

    before = main._asset_stamp()
    assert before in main.index().body.decode()

    (static / "app.js").write_text("var a = 2;", encoding="utf-8")
    after = main._asset_stamp()
    assert after != before, "editing app.js did not change the asset URL"
    assert after in main.index().body.decode()

    # And a stamp is stable when nothing changes, or every reload re-downloads.
    assert main._asset_stamp() == after


def test_the_root_route_serves_the_console_not_a_helper():
    """A decorator left attached to the wrong function served the shell as a
    bare cache key: the page rendered the twelve-character hash and nothing
    else. Cheap to assert, and invisible until you open the browser."""
    from app.main import app

    root = [r for r in app.routes if getattr(r, "path", "") == "/"]
    assert root, "no route serves /"
    assert root[0].endpoint.__name__ == "index", (
        "/ is served by %r, not the console shell" % root[0].endpoint.__name__)

    body = root[0].endpoint().body.decode("utf-8")
    assert "<!doctype html>" in body.lower()
    assert "__V__" not in body, "the asset stamp was not substituted"
    assert 'id="map"' in body


def test_the_spread_chart_plots_the_job_document_not_a_re_derivation():
    """A chart that disagrees with the number printed beside it is worse than
    no chart. The first version re-derived the ensemble radius in the browser
    from the envelope ring geometry, which landed 9 percent away from the
    origin's own reported spread_km. It now reads hindcast_hourly directly."""
    js = _js()
    assert "d.hindcast_hourly" in js, "the chart does not read the hourly series"
    assert "ringRadiusKm" not in js, "the client-side radius re-derivation is still present"
    # The marker must be placed on the plotted point, not at an arbitrary height.
    assert "pts[Math.round(oh)].r" in js, "the origin marker is not tied to the plotted value"
