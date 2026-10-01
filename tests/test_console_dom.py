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


def test_no_id_appears_twice_in_the_markup():
    """`document.getElementById` returns the first match and ignores the rest,
    so a duplicated id does not fail loudly. It just quietly makes one of the
    two elements unreachable: the panel below it goes stale and the operator
    reads a number from the previous case without any indication of it."""
    import collections

    ids = re.findall(r'id="([^"]+)"', (STATIC / "index.html").read_text(encoding="utf-8"))
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    assert not dupes, "these ids appear more than once in index.html: %s" % sorted(dupes)


def test_every_mode_has_a_nav_tab_and_a_panel():
    """The nav tab and its panel are wired together by a data attribute, not by
    any check, so a renamed mode produces a tab that switches to nothing and no
    error anywhere. Both halves are pinned here.

    The set itself is not pinned to a fixed list. What matters is that the two
    sides agree, because that disagreement is the silent failure; adding a mode
    is a design decision and renaming one is a migration, neither of which this
    test should be able to block."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    tabs = re.findall(r'class="navlink"[^>]*data-view="([a-z]+)"', html)
    panels = re.findall(r'class="view" data-view="([a-z]+)"', html)
    assert tabs, "no nav tabs found in index.html"
    assert sorted(tabs) == sorted(panels), (
        "the nav and the panels disagree: %s" % sorted(set(tabs) ^ set(panels)))
    assert len(set(tabs)) == len(tabs), "a mode has more than one tab: %s" % tabs


def test_the_console_opens_on_a_mode_that_actually_exists():
    """initNav used to select a view name that no longer existed, which hid
    every panel at once: an empty rail with no visible reason. The default is
    read out of a named constant rather than a literal so the two cannot drift."""
    js = _js()
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    tabs = re.findall(r'class="navlink"[^>]*data-view="([a-z]+)"', html)
    declared = re.search(r'var DEFAULT_VIEW\s*=\s*"([a-z]+)"', js)
    assert declared, "the default view is no longer declared as DEFAULT_VIEW"
    assert declared.group(1) in tabs, (
        "DEFAULT_VIEW is %r, which is not one of the tabs %s"
        % (declared.group(1), tabs))
    assert "select(DEFAULT_VIEW)" in js, (
        "initNav no longer opens on DEFAULT_VIEW")


def test_a_deep_link_naming_a_mode_that_no_longer_exists_still_shows_something():
    """Case URLs are shared. A rename must not turn an old link into a blank
    screen, so the name is validated against the tabs that exist, and the
    fallback is announced rather than applied silently."""
    js = _js()
    assert "known.indexOf(requested) < 0" in js, (
        "an unknown #view= name is not guarded, so a stale link hides every panel")
    assert 'el("div", "notice"' in js, (
        "the fallback is applied silently, so a stale link looks like a view "
        "the analyst chose")


def test_the_time_scrubber_is_re_enabled_when_a_new_run_supplies_frames():
    """resetRun disables the slider, and the first version never re-enabled it.
    The first run of a session worked; every run after picking a different
    scene left the scrubber greyed out and inert, and nothing said why."""
    js = _js()
    assert 'slider.disabled = true' in js, "resetRun no longer clears the scrubber"
    assert "slider.disabled = state.frames.length < 2" in js, (
        "buildFrames does not restore the scrubber's enabled state")


def test_probe_mode_is_visible_on_the_map_and_not_only_in_a_checkbox():
    """The script has always set `.probing` on the map element. With no rule
    for it the class did nothing, so the operator had a checkbox and no way to
    see that the next click would do something different."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "#map.probing" in css, "arming the probe changes nothing on screen"
    js = _js()
    assert 'classList.toggle("probing"' in js, "the probe class is never toggled"


def test_the_finding_grid_is_actually_styled():
    """The markup writes `verdict-facts` and the rule was written against
    `.facts`, so the one element that has to read as a form rendered as
    unstyled stacked text. Both names are now accepted."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert ".verdict-facts" in css, "the finding grid has no styling"
    assert '"verdict-facts"' in _js(), "the renderer no longer writes the class"


def test_the_header_carries_the_case_quality_band_and_it_is_cleared_on_reset():
    """The band changes how every other number should be read, so it belongs in
    the header where it is visible in every mode. A reset that leaves the last
    case's band up is worse than one that never drew it."""
    js = _js()
    assert '$("case-q")' in js, "the header has no quality band"
    assert "EVIDENCE " in js, "the band is not labelled in words"


def test_counter_evidence_is_rendered_next_to_the_candidate_it_opposes():
    """The objection is the part that stops a lead being read as a finding, so
    it needs a fixed place rather than a row in a leaderboard that gets
    skimmed for the supporting reasons."""
    js = _js()
    assert "Why this vessel" in js
    assert "Why not" in js
    assert "why_not" in js
    # And the absence of a recorded objection must read as absence, not as
    # evidence that none exists.
    assert "not evidence that none exists" in js


def test_a_case_with_no_vessel_candidate_does_not_render_a_lead():
    """The pipeline deliberately reports nothing rather than forcing a culprit.
    The console has to honour that, so the empty state is a finding and not a
    blank panel."""
    js = _js()
    assert "No vessel passed the spatio-temporal filter" in js
    assert "no reliable" in js.lower() or "rather than forcing" in js


def test_unevaluable_scenarios_are_shown_as_such_rather_than_omitted():
    """An absent row reads as 'this did not matter', which is the opposite of
    what a missing input means. Both the per-candidate scenario list and the
    ablation ladder have to carry a NOT EVALUATED slot, and an inapplicable
    scenario has to give its reason rather than just its absence."""
    js = _js()
    assert "NOT EVALUATED" in js
    assert "this scenario could not be evaluated" in js
    assert "this rung was not computed" in js


def test_the_uncalibrated_score_is_said_out_loud_where_the_score_is_shown():
    """Every surface that ranks vessels has to carry the caveat, because a
    score lifted out of context into a slide deck keeps the number and loses
    the sentence."""
    js = _js()
    assert "uncalibrated" in js.lower()
    assert "not a probability" in js.lower() or "not a finding of discharge" in js.lower()


def test_no_renderer_invents_a_field_the_pipeline_does_not_write():
    """The renderers were written against an assumed shape and three of the
    assumptions were wrong, so the panels silently rendered empty: the
    uncertainty chain read `st.uncertainty` where the writer emits
    `st.quantity`, sensitivity read one flat row per scenario where the writer
    emits one record per candidate with nested `scenarios`, and the ablation
    read `study.rungs` where the writer emits `study.ladder`.

    Nothing throws in any of those cases. A missing field reads as absent data,
    which is the one failure mode this project cannot afford, because "not
    reported" and "not there" look identical on the page.

    So the field names are pinned against the writers."""
    js = _js()
    stale = {
        "st.uncertainty": "the chain writes st.quantity",
        "st.uncertainty_level": "the chain writes st.level",
        "study.rungs": "ablation writes study.ladder",
        "r.spearman_with_full": "ablation writes r.spearman_vs_full",
        "r.top_1_unchanged": "ablation writes agreement.top1_match",
        "r.available === false": "ablation writes r.status",
    }
    for wrong, right in stale.items():
        assert wrong not in js, "app.js still reads %s, but %s" % (wrong, right)

    for real in ("st.quantity", "st.level", "study.ladder",
                 "r.spearman_vs_full", "r.agreement", "top1_match", "r.status"):
        assert real in js, "app.js does not read %s" % real

    # Sensitivity is per candidate, so it must read the candidate fields and
    # walk the nested scenarios rather than assuming a flat scenario list.
    assert "r.baseline_rank" in js, "sensitivity does not read the candidate baseline"
    assert "r.stability" in js, "sensitivity does not read the stability label"
    assert "r.scenarios" in js, "sensitivity does not walk the nested scenarios"
    # `ordered_factors` is only a list of stage keys; the level lives in
    # `factors`, so the panel has to resolve the two together.
    assert "factors[key]" in js, (
        "the safe-fail panel reads ordered_factors without resolving factors, "
        "so every row renders with no level beside it")

    # The claims each get a renderer. Falling back to a raw JSON dump for all of
    # them is the same silent failure wearing a different hat: the data is
    # technically on the page, but nothing is claimed, so an analyst reading it
    # draws their own conclusion about what a missing field means.
    for renderer in ("renderCandidates", "renderUncertainty", "renderSensitivity",
                     "renderAblation", "renderCalibration"):
        assert renderer in js, "app.js no longer renders %s" % renderer
    assert "prettyResult(part)" in js, (
        "the case panel has gone back to dumping raw JSON for the claims")


def test_the_uncertainty_bar_is_certainty_and_is_labelled_as_such():
    """`quantity` runs from 0.0 for 'this stage could not be run' to 1.0 for
    'this stage's inputs are in good order'. Drawn as an uncertainty it would
    fill up on a well-run stage and empty out on a stage that never executed,
    which is exactly backwards, and the colour followed it: the first version
    painted a HIGH stage red."""
    js = _js()
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "stage certainty" in js.lower(), (
        "the bar has no label saying which direction it runs")
    assert "not a probability" in js.lower() or "not a" in js.lower()
    # The good end of the scale must be the good colour. Matched to the closing
    # brace: the custom property reference is `var(--cyan)`, and a character
    # class stops at the parenthesis.
    high = re.search(r"\.ustage\.high\s*\{[^}]*border-left-color:\s*([^;}]+)", css)
    low = re.search(r"\.ustage\.low\s*\{[^}]*border-left-color:\s*([^;}]+)", css)
    assert high and low, "the chain rows have no per-level colouring"
    assert "var(--cyan)" in high.group(1), (
        "a HIGH certainty stage is painted %s; certainty runs the other way"
        % high.group(1))
    assert "var(--bad)" in low.group(1), (
        "a LOW certainty stage is painted %s" % low.group(1))

