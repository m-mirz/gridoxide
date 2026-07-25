#!/usr/bin/env python3
"""Check a CGMES fixture's published `SvVoltage` against its own EQ/SSH data —
no solver, no importer, no second tool involved.

Section 6's accuracy tables report each tool's deviation from the fixture's
published SV values, treating those as the reference. That is only meaningful
where the reference is itself consistent with the network the same fixture
describes, and on real-world-derived fixtures it sometimes isn't. This script
tests that precondition directly.

The test: for every two-winding `PowerTransformer` whose ends both have a
published `SvVoltage`, compare

    published ratio  = v(end1) / v(end2)
    nameplate ratio  = ratedU(end1) / ratedU(end2), adjusted by the actual
                       `RatioTapChanger.step` in the SSH profile

Those two can differ legitimately, but only by the voltage drop across the
transformer's own series impedance — percent, not tens of percent. A published
ratio that misses the tap-adjusted nameplate ratio by 70%+ cannot be produced
by any tap position or load level; it means the SV profile and the EQ profile
disagree about what the network is, and no importer can reconcile them.

Buses flagged here should be excluded from, or annotated in, any
accuracy-vs-published-SV metric: a tool is being penalized there for matching
the equipment data rather than the reference solution.

Usage:
    python3 check_cgmes_sv_consistency.py [fixture ...]     (default: RealGrid Svedala)
    python3 check_cgmes_sv_consistency.py RealGrid --top 20
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CGMES_DIR = REPO_ROOT / "tests" / "data" / "CGMES-Test-Configurations" / "v3.0"


def _tag(body, tag):
    m = re.search(rf"<cim:{tag}>([^<]*)<", body)
    return m.group(1) if m else None


def _ref(body, tag):
    m = re.search(rf'{tag} rdf:resource="#([^"]+)"', body)
    return m.group(1) if m else None


def load_fixture(fixture):
    base = CGMES_DIR / fixture / f"{fixture}-Merged"
    if not base.exists():
        return None
    texts = {}
    for profile in ("EQ", "SSH", "TP", "SV"):
        p = base / f"{fixture}_{profile}.xml"
        texts[profile] = p.read_text() if p.exists() else ""
    return texts


def analyze(fixture, top):
    t = load_fixture(fixture)
    if t is None:
        print(f"{fixture}: fixture directory not found — skipping")
        return 0

    published = {}
    for m in re.finditer(r"<cim:SvVoltage[^>]*>(.*?)</cim:SvVoltage>", t["SV"], re.S):
        tn = _ref(m.group(1), "SvVoltage.TopologicalNode")
        v = _tag(m.group(1), "SvVoltage.v")
        if tn and v:
            published[tn] = float(v)

    term_tn = {}
    for m in re.finditer(r'<cim:Terminal rdf:about="#([^"]+)">(.*?)</cim:Terminal>', t["TP"], re.S):
        tn = _ref(m.group(2), "Terminal.TopologicalNode")
        if tn:
            term_tn[m.group(1)] = tn

    nominal = {}
    for m in re.finditer(r'<cim:BaseVoltage rdf:ID="([^"]+)">(.*?)</cim:BaseVoltage>', t["EQ"], re.S):
        v = _tag(m.group(2), "BaseVoltage.nominalVoltage")
        if v:
            nominal[m.group(1)] = float(v)
    tn_base = {}
    for m in re.finditer(r'<cim:TopologicalNode rdf:ID="([^"]+)">(.*?)</cim:TopologicalNode>', t["TP"], re.S):
        b = _ref(m.group(2), "TopologicalNode.BaseVoltage")
        if b:
            tn_base[m.group(1)] = nominal.get(b)

    # RatioTapChanger, keyed by the TransformerEnd it is attached to.
    steps = {}
    for m in re.finditer(r'<cim:RatioTapChanger rdf:about="#([^"]+)">(.*?)</cim:RatioTapChanger>', t["SSH"], re.S):
        s = _tag(m.group(2), "TapChanger.step")
        if s:
            steps[m.group(1)] = float(s)
    tap_by_end = {}
    for m in re.finditer(r'<cim:RatioTapChanger rdf:ID="([^"]+)">(.*?)</cim:RatioTapChanger>', t["EQ"], re.S):
        body = m.group(2)
        end = _ref(body, "RatioTapChanger.TransformerEnd")
        inc = _tag(body, "RatioTapChanger.stepVoltageIncrement")
        neutral = _tag(body, "TapChanger.neutralStep")
        if end and inc and neutral:
            step = steps.get(m.group(1))
            if step is not None:
                tap_by_end[end] = 1.0 + (step - float(neutral)) * float(inc) / 100.0

    ends = {}
    for m in re.finditer(r'<cim:PowerTransformerEnd rdf:ID="([^"]+)">(.*?)</cim:PowerTransformerEnd>', t["EQ"], re.S):
        body = m.group(2)
        pt = _ref(body, "PowerTransformerEnd.PowerTransformer")
        ru = _tag(body, "PowerTransformerEnd.ratedU")
        term = _ref(body, "TransformerEnd.Terminal")
        num = _tag(body, "TransformerEnd.endNumber")
        if pt and ru and term:
            ends.setdefault(pt, []).append((int(num or 0), float(ru), term, m.group(1)))

    findings = []
    checked = 0
    for pt, es in ends.items():
        if len(es) != 2:
            continue
        es.sort()
        (_, ru1, tm1, id1), (_, ru2, tm2, id2) = es
        n1, n2 = term_tn.get(tm1), term_tn.get(tm2)
        if n1 not in published or n2 not in published or ru2 == 0:
            continue
        v1, v2 = published[n1], published[n2]
        if v1 <= 0 or v2 <= 0:
            continue
        checked += 1
        # Tap on either end scales that end's effective rated voltage.
        declared = (ru1 * tap_by_end.get(id1, 1.0)) / (ru2 * tap_by_end.get(id2, 1.0))
        off = abs(v1 / v2 - declared) / declared
        findings.append((off, ru1, ru2, v1, v2, declared, n1, n2))

    findings.sort(reverse=True)
    over = [f for f in findings if f[0] > 0.05]
    print(f"\n=== {fixture} ===")
    print(f"  {len(published)} published SvVoltage values, {checked} two-winding transformers with both ends published")
    print(f"  transformers whose published ratio misses tap-adjusted nameplate by >5%: {len(over)}")
    if not findings:
        return 0
    print(f"\n  {'off':>9} {'ratedU1':>8} {'ratedU2':>8} {'pub v1':>10} {'pub v2':>10} {'nameplate':>10} {'published':>10}")
    for off, ru1, ru2, v1, v2, declared, n1, n2 in findings[:top]:
        print(f"  {off:>9.2%} {ru1:>8.1f} {ru2:>8.1f} {v1:>10.4f} {v2:>10.4f} {declared:>10.4f} {v1 / v2:>10.4f}")
        print(f"            end1 TN {n1} (base {tn_base.get(n1)} kV)")
        print(f"            end2 TN {n2} (base {tn_base.get(n2)} kV)")
    return len(over)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fixtures", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=8, help="how many worst transformers to list (default 8)")
    args = ap.parse_args()
    fixtures = args.fixtures or ["RealGrid", "Svedala"]
    total = sum(analyze(f, args.top) for f in fixtures)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
