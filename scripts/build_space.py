"""Build the static demo page (site/) from results/*.json and media/*.mp4. No number is typed by hand."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
NAMES = {
    "expert": "Scripted expert (reference)",
    "bc": "MLP behavior cloning",
    "bc_noft": "MLP behavior cloning, no wrist F/T",
    "act": "ACT (LeRobot)",
    "dp": "Diffusion Policy (LeRobot)",
    "dp_noft": "Diffusion Policy, no wrist F/T",
}
VIDEOS = [
    (
        "bc_seed100003_off2mm.mp4",
        "MLP behavior cloning with wrist F/T, fixture perceived 2 mm off: touches the chamfer, corrects, seats the part.",
    ),
    (
        "bc_noft_seed100003_off2mm.mp4",
        "Same seed, same policy class without F/T: it hovers above the pocket and never commits to the insertion.",
    ),
    ("expert_seed100003_off2mm.mp4", "Scripted demonstrator on the same seed (force-guided search)."),
]


def s(policy, off):
    p = ROOT / "results" / f"{policy}_off{off:.1f}.json"
    return json.loads(p.read_text())["summary"] if p.exists() else None


def cell(x):
    if not x:
        return "<td>-</td>"
    return (
        f"<td><b>{x['success']}/{x['n']}</b> <span class=ci>({100 * x['ci95'][0]:.0f}-{100 * x['ci95'][1]:.0f}%)</span></td>"
    )


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "media").mkdir(parents=True)
    rows = []
    for k in ["expert", "bc", "bc_noft", "act", "dp", "dp_noft"]:
        a, b = s(k, 0.0), s(k, 2.0)
        if a or b:
            rows.append(f"<tr><td>{NAMES[k]}</td>{cell(a)}{cell(b)}</tr>")
    vids = []
    for f, cap in VIDEOS:
        src = ROOT / "media" / f
        if src.exists():
            shutil.copy(src, SITE / "media" / f)
            vids.append(
                f'<figure><video src="media/{f}" controls muted loop playsinline></video><figcaption>{cap}</figcaption></figure>'
            )
    demo = json.loads((ROOT / "data" / "demos_meta.json").read_text())
    results_md = (ROOT / "docs" / "RESULTS.md").read_text()
    mc = [
        ln
        for ln in results_md.splitlines()
        if ln.startswith("| MLP behavior cloning vs") or ln.startswith("| Diffusion Policy (LeRobot) vs")
    ]
    mc_rows = "".join("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in ln.strip("|").split("|")) + "</tr>" for ln in mc)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>assembly-lab</title><style>
:root{{--bg:#f7f8fa;--fg:#1c2330;--mut:#5b6474;--card:#fff;--line:#dfe3ea;--acc:#2a5db0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e6e9ef;--mut:#9aa3b2;--card:#1b2029;--line:#2c3340;--acc:#7aa7ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}}main{{max-width:980px;margin:auto;padding:24px 16px}}
h1{{margin:0 0 4px}}p.lead{{color:var(--mut);margin-top:0}}table{{width:100%;border-collapse:collapse;background:var(--card);margin:12px 0 24px;font-size:15px}}
td,th{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left}}.ci{{color:var(--mut);font-size:13px}}
.grid{{display:grid;grid-template-columns:1fr;gap:18px}}video{{width:100%;border-radius:8px;background:#000}}figcaption{{color:var(--mut);font-size:14px}}
a{{color:var(--acc)}}.note{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.wrap{{overflow-x:auto}}</style></head><body><main>
<h1>assembly-lab</h1>
<p class=lead>Learned insertion of an ECU-sized part into a locating fixture on a simulated Franka Emika Panda (MuJoCo), with a wrist force-torque sensor.
1 mm clearance per side, 2 mm lead-in chamfer. Code: <a href="https://github.com/pavanyadava007/assembly-lab">github.com/pavanyadava007/assembly-lab</a></p>
<p class=note>Simulation only, no real robot. Policies are trained on {demo["episodes_kept"]} scripted demonstrations ({demo["frames"]:,} frames), not human teleoperation.
Every number below is read from the evaluation files: 200 unseen seeds per cell, Wilson 95% intervals.</p>
<h2>Insertion success</h2><div class=wrap><table><tr><th>Policy</th><th>Nominal perception</th><th>Fixture perceived 2 mm off</th></tr>{"".join(rows)}</table></div>
<h2>Does wrist force-torque sensing matter?</h2><p>Paired on the same seeds, exact McNemar test.</p>
<div class=wrap><table><tr><th>Comparison</th><th>Condition</th><th>Only with F/T succeeded</th><th>Only without F/T succeeded</th><th>p</th></tr>{mc_rows}</table></div>
<h2>Same scene, three controllers</h2><div class=grid>{"".join(vids)}</div>
</main></body></html>"""
    (SITE / "index.html").write_text(html)
    (SITE / "README.md").write_text(
        "---\ntitle: assembly-lab\nemoji: 🔧\ncolorFrom: blue\ncolorTo: gray\nsdk: static\npinned: false\nlicense: mit\nshort_description: Learned contact-rich insertion on a simulated Franka\n---\n"
    )
    print("site built:", [p.name for p in SITE.rglob("*") if p.is_file()])


if __name__ == "__main__":
    main()
