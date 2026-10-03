"""Make the paper figures and derived numbers from results/*.json (no simulation, no training).

python paper/make_figures.py   ->  paper/fig_success.pdf, paper/fig_force.pdf, paper/fig_failures.pdf,
                                   paper/derived_numbers.json

Every number quoted in the paper that is not printed directly in a results file is computed here and written to
derived_numbers.json, so it can be checked against paper/NUMBERS.md.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_results import mcnemar  # noqa: E402  (same exact test as docs/RESULTS.md)

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def load(tag, off):
    return json.loads((ROOT / "results" / f"{tag}_off{off:.1f}.json").read_text())


def succ_map(tag, off):
    return {e["seed"]: e["success"] for e in load(tag, off)["episodes"]}


ROWS = [
    ("expert", "Scripted expert (position)"),
    ("expert_imp", "Scripted expert (impedance)"),
    ("bc", "MLP BC"),
    ("bc_noft", "MLP BC, no F/T"),
    ("bc_imp_imp", "MLP BC (impedance)"),
    ("dp", "Diffusion Policy"),
    ("dp_na2", "DP, re-plan every 2"),
    ("dp_noft", "DP, no F/T"),
    ("act", "ACT"),
    ("act_na1", "ACT, re-plan every 1"),
]


def fig_success():
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    y = np.arange(len(ROWS))[::-1]
    h = 0.36
    for off, col, dy, lab in [
        (0.0, BLUE, h / 2 + 0.02, "nominal (0.5 mm noise)"),
        (2.0, ORANGE, -h / 2 - 0.02, "fixture perceived 2 mm off"),
    ]:
        rates, lo, hi = [], [], []
        for tag, _ in ROWS:
            s = load(tag, off)["summary"]
            rates.append(100 * s["success_rate"])
            lo.append(100 * (s["success_rate"] - s["ci95"][0]))
            hi.append(100 * (s["ci95"][1] - s["success_rate"]))
        ax.barh(y + dy, rates, height=h, color=col, label=lab, edgecolor="white", linewidth=0.6)
        ax.errorbar(rates, y + dy, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
    ax.set_yticks(y, [n for _, n in ROWS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Success rate (%), 200 unseen seeds")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, frameon=False, fontsize=6.0, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_success.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_force():
    """Peak part-fixture contact force per episode (ECDF), 2 mm perception error."""
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for tag, lab, col, ls in [
        ("expert", "Expert, position control", ORANGE, "-"),
        ("expert_imp", "Expert, impedance control", BLUE, "-"),
        ("bc", "MLP BC (position)", AQUA, "--"),
        ("act", "ACT (position)", GREY, ":"),
    ]:
        f = np.sort([e["peak_force"] for e in load(tag, 2.0)["episodes"]])
        ax.step(f, np.arange(1, len(f) + 1) / len(f), where="post", color=col, linestyle=ls, linewidth=1.4, label=lab)
    ax.axvline(50, color=MUTED, linewidth=0.6, label="expert wrist-force limit (50 N)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Peak part-fixture contact force in an episode (N)")
    ax.set_ylabel("Fraction of episodes")
    ax.grid(color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", frameon=True, fontsize=6, framealpha=0.95, edgecolor="none")
    leg.get_frame().set_facecolor("white")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_force.pdf")
    plt.close(fig)


def failure_split(tag, off):
    eps = load(tag, off)["episodes"]
    fails = [e for e in eps if not e["success"]]
    seated = sum(e.get("seated", False) for e in fails)
    no_contact = sum((not e.get("seated", False)) and e["peak_force"] < 5.0 for e in fails)
    return dict(
        failures=len(fails), seated_not_released=seated, never_above_5N=no_contact, other=len(fails) - seated - no_contact
    )


def fig_failures(splits):
    keys = [("bc", 0.0), ("bc_noft", 0.0), ("bc", 2.0), ("bc_noft", 2.0)]
    labels = ["BC\nnominal", "BC no F/T\nnominal", "BC\n2 mm off", "BC no F/T\n2 mm off"]
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for field, col, lab in [
        ("seated_not_released", BLUE, "part seated, gripper did not release"),
        ("never_above_5N", ORANGE, "contact force never above 5 N"),
        ("other", GREY, "other (contact, not seated)"),
    ]:
        v = np.array([splits[f"{t}_off{o:.1f}"][field] for t, o in keys])
        ax.bar(x, v, bottom=bottom, color=col, width=0.55, label=lab, edgecolor="white", linewidth=0.8)
        bottom += v
    for xi, b in zip(x, bottom, strict=True):
        ax.text(xi, b + 2, f"{int(b)}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Failed episodes (of 200)")
    ax.set_ylim(0, 175)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_failures.pdf")
    plt.close(fig)


def fig_scene():
    """Two frames of media/expert_seed100003_off2mm.mp4 (front and close camera side by side, as rendered)."""
    import imageio.v2 as imageio

    rd = imageio.get_reader(ROOT / "media" / "expert_seed100003_off2mm.mp4")
    frames = {i: f for i, f in enumerate(rd) if i in (160, 230)}
    rd.close()
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6))
    for ax, i, lab in zip(
        axes, (160, 230), ("(a) step 160: part above the fixture", "(b) step 230: part seated, gripper open"), strict=True
    ):
        ax.imshow(frames[i])
        ax.set_axis_off()
        ax.set_title(lab, fontsize=6.5, color=INK, pad=2)
    fig.tight_layout(pad=0.2, h_pad=0.4)
    fig.savefig(OUT / "fig_scene.png", dpi=300)
    plt.close(fig)


def main():
    derived = {}
    splits = {}
    for tag in ["bc", "bc_noft", "bc_imp_imp", "dp", "dp_noft", "dp_na2", "act", "act_na1"]:
        for off in (0.0, 2.0):
            splits[f"{tag}_off{off:.1f}"] = failure_split(tag, off)
    derived["failure_split"] = splits
    mc = {}
    for a, b in [
        ("bc", "bc_noft"),
        ("dp", "dp_noft"),
        ("expert_imp", "expert"),
        ("bc_imp_imp", "bc"),
        ("bc", "dp"),
        ("dp_na2", "dp"),
        ("act_na1", "act"),
    ]:
        for off in (0.0, 2.0):
            n01, n10, p = mcnemar(succ_map(a, off), succ_map(b, off))
            mc[f"{a}_vs_{b}_off{off:.1f}"] = dict(only_first=n01, only_second=n10, p=p)
    derived["mcnemar_exact"] = mc
    act_f = [e["err_xy_mm"] for e in load("act", 0.0)["episodes"] if not e["success"]]
    derived["act_off0.0_failed_err_xy_mm_quartiles"] = [float(np.percentile(act_f, q)) for q in (25, 50, 75)]
    derived["act_off0.0_failed_touching_fixture"] = int(
        sum(e["peak_force"] >= 5.0 for e in load("act", 0.0)["episodes"] if not e["success"])
    )
    for tag in ["expert", "expert_imp", "bc", "act"]:
        derived[f"{tag}_off2.0_peak_force_median_N"] = load(tag, 2.0)["summary"]["peak_force_median_N"]
    for tag in ["expert_off2.0", "expert_imp_off2.0", "expert_off0.0", "expert_imp_off0.0"]:
        eps = json.loads((ROOT / "results" / f"{tag}.json").read_text())["episodes"]
        phases = {}
        for e in eps:
            if not e["success"]:
                phases[e["final_phase"]] = phases.get(e["final_phase"], 0) + 1
        derived[f"{tag}_failed_final_phase"] = phases
        derived[f"{tag}_episodes_with_wall_contact_in_insert"] = int(sum(e["retries"] > 0 for e in eps))
    (OUT / "derived_numbers.json").write_text(json.dumps(derived, indent=2))
    fig_success()
    fig_force()
    fig_failures(splits)
    fig_scene()
    print(json.dumps(derived, indent=2))


if __name__ == "__main__":
    main()
