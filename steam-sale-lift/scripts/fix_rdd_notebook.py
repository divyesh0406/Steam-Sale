"""Patch cell cd03cb85b5 in 03_hypothesis_2_rdd.ipynb to fix format string bug."""
import json
from pathlib import Path

nb_path = Path(__file__).parent.parent / "notebooks" / "03_hypothesis_2_rdd.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

fixed_source = [
    'placebos = placebo_cutoffs(rdd, outcome="log_reviews")\n',
    'print("=== Placebo cutoffs (should show null effects) ===")\n',
    'placebo_cutoff_vals = [20, 30, 40, 60, 70, 80]\n',
    'for c, r in zip(placebo_cutoff_vals, placebos):\n',
    '    if "error" not in r:\n',
    '        sig = "* SIGNIFICANT" if r["p_value"] < 0.05 else ""\n',
    '        print(f"  Cutoff={c:3d}%  BW=15  "\n',
    '              f"effect={r[\'effect_at_cutoff\']:+.4f}  p={r[\'p_value\']:.3f} {sig}")\n',
    '\n',
    '# Patch cutoff into results for display\n',
    'placebo_results_with_c = []\n',
    'for c, r in zip(placebo_cutoff_vals, placebos):\n',
    '    if "error" not in r:\n',
    '        placebo_results_with_c.append({**r, "cutoff": c})\n',
    '\n',
    'fig, ax = plt.subplots(figsize=(9, 4))\n',
    'real_cutoffs  = [main["effect_at_cutoff"]]\n',
    'real_labels   = ["50% (real)"]\n',
    'placebo_effs  = [r["effect_at_cutoff"] for r in placebo_results_with_c]\n',
    'placebo_labels = [f"{r[\'cutoff\']}%" for r in placebo_results_with_c]\n',
    'all_labels = placebo_labels[:3] + real_labels + placebo_labels[3:]\n',
    'all_effs   = placebo_effs[:3]   + real_cutoffs + placebo_effs[3:]\n',
    'colors = ["steelblue"] * 3 + ["crimson"] + ["steelblue"] * 3\n',
    'ax.bar(all_labels, all_effs, color=colors)\n',
    'ax.axhline(0, color="black", lw=1)\n',
    'ax.set_ylabel("Effect at cutoff (log reviews)")\n',
    'ax.set_title("Placebo cutoff test — effect should be near zero at fake cutoffs")\n',
    'plt.tight_layout()\n',
    'plt.savefig(RESULTS / "fig_rdd_placebo.png", dpi=150)\n',
    'plt.show()\n',
]

patched = False
for cell in nb["cells"]:
    if cell.get("id") == "cd03cb85b5":
        cell["source"] = fixed_source
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    # Fall back: find by content
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "r.get('cutoff', '?'):3.0f" in src:
            cell["source"] = fixed_source
            cell["outputs"] = []
            cell["execution_count"] = None
            patched = True
            break

if patched:
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Patched successfully.")
else:
    print("ERROR: target cell not found.")
