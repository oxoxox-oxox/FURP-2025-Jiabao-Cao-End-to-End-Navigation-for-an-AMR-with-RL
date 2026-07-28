"""
Create grouped bar charts from final evaluation JSON without extra packages.

Run:

    cd ~/furp-2026-WenjingChen-RL-Navigation-for-AMR
    python src/code/Wenjing_Chen/CNNTD3/final_stps/plot_final_bars.py

Input:

    src/results/final/unified_comparison.json

Outputs:

    docs/img/final_success_rate_bars.svg
    docs/img/final_success_rate_avg.svg
"""
import json
import os


REPO_DIR = os.path.expanduser("~/furp-2026-WenjingChen-RL-Navigation-for-AMR")
INPUT_JSON = os.path.join(REPO_DIR, "src", "results", "final", "unified_comparison.json")
OUT_DIR = os.path.join(REPO_DIR, "docs", "img")

SCENARIOS = ["S1_U_trap", "S2_Double_U", "S3_Narrow_door", "S5_Corridor"]
SCENARIO_LABELS = ["U-trap", "Double-U", "Narrow door", "Corridor"]
METHODS = ["CNNTD3_baseline", "NeuPAN", "STPS_v2"]
METHOD_LABELS = ["CNNTD3", "NeuPAN", "STPS v2"]
COLORS = ["#4c78a8", "#9aa3ad", "#f58518"]


def load_values():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    means = [
        [data[method][scenario]["mean"] * 100 for scenario in SCENARIOS]
        for method in METHODS
    ]
    stds = [
        [data[method][scenario]["std"] * 100 for scenario in SCENARIOS]
        for method in METHODS
    ]
    return means, stds


def svg_text(x, y, text, size=14, anchor="middle", weight="normal", color="#222"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{text}</text>'
    )


def plot_grouped_bars(means, stds):
    width, height = 980, 520
    left, right, top, bottom = 82, 35, 70, 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    scenario_w = chart_w / len(SCENARIOS)
    bar_w = 42

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "Hard-scenario Navigation Success Rate", 22, weight="bold"),
    ]

    for tick in range(0, 101, 20):
        y = top + chart_h - tick / 100 * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dde4" stroke-dasharray="4 4"/>')
        parts.append(svg_text(left - 12, y + 5, str(tick), 12, anchor="end", color="#555"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#333"/>')
    parts.append(svg_text(24, top + chart_h / 2, "Success Rate (%)", 13, anchor="middle"))

    for mi, (label, color) in enumerate(zip(METHOD_LABELS, COLORS)):
        lx = left + 250 + mi * 145
        parts.append(f'<rect x="{lx}" y="48" width="18" height="18" fill="{color}"/>')
        parts.append(svg_text(lx + 26, 63, label, 13, anchor="start"))

    for si, scenario in enumerate(SCENARIO_LABELS):
        center = left + scenario_w * (si + 0.5)
        parts.append(svg_text(center, top + chart_h + 34, scenario, 13))
        for mi in range(len(METHODS)):
            value = means[mi][si]
            std = stds[mi][si]
            x = center + (mi - 1) * (bar_w + 8) - bar_w / 2
            h = value / 100 * chart_h
            y = top + chart_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{COLORS[mi]}" rx="2"/>')
            parts.append(svg_text(x + bar_w / 2, y - 7, f"{value:.0f}", 11))
            if std > 0:
                err_h = std / 100 * chart_h
                err_y1 = max(top, y - err_h)
                err_y2 = min(top + chart_h, y + err_h)
                cx = x + bar_w / 2
                parts.append(f'<line x1="{cx:.1f}" y1="{err_y1:.1f}" x2="{cx:.1f}" y2="{err_y2:.1f}" stroke="#222" stroke-width="1.4"/>')
                parts.append(f'<line x1="{cx-7:.1f}" y1="{err_y1:.1f}" x2="{cx+7:.1f}" y2="{err_y1:.1f}" stroke="#222" stroke-width="1.4"/>')
                parts.append(f'<line x1="{cx-7:.1f}" y1="{err_y2:.1f}" x2="{cx+7:.1f}" y2="{err_y2:.1f}" stroke="#222" stroke-width="1.4"/>')

    parts.append("</svg>")
    with open(os.path.join(OUT_DIR, "final_success_rate_bars.svg"), "w") as f:
        f.write("\n".join(parts))


def plot_average_bar(means):
    avg = [sum(row) / len(row) for row in means]
    width, height = 620, 430
    left, right, top, bottom = 72, 35, 65, 70
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_w = 86
    spacing = chart_w / len(METHODS)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "Average over Four Hard Scenarios", 21, weight="bold"),
    ]
    for tick in range(0, 101, 20):
        y = top + chart_h - tick / 100 * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dde4" stroke-dasharray="4 4"/>')
        parts.append(svg_text(left - 10, y + 5, str(tick), 12, anchor="end", color="#555"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#333"/>')

    for mi, label in enumerate(METHOD_LABELS):
        center = left + spacing * (mi + 0.5)
        value = avg[mi]
        h = value / 100 * chart_h
        y = top + chart_h - h
        x = center - bar_w / 2
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{COLORS[mi]}" rx="3"/>')
        parts.append(svg_text(center, y - 9, f"{value:.0f}%", 14, weight="bold"))
        parts.append(svg_text(center, top + chart_h + 34, label, 13))

    parts.append(svg_text(24, top + chart_h / 2, "Average SR (%)", 13))
    parts.append("</svg>")
    with open(os.path.join(OUT_DIR, "final_success_rate_avg.svg"), "w") as f:
        f.write("\n".join(parts))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    means, stds = load_values()
    plot_grouped_bars(means, stds)
    plot_average_bar(means)
    print("Saved:")
    print(os.path.join(OUT_DIR, "final_success_rate_bars.svg"))
    print(os.path.join(OUT_DIR, "final_success_rate_avg.svg"))


if __name__ == "__main__":
    main()
