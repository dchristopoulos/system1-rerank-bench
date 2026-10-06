"""Draw the reranker benchmark figures from results/reranker-benchmark.json. Static SVG."""

import json
from pathlib import Path

# arm, name, detail, kind, shown in the portrait figure
ROWS = [
    ("jev", "Jev 1.13", "hosted decision model", "decision", True),
    ("luna", "GPT-5.6 Luna", "LLM, ranks the whole list", "llm", True),
    ("qwen3-4b", "Qwen3-Reranker-4B", "open reranker", "reranker", True),
    ("clef", "Clef-flash 9B", "open decision model", "decision", True),
    ("qwen3-0.6b", "Qwen3-Reranker-0.6B", "open reranker", "reranker", True),
    ("bge-v2-m3", "BGE-reranker-v2-m3", "open reranker, 568M", "reranker", True),
    ("ettin-150m", "Ettin reranker 150M", "open reranker", "reranker", True),
    ("hybrid", "No reranker", "hybrid search only", "baseline", True),
    ("bge", "BGE-reranker-base", "open reranker, 278M", "reranker", False),
    ("minilm", "MiniLM-L6", "open reranker, 22M", "reranker", False),
    ("laya-rubric", "Laya 421M", "open decision model", "decision", True),
]
INK, SUB, MUTED, BG = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb"
FILL = {"decision": "#2a78d6", "llm": "#52514e", "reranker": "#9c9b95", "baseline": "#d3d2cc"}
F = 'font-family="Helvetica, Arial, sans-serif"'
PANELS = [("scifact", "SciFact", 0.9), ("nfcorpus", "NFCorpus", 0.5)]
ARTIFACT = Path("results/reranker-benchmark.json")


def load():
    """Rows present in both datasets, best mean nDCG@10 first, with seconds and price."""
    data = json.loads(ARTIFACT.read_text())["datasets"]
    summaries = {name: data[name]["comparison_set"] for name, _, _ in PANELS}
    usage = [data[name]["usage"] for name, _, _ in PANELS]
    queries = sum(s["queries"] for s in summaries.values())
    cost = {
        "jev": 1000 * sum(u["jev"]["billed_usd"] for u in usage) / queries,
        "luna": 1000 * sum(u["luna"]["list_price_usd"] for u in usage) / queries,
    }
    rows = [row for row in ROWS if all(row[0] in s["at_10"] for s in summaries.values())]
    seconds = {
        row[0]: sum(s["median_seconds_per_query"].get(row[0], 0) for s in summaries.values()) / 2
        for row in rows
    }
    # Jev was scored with its three requests in sequence and timed again with them in parallel.
    seconds["jev-parallel"] = sum(u["jev_parallel"]["median_seconds"] for u in usage) / 2
    mean = {
        row[0]: sum(s["at_10"][row[0]]["ndcg"] for s in summaries.values()) / len(summaries)
        for row in rows
    }
    rows.sort(key=lambda row: -mean[row[0]])
    return summaries, rows, seconds, cost, mean


def duration(seconds):
    return f"{seconds:.0f} s" if seconds >= 10 else f"{seconds:.2g} s"


def bar(x, y, width, height, radius, color):
    right = x + width
    return (
        f'<path d="M{x},{y} H{right - radius:.1f} Q{right:.1f},{y} {right:.1f},{y + radius} '
        f"V{y + height - radius} Q{right:.1f},{y + height} {right - radius:.1f},{y + height} "
        f'H{x} Z" fill="{color}"/>'
    )


def main():
    summaries, rows, seconds, cost, _ = load()
    step, top, left, label_w, panel_w, gap = 46, 170, 48, 300, 250, 50
    height = top + step * len(rows) + 76
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" width="1200" '
        f'height="{height}" role="img" aria-labelledby="t d">',
        '<title id="t">Reranking quality, time and price on two BEIR datasets</title>',
        '<desc id="d">nDCG at 10 for each reranker on 100 SciFact and 100 NFCorpus test '
        "queries, with median seconds per query and price per thousand queries.</desc>",
        f'<rect width="1200" height="{height}" fill="{BG}"/>',
        f'<text x="48" y="58" {F} font-size="27" font-weight="700" fill="{INK}">'
        "Reranking the same 50 hybrid-search results per query</text>",
        f'<text x="48" y="88" {F} font-size="16" fill="{SUB}">nDCG@10 on 100 test queries per '
        "BEIR dataset; higher is better. Blue: decision models. Dashed line: no reranker.</text>",
    ]
    for index, (name, title, scale_max) in enumerate(PANELS):
        x0 = left + label_w + index * (panel_w + gap)
        at_10 = summaries[name]["at_10"]
        out.append(
            f'<text x="{x0}" y="{top - 26}" {F} font-size="17" font-weight="700" fill="{INK}">'
            f"{title}</text>"
        )
        reference = x0 + panel_w * at_10["hybrid"]["ndcg"] / scale_max
        out.append(
            f'<line x1="{reference:.1f}" x2="{reference:.1f}" y1="{top - 10}" '
            f'y2="{top + step * len(rows) - 12}" stroke="{MUTED}" stroke-width="1" '
            'stroke-dasharray="3 3"/>'
        )
        for row, (arm, _, _, kind, _) in enumerate(rows):
            y, value = top + step * row, at_10[arm]["ndcg"]
            width = panel_w * value / scale_max
            out.append(bar(x0, y, width, 24, 4, FILL[kind]))
            out.append(
                f'<text x="{x0 + width + 8:.1f}" y="{y + 18}" {F} font-size="15" '
                f'font-weight="700" fill="{INK}" stroke="{BG}" stroke-width="8" '
                f'paint-order="stroke">{value:.3f}</text>'
            )
    side = left + label_w + 2 * (panel_w + gap) + 30
    out.append(
        f'<text x="{side}" y="{top - 26}" {F} font-size="17" font-weight="700" fill="{INK}">'
        "Per query</text>"
    )
    for row, (arm, name, detail, _, _) in enumerate(rows):
        y = top + step * row
        out.append(
            f'<text x="{left}" y="{y + 18}" {F} font-size="15" font-weight="700" fill="{INK}">'
            f'{name}<tspan font-weight="400" fill="{SUB}" font-size="13">  {detail}</tspan></text>'
        )
        if arm == "hybrid":
            note = "baseline"
        elif arm == "clef":
            note = "quality only"
        elif arm in cost:
            time = duration(seconds[arm])
            if arm == "jev":
                time = f"{seconds['jev-parallel']:.2g} to {time}"
            note = f"{time} · ${cost[arm]:.2f} per 1,000"
        else:
            note = f"{duration(seconds[arm])} on a laptop GPU"
        out.append(f'<text x="{side}" y="{y + 18}" {F} font-size="15" fill="{SUB}">{note}</text>')
    foot = top + step * len(rows) + 18
    out += [
        f'<text x="48" y="{foot}" {F} font-size="13" fill="{MUTED}">Decision models use one '
        "four-level relevance rubric. Open models ran on an Apple M4 Pro, which is not their "
        "serving hardware. Hosted times include the network.</text>",
        f'<text x="48" y="{foot + 20}" {F} font-size="13" fill="{MUTED}">Jev: three requests '
        "in parallel to in sequence. Luna ran on a ChatGPT plan route; its price is list API "
        "price for the measured tokens. See the report for intervals.</text>",
        "</svg>",
    ]
    Path("results/reranker-benchmark.svg").write_text("\n".join(out) + "\n")


def social():
    """Portrait 4:5 figure for a feed: one bar per model, mean of the two datasets."""
    summaries, rows, seconds, cost, score = load()
    rows = [row for row in rows if row[4]]
    width, height, x0, top = 1080, 1350, 56, 372
    step = min(110, (1244 - top) / len(rows))
    bar_max, scale_max, time_x, price_x = 560, 0.7, 872, 1024
    slowest, fastest = seconds["luna"] / seconds["jev"], seconds["luna"] / seconds["jev-parallel"]
    cheaper = cost["luna"] / cost["jev"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t d">',
        '<title id="t">Reranking quality, time and price by model</title>',
        '<desc id="d">Average nDCG at 10 over SciFact and NFCorpus for each reranker, with '
        "time and price per query.</desc>",
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<text x="{x0}" y="92" {F} font-size="55" font-weight="700" fill="{INK}">'
        "A decision model reranked</text>",
        f'<text x="{x0}" y="156" {F} font-size="55" font-weight="700" fill="{INK}">'
        "as well as an LLM,</text>",
        f'<text x="{x0}" y="220" {F} font-size="55" font-weight="700" fill="{FILL["decision"]}">'
        f"{cheaper:.1f}x cheaper, {slowest:.0f} to {fastest:.0f}x faster.</text>",
        f'<text x="{x0}" y="272" {F} font-size="25" fill="{SUB}">Each model reorders the same '
        "50 search results per query.</text>",
        f'<text x="{x0}" y="304" {F} font-size="25" fill="{SUB}">Bars: ranking quality '
        '(nDCG@10), higher is better. <tspan fill="#2a78d6" font-weight="700">Blue</tspan>: '
        "decision models.</text>",
    ]
    for x, text in [(time_x, "time"), (price_x, "$ / 1,000")]:
        out.append(
            f'<text x="{x}" y="{top - 22}" {F} font-size="21" font-weight="700" fill="{SUB}" '
            f'text-anchor="end">{text}</text>'
        )
    reference = x0 + bar_max * score["hybrid"] / scale_max
    for index, (arm, name, detail, kind, _) in enumerate(rows):
        y = top + step * index
        length = bar_max * score[arm] / scale_max
        out.append(
            f'<text x="{x0}" y="{y + 22:.1f}" {F} font-size="26" font-weight="700" fill="{INK}" '
            f'stroke="{BG}" stroke-width="8" paint-order="stroke">'
            f'{name}<tspan font-weight="400" fill="{SUB}" font-size="22">  {detail}</tspan></text>'
        )
        out.append(bar(x0, round(y + 34, 1), length, 38, 6, FILL[kind]))
        out.append(
            f'<line x1="{reference:.1f}" x2="{reference:.1f}" y1="{y + 28:.1f}" y2="{y + 78:.1f}" '
            f'stroke="{MUTED}" stroke-width="2" stroke-dasharray="5 5"/>'
        )
        out.append(
            f'<text x="{x0 + length + 12:.1f}" y="{y + 64:.1f}" {F} font-size="30" '
            f'font-weight="700" fill="{INK}" stroke="{BG}" stroke-width="9" '
            f'paint-order="stroke">{score[arm]:.2f}</text>'
        )
        hosted = arm in cost
        if arm == "hybrid":
            cells = [(price_x, "baseline")]
        elif arm == "clef":
            cells = [(time_x, "not timed"), (price_x, "open")]
        else:
            time = duration(seconds[arm]) + ("" if hosted else "*")
            if arm == "jev":
                time = f"{seconds['jev-parallel']:.2g} to {time}"
            cells = [
                (time_x, time),
                (price_x, f"${cost[arm]:.2f}" if hosted else "open"),
            ]
        for x, text in cells:
            out.append(
                f'<text x="{x}" y="{y + 63:.1f}" {F} font-size="25" '
                f'font-weight="{700 if hosted else 400}" fill="{INK if hosted else SUB}" '
                f'text-anchor="end">{text}</text>'
            )
    lines = [
        "Dashed line: no reranker. Mean of SciFact and NFCorpus (BEIR), 100 test queries each.",
        "The top three could not be separated. Jev time: three requests in parallel to in sequence.",
        "LLM time is a ChatGPT plan route; its price is list API price for the measured tokens.",
        "*Open models ran on an Apple M4 Pro laptop GPU, not their serving hardware.",
    ]
    for index, text in enumerate(lines):
        out.append(
            f'<text x="{x0}" y="{1262 + 25 * index}" {F} font-size="20" fill="{MUTED}">'
            f"{text}</text>"
        )
    out.append("</svg>")
    Path("results/reranker-benchmark-social.svg").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
    social()
