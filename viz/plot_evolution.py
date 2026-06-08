"""Interactive bump chart showing how buy priorities evolve over generations.

Use the phase selector (Early / Mid / Late) to switch views.
Hover over any point to see the full buy order at that generation.

Usage:
    python plot_evolution.py [best_model_dir]
"""

import json
import os
import re
import sys

import plotly.graph_objects as go


CARD_COLORS = {
    "Province": "#6a0dad",
    "Duchy": "#9b59b6",
    "Estate": "#c39bd3",
    "Gold": "#f0b90b",
    "Silver": "#95a5a6",
    "Copper": "#cd7f32",
    "Village": "#27ae60",
    "Smithy": "#2980b9",
    "Market": "#16a085",
    "Laboratory": "#2471a3",
    "Festival": "#c0392b",
    "Chapel": "#d35400",
    "PASS": "#2c3e50",
}

PHASE_KEYS = [
    ("early_buy_priority", "Early"),
    ("mid_buy_priority", "Mid"),
    ("late_buy_priority", "Late"),
]


def get_color(card: str) -> str:
    return CARD_COLORS.get(card, "#7f8c8d")


def load_generations(model_dir: str) -> list[dict]:
    """Load all gen_NNN/strategy.json + summary.txt into a sorted list."""
    entries = []
    for name in os.listdir(model_dir):
        m = re.match(r"gen_(\d+)", name)
        if not m:
            continue
        gen = int(m.group(1))
        json_path = os.path.join(model_dir, name, "strategy.json")
        summary_path = os.path.join(model_dir, name, "summary.txt")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        win_rate = None
        opponent = "Big Money"
        if os.path.exists(summary_path):
            text = open(summary_path).read()
            wr = re.search(r"Win rate:\s+(\d+)%", text)
            if wr:
                win_rate = int(wr.group(1)) / 100
            opp = re.search(r"Performance \(\d+ games vs (.+?)\)", text)
            if opp:
                opponent = opp.group(1)

        entries.append({
            "gen": gen, "data": data,
            "win_rate": win_rate, "opponent": opponent,
        })

    entries.sort(key=lambda e: e["gen"])
    return entries


def build_hover(entries: list[dict], key: str) -> dict[str, list[str]]:
    """Build hover text per card showing the full buy order context."""
    hovers = {}
    for e in entries:
        lst = e["data"][key]
        # Build a readable buy order string
        order_lines = []
        for i, c in enumerate(lst):
            order_lines.append(f"{'>' if i > 0 else ''} {i+1}. {c}")
            if c == "PASS":
                order_lines.append(f"  — stops buying here —")
                break
        order_str = "<br>".join(order_lines)

        wr = f"{e['win_rate']:.0%}" if e["win_rate"] is not None else "?"
        header = f"<b>Gen {e['gen']}</b> (win rate: {wr})<br><br>"

        for i, card in enumerate(lst):
            if card not in hovers:
                hovers[card] = []
            hovers[card].append(header + order_str)
            if card == "PASS":
                break

    return hovers


def build_dashboard(entries: list[dict], output: str) -> None:
    """Build interactive bump chart with phase selector."""
    gens = [e["gen"] for e in entries]
    n_gens = len(entries)

    # Collect all cards across all phases
    all_cards = set()
    for e in entries:
        for key, _ in PHASE_KEYS:
            for c in e["data"][key]:
                all_cards.add(c)
    all_cards = sorted(all_cards, key=lambda c: (c == "PASS", c))

    fig = go.Figure()

    # For each phase, create one trace per card.
    # Use visibility toggling for the phase slicer.
    traces_per_phase = []

    for key, phase_label in PHASE_KEYS:
        hovers = build_hover(entries, key)
        phase_traces = []

        for card in all_cards:
            ranks = []
            card_gens = []
            card_hovers = []

            for i, e in enumerate(entries):
                lst = e["data"][key]
                # Only show up to PASS
                visible = []
                for c in lst:
                    visible.append(c)
                    if c == "PASS":
                        break

                if card in visible:
                    ranks.append(visible.index(card) + 1)
                    card_gens.append(e["gen"])
                    if card in hovers and i < len(hovers[card]):
                        card_hovers.append(hovers[card][len(card_gens) - 1]
                                           if len(card_gens) - 1 < len(hovers.get(card, []))
                                           else "")
                    else:
                        card_hovers.append("")

            if not ranks:
                # Card never appears before PASS in this phase
                phase_traces.append(None)
                continue

            is_pass = card == "PASS"
            trace = go.Scatter(
                x=card_gens, y=ranks,
                mode="lines+markers+text",
                name=card,
                legendgroup=card,
                showlegend=(key == PHASE_KEYS[0][0]),
                marker=dict(
                    size=10 if is_pass else 8,
                    color=get_color(card),
                    symbol="x" if is_pass else "circle",
                    line=dict(width=1, color="white"),
                ),
                line=dict(
                    width=3 if is_pass else 2,
                    color=get_color(card),
                    dash="dot" if is_pass else "solid",
                ),
                text=[card if i == len(card_gens) - 1 else ""
                      for i in range(len(card_gens))],
                textposition="middle right",
                textfont=dict(size=11, color=get_color(card)),
                hovertext=card_hovers,
                hoverinfo="text",
                visible=(key == PHASE_KEYS[0][0]),  # Early visible by default
            )
            fig.add_trace(trace)
            phase_traces.append(len(fig.data) - 1)

        traces_per_phase.append(phase_traces)

    # Phase selector buttons
    buttons = []
    for phase_idx, (key, phase_label) in enumerate(PHASE_KEYS):
        visibility = [False] * len(fig.data)
        for pt in traces_per_phase[phase_idx]:
            if pt is not None:
                visibility[pt] = True

        buttons.append(dict(
            label=f"  {phase_label}  ",
            method="update",
            args=[
                {"visible": visibility},
                {"title.text": f"Buy Priority — {phase_label} Phase"},
            ],
        ))

    # Layout
    max_rank = 0
    for e in entries:
        for key, _ in PHASE_KEYS:
            lst = e["data"][key]
            pass_pos = lst.index("PASS") + 1 if "PASS" in lst else len(lst)
            max_rank = max(max_rank, pass_pos)

    fig.update_layout(
        title=dict(
            text="Buy Priority — Early Phase",
            font=dict(size=20),
            x=0.5,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=1.12, yanchor="top",
            buttons=buttons,
            bgcolor="#ecf0f1",
            font=dict(size=14),
            active=0,
        )],
        yaxis=dict(
            title="Buy Priority (1 = first to buy)",
            autorange="reversed",
            dtick=1,
            range=[0.3, max_rank + 0.7],
            gridcolor="#ecf0f1",
        ),
        xaxis=dict(
            title="Generation",
            gridcolor="#ecf0f1",
        ),
        template="plotly_white",
        height=600,
        width=1000,
        legend=dict(
            title="Card",
            orientation="v",
            x=1.02, y=1,
            font=dict(size=12),
        ),
        hovermode="closest",
        margin=dict(t=100),
    )

    fig.write_html(output, include_plotlyjs="cdn")
    print(f"Dashboard saved to {output}")
    print(f"Open in browser: file://{os.path.abspath(output)}")


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "best_model"
    if not os.path.isdir(model_dir):
        print(f"Error: {model_dir} not found. Run main.py first.")
        sys.exit(1)

    entries = load_generations(model_dir)
    if not entries:
        print(f"No gen_NNN/ subfolders found in {model_dir}")
        sys.exit(1)

    print(f"Found {len(entries)} generations "
          f"(gen {entries[0]['gen']} to {entries[-1]['gen']})")

    output = os.path.join(model_dir, "evolution.html")
    build_dashboard(entries, output)


if __name__ == "__main__":
    main()
