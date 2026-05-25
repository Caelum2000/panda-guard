#!/usr/bin/env python3
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


def apply_style(font_scale: float) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=font_scale)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "svg.fonttype": "none",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(path: Path) -> None:
    resolved = path.resolve()
    expected_suffix = Path("outputs/ai4sci_paper_draw/risk_radar_compare")
    suffixes = [Path(*resolved.parts[i:]) for i in range(len(resolved.parts))]
    if expected_suffix not in suffixes:
        raise ValueError(f"Refusing to remove unexpected output directory: {path}")
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def save_figure(fig: plt.Figure, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    ensure_dir(output_base.parent)
    fig.tight_layout(pad=0.2)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def load_risk_tables(eval_dirs: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    parts = []
    for group_name, paths in eval_dirs.items():
        df = pd.read_csv(paths["risk_dimension_margin"])
        df["model_group"] = group_name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def pivot_asr(df: pd.DataFrame, risk_order: List[str] | None = None) -> pd.DataFrame:
    weighted = (
        df.groupby(["model_name", "Risk Dimension"], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    weighted["asr"] = weighted["attacked"] / weighted["total"] * 100.0
    pivot = weighted.pivot(index="model_name", columns="Risk Dimension", values="asr")
    if risk_order is not None:
        pivot = pivot.reindex(columns=risk_order)
    return pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]


def resolve_risk_order(df: pd.DataFrame, order_cfg: object) -> List[str]:
    if isinstance(order_cfg, list):
        observed = set(df["Risk Dimension"].dropna().astype(str))
        configured = [str(item) for item in order_cfg if str(item) in observed]
        missing = sorted(observed.difference(configured))
        return configured + missing

    weighted = (
        df.groupby(["model_name", "Risk Dimension"], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    weighted["asr"] = weighted["attacked"] / weighted["total"] * 100.0
    if str(order_cfg) == "alphabetical":
        return sorted(weighted["Risk Dimension"].dropna().unique().tolist())
    return (
        weighted.groupby("Risk Dimension")["asr"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def plot_radar(
    pivot: pd.DataFrame,
    title: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    plot_cfg: Dict[str, object],
    show_risk_labels: bool = True,
) -> None:
    if pivot.empty:
        return

    top_n_models = int(plot_cfg.get("top_radar_models", 8))
    radar_df = pivot.fillna(0)
    radar_df = radar_df.loc[radar_df.mean(axis=1).sort_values(ascending=False).head(top_n_models).index]

    labels = radar_df.columns.tolist() if show_risk_labels else [""] * len(radar_df.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    figure_size = float(plot_cfg.get("figure_size", 9.8))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), subplot_kw={"polar": True})
    palette = sns.color_palette("tab10", n_colors=len(radar_df))
    line_width = float(plot_cfg.get("line_width", 2.4))
    fill_alpha = float(plot_cfg.get("fill_alpha", 0.08))

    for color, (model_name, values) in zip(palette, radar_df.iterrows()):
        series = values.tolist()
        series += series[:1]
        ax.plot(angles, series, linewidth=line_width, label=model_name, color=color)
        ax.fill(angles, series, alpha=fill_alpha, color=color)

    radial_limit = float(plot_cfg.get("radial_limit", 100))
    radial_ticks = list(plot_cfg.get("radial_ticks", [20, 40, 60, 80, 100]))
    axis_label_font_size = int(plot_cfg.get("axis_label_font_size", 11))
    radial_tick_font_size = int(plot_cfg.get("radial_tick_font_size", 10))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=axis_label_font_size, fontweight="bold")
    ax.set_ylim(0, radial_limit)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(
        [str(int(tick)) if float(tick).is_integer() else str(tick) for tick in radial_ticks],
        fontsize=radial_tick_font_size,
        fontweight="bold",
    )
    ax.set_title(title, y=1.08, fontsize=int(plot_cfg.get("title_font_size", 18)), fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, float(plot_cfg.get("legend_y", -0.11))),
        ncol=int(plot_cfg.get("legend_columns", 2)),
        frameon=False,
        fontsize=int(plot_cfg.get("legend_font_size", 11)),
    )

    save_figure(fig, output_base, formats, dpi)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to YAML config file.",
)
def main(config: str) -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    output_dir = Path(cfg["output_dir"])
    reset_output_dir(output_dir)

    plot_cfg = cfg.get("plot", {})
    formats = plot_cfg.get("formats", ["svg"])
    dpi = int(plot_cfg.get("dpi", 300))
    group_order = list(plot_cfg.get("group_order", ["base", "sci"]))
    group_labels = dict(plot_cfg.get("group_labels", {}))

    apply_style(float(plot_cfg.get("font_scale", 1.55)))
    table = load_risk_tables(cfg["inputs"]["eval_dirs"])
    risk_order = resolve_risk_order(table, plot_cfg.get("risk_dimension_order", "combined_mean"))

    for group_name in group_order:
        group_df = table[table["model_group"] == group_name]
        if group_df.empty:
            continue
        pivot = pivot_asr(group_df, risk_order)
        pivot.to_csv(output_dir / f"risk_dimension_model_{group_name}_radar_matrix.csv")
        plot_radar(
            pivot,
            f"Risk Dimension ASR Radar ({group_labels.get(group_name, group_name.capitalize())})",
            output_dir / f"risk_dimension_model_{group_name}_radar",
            formats,
            dpi,
            plot_cfg,
        )
        if bool(plot_cfg.get("output_no_risk_label", True)):
            suffix = str(plot_cfg.get("no_risk_label_suffix", "no_risk_label"))
            plot_radar(
                pivot,
                f"Risk Dimension ASR Radar ({group_labels.get(group_name, group_name.capitalize())})",
                output_dir / f"risk_dimension_model_{group_name}_radar_{suffix}",
                formats,
                dpi,
                plot_cfg,
                show_risk_labels=False,
            )

    print(f"[SAVED] risk radar compare outputs under {output_dir}")


if __name__ == "__main__":
    main()
