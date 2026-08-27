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
    expected_suffix = Path("outputs/ai4sci_paper_draw/discipline_compare_radar")
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


def load_subject_tables(eval_dirs: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    parts = []
    for group_name, paths in eval_dirs.items():
        df = pd.read_csv(paths["subject_margin"])
        df["model_group"] = group_name
        df["Subject"] = df["Subject"].astype(str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def average_group_asr(df: pd.DataFrame) -> pd.DataFrame:
    per_model = (
        df.groupby(["model_group", "model_name", "Subject"], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    per_model["asr"] = per_model["attacked"] / per_model["total"] * 100.0
    averaged = (
        per_model.groupby(["model_group", "Subject"], as_index=False)
        .agg(
            mean_asr=("asr", "mean"),
            std_asr=("asr", "std"),
            model_count=("model_name", "nunique"),
            total=("total", "sum"),
            attacked=("attacked", "sum"),
        )
    )
    return averaged


def resolve_subject_order(averaged: pd.DataFrame, order_cfg: object) -> List[str]:
    observed = averaged["Subject"].dropna().astype(str).unique().tolist()
    if isinstance(order_cfg, list):
        ordered = [str(subject) for subject in order_cfg if str(subject) in observed]
        ordered.extend(sorted([subject for subject in observed if subject not in ordered], key=str.lower))
        return ordered
    if str(order_cfg) == "alphabetical":
        return sorted(observed, key=str.lower)
    return (
        averaged.groupby("Subject", observed=True)["mean_asr"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def plot_compare_radar(
    averaged: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    plot_cfg: Dict[str, object],
) -> None:
    subject_order = resolve_subject_order(averaged, plot_cfg.get("subject_order", "combined_mean"))
    group_order = list(plot_cfg.get("group_order", ["base", "sci"]))
    subject_labels = dict(plot_cfg.get("subject_labels", {}))
    group_labels = dict(plot_cfg.get("group_labels", {}))
    group_colors = dict(plot_cfg.get("group_colors", {}))

    labels = [subject_labels.get(subject, subject.title()) for subject in subject_order]
    angles = np.linspace(0, 2 * np.pi, len(subject_order), endpoint=False).tolist()
    angles += angles[:1]

    figure_size = float(plot_cfg.get("figure_size", 8.8))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), subplot_kw={"polar": True})
    palette = sns.color_palette("Set2", n_colors=len(group_order))
    color_lookup = {group: palette[index] for index, group in enumerate(group_order)}
    color_lookup.update(group_colors)

    for group_name in group_order:
        group_df = averaged[averaged["model_group"] == group_name].set_index("Subject")
        if group_df.empty:
            continue
        values = group_df.reindex(subject_order)["mean_asr"].fillna(0).tolist()
        values += values[:1]
        color = color_lookup[group_name]
        ax.plot(
            angles,
            values,
            linewidth=float(plot_cfg.get("line_width", 2.8)),
            marker="o",
            markersize=float(plot_cfg.get("marker_size", 6)),
            label=group_labels.get(group_name, group_name.capitalize()),
            color=color,
        )
        ax.fill(angles, values, alpha=float(plot_cfg.get("fill_alpha", 0.12)), color=color)

    radial_limit = float(plot_cfg.get("radial_limit", 100))
    radial_ticks = list(plot_cfg.get("radial_ticks", [20, 40, 60, 80, 100]))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        labels,
        fontsize=int(plot_cfg.get("axis_label_font_size", 13)),
        fontweight="bold",
    )
    ax.set_ylim(0, radial_limit)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(
        [str(int(tick)) if float(tick).is_integer() else str(tick) for tick in radial_ticks],
        fontsize=int(plot_cfg.get("radial_tick_font_size", 11)),
        fontweight="bold",
    )
    ax.set_title(
        str(plot_cfg.get("title", "Average ASR by Scientific Discipline")),
        y=1.08,
        fontsize=int(plot_cfg.get("title_font_size", 18)),
        fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, float(plot_cfg.get("legend_y", -0.08))),
        ncol=len(group_order),
        frameon=False,
        fontsize=int(plot_cfg.get("legend_font_size", 12)),
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

    apply_style(float(plot_cfg.get("font_scale", 1.45)))
    table = load_subject_tables(cfg["inputs"]["eval_dirs"])
    averaged = average_group_asr(table)
    subject_order = resolve_subject_order(averaged, plot_cfg.get("subject_order", "combined_mean"))
    group_order = list(plot_cfg.get("group_order", ["base", "sci"]))
    averaged["Subject"] = pd.Categorical(averaged["Subject"], categories=subject_order, ordered=True)
    averaged["model_group"] = pd.Categorical(averaged["model_group"], categories=group_order, ordered=True)
    averaged = averaged.sort_values(["model_group", "Subject"])
    averaged.to_csv(output_dir / "discipline_group_average_asr.csv", index=False)

    plot_compare_radar(
        averaged,
        output_dir / "discipline_compare_average_radar",
        formats,
        dpi,
        plot_cfg,
    )

    print(f"[SAVED] discipline compare radar outputs under {output_dir}")


if __name__ == "__main__":
    main()
