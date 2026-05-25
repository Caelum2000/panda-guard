#!/usr/bin/env python3
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


SEPARATOR_PREFIX = "__separator__"


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
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
    expected_suffix = Path("outputs/ai4sci_paper_draw/risk_heatmap_compare")
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
        risk_path = paths["risk_dimension_margin"]
        df = pd.read_csv(risk_path)
        df["model_group"] = group_name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_model_order(df: pd.DataFrame, group_order: List[str]) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    order: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for group_name in group_order:
        group_df = df[df["model_group"] == group_name]
        if group_df.empty:
            continue
        means = (
            group_df.groupby("model_name", as_index=False)
            .agg(total=("total", "sum"), attacked=("attacked", "sum"))
            .assign(asr=lambda x: x["attacked"] / x["total"] * 100.0)
            .sort_values(["asr", "model_name"], ascending=[False, True])
        )
        start = len(order)
        group_models = means["model_name"].tolist()
        order.extend(group_models)
        spans[group_name] = (start, len(order) - 1)
    return order, spans


def pivot_risk_heatmap(df: pd.DataFrame, model_order: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(["model_name", "Risk Dimension"], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    grouped["asr"] = grouped["attacked"] / grouped["total"] * 100.0
    pivot = grouped.pivot(index="Risk Dimension", columns="model_name", values="asr")
    risk_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    return pivot.loc[risk_order, model_order]


def insert_group_separators(
    pivot: pd.DataFrame,
    group_order: List[str],
    spans: Dict[str, Tuple[int, int]],
    separator_columns: int,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    if separator_columns <= 0:
        return pivot, spans

    columns: List[str] = []
    adjusted_spans: Dict[str, Tuple[int, int]] = {}
    for group_index, group_name in enumerate(group_order):
        if group_name not in spans:
            continue
        start, end = spans[group_name]
        adjusted_start = len(columns)
        columns.extend(pivot.columns[start : end + 1].tolist())
        adjusted_spans[group_name] = (adjusted_start, len(columns) - 1)
        if group_index < len(group_order) - 1:
            for sep_index in range(separator_columns):
                columns.append(f"{SEPARATOR_PREFIX}{group_index}_{sep_index}")

    separated = pd.DataFrame(index=pivot.index, columns=columns, dtype=float)
    for column in pivot.columns:
        separated[column] = pivot[column]
    return separated, adjusted_spans


def add_group_labels(
    ax: plt.Axes,
    spans: Dict[str, Tuple[int, int]],
    group_order: List[str],
    group_labels: Dict[str, str],
) -> None:
    total_columns = len(ax.get_xticks())
    for group_name in group_order:
        if group_name not in spans:
            continue
        start, end = spans[group_name]
        center = ((start + end + 1) / 2) / max(total_columns, 1)
        ax.text(
            center,
            1.035,
            group_labels.get(group_name, group_name.capitalize()),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


def plot_horizontal_heatmap(
    pivot: pd.DataFrame,
    spans: Dict[str, Tuple[int, int]],
    group_order: List[str],
    group_labels: Dict[str, str],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    plot_cfg: Dict[str, object],
) -> None:
    cell_width = float(plot_cfg.get("cell_width", 0.46))
    cell_height = float(plot_cfg.get("cell_height", 0.44))
    width = max(
        float(plot_cfg.get("min_width", 13.5)),
        min(float(plot_cfg.get("max_width", 22.0)), pivot.shape[1] * cell_width + 3.4),
    )
    height = max(
        float(plot_cfg.get("min_height", 5.0)),
        min(float(plot_cfg.get("max_height", 8.0)), pivot.shape[0] * cell_height + 1.6),
    )

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=100,
        linewidths=0.25,
        linecolor="white",
        mask=pivot.isna(),
        cbar_kws={"label": "ASR (%)", "pad": 0.012},
        ax=ax,
    )

    labels = ["" if str(col).startswith(SEPARATOR_PREFIX) else str(col) for col in pivot.columns]
    ax.set_xticklabels(labels, rotation=58, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    ax.set_facecolor("white")
    add_group_labels(ax, spans, group_order, group_labels)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

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
    group_order = list(plot_cfg.get("group_order", ["sci", "base"]))
    group_labels = dict(plot_cfg.get("group_labels", {}))
    separator_columns = int(plot_cfg.get("separator_columns", 1))

    apply_style()
    table = load_risk_tables(cfg["inputs"]["eval_dirs"])
    model_order, spans = build_model_order(table, group_order)
    pivot = pivot_risk_heatmap(table, model_order)
    pivot, spans = insert_group_separators(pivot, group_order, spans, separator_columns)
    pivot.to_csv(output_dir / "risk_dimension_model_all_horizontal_matrix.csv")

    plot_horizontal_heatmap(
        pivot,
        spans,
        group_order,
        group_labels,
        output_dir / "risk_dimension_model_all_horizontal_heatmap",
        formats,
        dpi,
        plot_cfg,
    )

    print(f"[SAVED] risk heatmap compare outputs under {output_dir}")


if __name__ == "__main__":
    main()
