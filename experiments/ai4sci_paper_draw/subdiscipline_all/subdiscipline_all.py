#!/usr/bin/env python3
import shutil
import textwrap
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
            "axes.spines.left": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "svg.fonttype": "none",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(path: Path) -> None:
    resolved = path.resolve()
    expected_suffix = Path("outputs/ai4sci_paper_draw/subdiscipline_all")
    suffixes = [Path(*resolved.parts[i:]) for i in range(len(resolved.parts))]
    if expected_suffix not in suffixes:
        raise ValueError(f"Refusing to remove unexpected output directory: {path}")
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def save_figure(fig: plt.Figure, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    ensure_dir(output_base.parent)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def load_subdiscipline_tables(eval_dirs: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    parts = []
    for group_name, paths in eval_dirs.items():
        df = pd.read_csv(paths["subject_subdiscipline"])
        df["model_group"] = group_name
        parts.append(df)
    table = pd.concat(parts, ignore_index=True)
    table["Subject"] = table["Subject"].astype(str)
    table["Sub-discipline"] = table["Sub-discipline"].astype(str)
    return table


def average_asr_by_subdiscipline(df: pd.DataFrame) -> pd.DataFrame:
    per_model = (
        df.groupby(["Subject", "Sub-discipline", "model_name"], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    per_model["asr"] = per_model["attacked"] / per_model["total"] * 100.0
    averaged = (
        per_model.groupby(["Subject", "Sub-discipline"], as_index=False)
        .agg(
            mean_asr=("asr", "mean"),
            std_asr=("asr", "std"),
            model_count=("model_name", "nunique"),
            total=("total", "sum"),
            attacked=("attacked", "sum"),
        )
    )
    return averaged


def resolve_subject_order(df: pd.DataFrame, configured_order: List[str] | None) -> List[str]:
    observed = df["Subject"].dropna().unique().tolist()
    if not configured_order:
        return sorted(observed, key=str.lower)
    ordered = [subject for subject in configured_order if subject in observed]
    ordered.extend(sorted([subject for subject in observed if subject not in ordered], key=str.lower))
    return ordered


def sorted_subject_df(subject_df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if sort_mode == "alphabetical":
        return subject_df.sort_values("Sub-discipline", key=lambda x: x.str.lower())
    return subject_df.sort_values(["mean_asr", "Sub-discipline"], ascending=[False, True])


def build_plot_table(averaged: pd.DataFrame, plot_cfg: Dict[str, object]) -> pd.DataFrame:
    subject_order = resolve_subject_order(averaged, plot_cfg.get("subject_order"))
    subject_prefixes = dict(plot_cfg.get("subject_prefixes", {}))
    sort_mode = str(plot_cfg.get("sort_subdisciplines", "mean_asr_desc"))
    subject_gap = float(plot_cfg.get("subject_gap", 0.9))

    rows = []
    x_cursor = 0.0
    for subject in subject_order:
        subject_df = averaged[averaged["Subject"] == subject].copy()
        subject_df = sorted_subject_df(subject_df, sort_mode)
        if subject_df.empty:
            continue
        prefix = subject_prefixes.get(subject, subject[:1].upper())
        for index, (_, row) in enumerate(subject_df.iterrows(), start=1):
            rows.append(
                {
                    "Subject": subject,
                    "Sub-discipline": row["Sub-discipline"],
                    "code": f"{prefix}-{index}",
                    "mean_asr": row["mean_asr"],
                    "std_asr": row["std_asr"],
                    "model_count": row["model_count"],
                    "total": row["total"],
                    "attacked": row["attacked"],
                    "x": x_cursor,
                }
            )
            x_cursor += 1.0
        x_cursor += subject_gap
    return pd.DataFrame(rows)


def plot_subdiscipline_all(
    averaged: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    plot_cfg: Dict[str, object],
) -> None:
    subject_order = resolve_subject_order(averaged, plot_cfg.get("subject_order"))
    subject_labels = dict(plot_cfg.get("subject_labels", {}))
    palette = dict(plot_cfg.get("palette", {}))
    plot_df = build_plot_table(averaged, plot_cfg)

    fig_width = float(plot_cfg.get("figure_width", 22.0))
    fig_height = float(plot_cfg.get("figure_height", 6.8))
    max_asr = float(plot_cfg.get("max_asr", 100))
    x_tick_step = float(plot_cfg.get("x_tick_step", 20))
    bar_width = float(plot_cfg.get("bar_width", 0.72))
    value_padding = float(plot_cfg.get("value_padding", 1.2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    subject_spans = []
    for span_index, subject in enumerate(subject_order):
        subject_df = plot_df[plot_df["Subject"] == subject]
        if subject_df.empty:
            continue
        x_min = subject_df["x"].min() - 0.5
        x_max = subject_df["x"].max() + 0.5
        x_center = (x_min + x_max) / 2
        subject_spans.append((subject, x_min, x_max, x_center))
        if span_index % 2 == 0:
            ax.axvspan(x_min, x_max, color="#F5F7FA", zorder=0)

    for subject in subject_order:
        subject_df = plot_df[plot_df["Subject"] == subject]
        if subject_df.empty:
            continue
        color = palette.get(subject, "#4E79A7")
        ax.bar(
            subject_df["x"],
            subject_df["mean_asr"],
            width=bar_width,
            color=color,
            edgecolor="#1F2933",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )
        ax.errorbar(
            subject_df["x"],
            subject_df["mean_asr"],
            yerr=subject_df["std_asr"].fillna(0),
            fmt="none",
            ecolor="#2F3437",
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            alpha=0.58,
            clip_on=False,
            zorder=4,
        )

    for _, row in plot_df.iterrows():
        label_y = min(float(row["mean_asr"]) + value_padding, max_asr - 0.8)
        va = "bottom" if row["mean_asr"] <= max_asr - 7 else "top"
        ax.text(
            row["x"],
            label_y,
            f"{row['mean_asr']:.1f}",
            ha="center",
            va=va,
            fontsize=int(plot_cfg.get("value_font_size", 9)),
            fontweight="bold",
            color="#1F2933",
            zorder=5,
        )

    for subject, _, _, x_center in subject_spans:
        ax.text(
            x_center,
            -0.08,
            subject_labels.get(subject, subject.title()),
            ha="center",
            va="top",
            fontsize=int(plot_cfg.get("subject_font_size", 15)),
            fontweight="bold",
            color=palette.get(subject, "#4E79A7"),
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(plot_df["x"])
    ax.set_xticklabels(
        plot_df["code"],
        fontsize=int(plot_cfg.get("code_font_size", 12)),
        fontweight="bold",
        rotation=0,
    )
    ax.set_xlim(plot_df["x"].min() - 0.8, plot_df["x"].max() + 0.8)
    ax.set_ylim(0, max_asr)
    ax.set_ylabel("Average Attack Success Rate (%)", fontsize=int(plot_cfg.get("tick_font_size", 11)), fontweight="bold")
    ax.set_title(
        "Average Attack Success Rate Across Scientific Sub-disciplines",
        fontsize=int(plot_cfg.get("title_font_size", 16)),
        fontweight="bold",
        pad=14,
    )
    ticks = np.arange(0, max_asr + 0.001, x_tick_step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(int(tick)) for tick in ticks])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D4DAE0", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=int(plot_cfg.get("tick_font_size", 11)))
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.9, bottom=0.22)
    save_figure(fig, output_base, formats, dpi)


def plot_code_legend(
    plot_df: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    plot_cfg: Dict[str, object],
) -> None:
    subject_order = resolve_subject_order(plot_df, plot_cfg.get("subject_order"))
    subject_labels = dict(plot_cfg.get("subject_labels", {}))
    palette = dict(plot_cfg.get("palette", {}))
    fig_width = float(plot_cfg.get("legend_figure_width", 18.5))
    fig_height = float(plot_cfg.get("legend_figure_height", 5.8))
    legend_font_size = int(plot_cfg.get("legend_font_size", 10))
    wrap_width = int(plot_cfg.get("legend_wrap_width", 25))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    x_positions = np.linspace(0.015, 0.865, len(subject_order))
    for x_pos, subject in zip(x_positions, subject_order):
        subject_df = plot_df[plot_df["Subject"] == subject]
        if subject_df.empty:
            continue
        color = palette.get(subject, "#4E79A7")
        ax.text(
            x_pos,
            0.96,
            subject_labels.get(subject, subject.title()),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_font_size + 2,
            fontweight="bold",
            color=color,
        )
        y_pos = 0.84
        for _, row in subject_df.iterrows():
            text = textwrap.fill(
                f"{row['code']}: {row['Sub-discipline']}",
                width=wrap_width,
                subsequent_indent=" " * (len(str(row["code"])) + 2),
                break_long_words=False,
            )
            line_count = text.count("\n") + 1
            ax.text(
                x_pos,
                y_pos,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=legend_font_size,
                fontweight="bold",
                color="#1F2933",
            )
            y_pos -= line_count * 0.045 + 0.028

    fig.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.04)
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

    apply_style(float(plot_cfg.get("font_scale", 1.35)))
    table = load_subdiscipline_tables(cfg["inputs"]["eval_dirs"])
    averaged = average_asr_by_subdiscipline(table)
    subject_order = resolve_subject_order(averaged, plot_cfg.get("subject_order"))
    averaged["Subject"] = pd.Categorical(averaged["Subject"], categories=subject_order, ordered=True)
    averaged = averaged.sort_values(["Subject", "mean_asr"], ascending=[True, False])
    averaged.to_csv(output_dir / "subdiscipline_average_asr_all.csv", index=False)
    plot_df = build_plot_table(averaged, plot_cfg)
    plot_df.drop(columns=["x"]).to_csv(output_dir / "subdiscipline_code_legend.csv", index=False)

    plot_subdiscipline_all(
        averaged,
        output_dir / "subdiscipline_average_asr_all",
        formats,
        dpi,
        plot_cfg,
    )
    plot_code_legend(
        plot_df,
        output_dir / "subdiscipline_code_legend",
        formats,
        dpi,
        plot_cfg,
    )

    print(f"[SAVED] subdiscipline all outputs under {output_dir}")


if __name__ == "__main__":
    main()
