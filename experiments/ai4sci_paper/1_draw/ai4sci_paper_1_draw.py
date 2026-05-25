#!/usr/bin/env python3
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


MODEL_ALIASES = {
    "deepseekv4pro": "deepseekv4pro",
    "deepseekv4promax": "deepseekv4pro",
    "deepseekv4flash": "deepseekv4flash",
    "deepseekv4flashmax": "deepseekv4flash",
    "deepseekv32": "deepseekv32speciale",
    "deepseekv32speciale": "deepseekv32speciale",
    "kimi-k2-0905": "kimik2",
    "kimik20905": "kimik2",
    "kimik26": "kimik26",
    "kimi-k26": "kimik26",
    "mimov25pro": "xiaomimimov25pro",
    "xiaomimimov25pro": "xiaomimimov25pro",
    "nemotron3super": "nvidianemotron3super120ba12b",
    "nvidianemotron3super": "nvidianemotron3super120ba12b",
    "nvidianemotron3super120ba12b": "nvidianemotron3super120ba12b",
    "gemini3flash": "gemini3flashpreview",
    "gemini3flashpreview": "gemini3flashpreview",
    "gemma431b": "gemma431bit",
    "gemma431bit": "gemma431bit",
    "claudeopus47": "claudeopus47",
    "claudesonnet46": "claudesonnet46",
    "claudehaiku45": "claudehaiku45",
    "qwen36maxpreview": "qwen36maxpreview",
    "qwen3max": "qwen3max",
    "qwen3max20260123": "qwen3max",
    "qwen35397ba17b": "qwen35397ba17b",
    "qwen35122ba10b": "qwen35122ba10b",
    "qwen3535ba3b": "qwen3535ba3b",
    "qwen3527b": "qwen3527b",
    "s1baselite": "s1baselite",
    "s1base8b": "s1baselite",
    "s1basepro": "s1basepro",
    "s1base32b": "s1basepro",
    "s1baseultra": "s1baseultra",
    "s1base671b": "s1baseultra",
    "interns1pro": "interns1pro",
    "interns1": "interns1",
    "interns1mini": "interns1mini",
}


def normalize_model_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("&", "and")
    key = re.sub(r"[^a-z0-9]+", "", text)
    return MODEL_ALIASES.get(key, key)


def slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def parse_param_count(value: object) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value).replace(",", "")
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([bBmM])", text)
    if not matches:
        numeric = pd.to_numeric(text, errors="coerce")
        return float(numeric) if not pd.isna(numeric) else math.nan
    number, unit = matches[0]
    scale = 1.0 if unit.lower() == "b" else 0.001
    return float(number) * scale


def load_eval_tables(eval_dirs: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    tables = {"subject_margin": [], "subject_subdiscipline": [], "risk_dimension_margin": []}
    for group_name, paths in eval_dirs.items():
        for table_name, csv_path in paths.items():
            df = pd.read_csv(csv_path)
            df["model_group"] = group_name
            df["model_key"] = df["model_name"].map(normalize_model_name)
            tables[table_name].append(df)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}


def compute_model_asr(subject_margin: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        subject_margin.groupby(["model_key", "model_name", "model_group"], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    grouped["asr"] = grouped["attacked"] / grouped["total"] * 100.0
    return grouped.sort_values(["model_group", "asr", "model_name"], ascending=[True, False, True])


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
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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
    expected_suffix = Path("outputs/ai4sci_paper/1_draw")
    if expected_suffix not in [Path(*resolved.parts[i:]) for i in range(len(resolved.parts))]:
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


def join_release_metadata(model_asr: pd.DataFrame, release_path: str) -> pd.DataFrame:
    meta = pd.read_csv(release_path)
    meta = meta.dropna(subset=["Model Name"]).copy()
    meta["model_key"] = meta["Model Name"].map(normalize_model_name)
    meta["release_date"] = pd.to_datetime(meta["Release Date"], errors="coerce")
    meta["parameters_b"] = meta["Parameter Count"].map(parse_param_count)
    meta = meta.drop_duplicates(subset=["model_key"], keep="first")
    return model_asr.merge(
        meta[["model_key", "Model Name", "release_date", "parameters_b"]],
        on="model_key",
        how="inner",
    )


def annotate_points(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, label_col: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            str(row[label_col]),
            (row[x_col], row[y_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.8,
            fontweight="bold",
            alpha=0.78,
        )


def plot_release_scatter(df: pd.DataFrame, output_dir: Path, formats: Iterable[str], dpi: int) -> None:
    df = df.dropna(subset=["release_date", "asr"]).sort_values("release_date")
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    sns.scatterplot(
        data=df,
        x="release_date",
        y="asr",
        hue="model_group",
        style="model_group",
        size="parameters_b",
        sizes=(45, 150),
        edgecolor="white",
        linewidth=0.75,
        palette={"base": "#2F6F9F", "sci": "#C7503C"},
        ax=ax,
    )
    annotate_points(ax, df, "release_date", "asr", "model_name")
    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR vs. Model Release Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.set_ylim(0, min(100, max(20, df["asr"].max() + 10)))
    ax.legend(frameon=False, loc="best", fontsize=9)
    save_figure(fig, output_dir / "asr_vs_release_date", formats, dpi)


def pivot_asr(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    weighted = (
        df.groupby(["model_name", category_col], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    weighted["asr"] = weighted["attacked"] / weighted["total"] * 100.0
    pivot = weighted.pivot(index="model_name", columns=category_col, values="asr")
    return pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    cell_width: float,
    cell_height: float,
) -> None:
    width = max(8.0, min(24.0, pivot.shape[1] * cell_width + 3.2))
    height = max(4.8, min(18.0, pivot.shape[0] * cell_height + 1.8))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=100,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "ASR (%)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    save_figure(fig, output_base, formats, dpi)


def radar_categories(pivot: pd.DataFrame) -> List[str]:
    return pivot.mean(axis=0).sort_values(ascending=False).index.tolist()


def plot_radar(
    pivot: pd.DataFrame,
    title: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    top_n_models: int,
) -> None:
    if pivot.empty:
        return
    ordered_cols = radar_categories(pivot)
    radar_df = pivot[ordered_cols].fillna(0)
    radar_df = radar_df.loc[radar_df.mean(axis=1).sort_values(ascending=False).head(top_n_models).index]

    labels = radar_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw={"polar": True})
    palette = sns.color_palette("tab10", n_colors=len(radar_df))
    for color, (model_name, values) in zip(palette, radar_df.iterrows()):
        series = values.tolist()
        series += series[:1]
        ax.plot(angles, series, linewidth=2.1, label=model_name, color=color)
        ax.fill(angles, series, alpha=0.08, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9, fontweight="bold")
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)
    save_figure(fig, output_base, formats, dpi)


def average_asr_by_category(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    per_model = (
        df.groupby(["model_name", category_col], as_index=False)
        .agg(total=("total", "sum"), attacked=("attacked", "sum"))
    )
    per_model["asr"] = per_model["attacked"] / per_model["total"] * 100.0
    averaged = (
        per_model.groupby(category_col, as_index=False)
        .agg(mean_asr=("asr", "mean"), std_asr=("asr", "std"), model_count=("model_name", "nunique"))
        .sort_values("mean_asr", ascending=False)
    )
    return averaged


def plot_average_bar(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    averaged = average_asr_by_category(df, category_col)
    ensure_dir(output_base.parent)
    averaged.to_csv(output_base.with_suffix(".csv"), index=False)

    height = max(4.2, min(12.0, len(averaged) * 0.34 + 1.4))
    fig, ax = plt.subplots(figsize=(9.2, height))
    sns.barplot(
        data=averaged,
        x="mean_asr",
        y=category_col,
        color="#3B78A8",
        edgecolor="#1F3D54",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Average Attack Success Rate (%)")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xlim(0, min(100, max(20, averaged["mean_asr"].max() + 8)))
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=10, fontweight="bold")
    save_figure(fig, output_base, formats, dpi)


def plot_grouped_matrix_figures(
    df: pd.DataFrame,
    category_col: str,
    prefix: str,
    readable_name: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    top_radar_models: int,
    cell_width: float,
    cell_height: float,
    skip_heatmap_groups: Iterable[str] = (),
) -> None:
    skip_heatmap_groups = set(skip_heatmap_groups)
    for group_name, group_df in [("all", df), *sorted(df.groupby("model_group"), key=lambda x: x[0])]:
        group_label = "All Models" if group_name == "all" else f"{group_name.capitalize()} Models"
        pivot = pivot_asr(group_df, category_col)
        ensure_dir(output_dir)
        pivot.to_csv(output_dir / f"{prefix}_{group_name}_matrix.csv")
        if group_name not in skip_heatmap_groups:
            plot_heatmap(
                pivot,
                f"{readable_name} ASR Heatmap ({group_label})",
                output_dir / f"{prefix}_{group_name}_heatmap",
                formats,
                dpi,
                cell_width,
                cell_height,
            )
        plot_radar(
            pivot,
            f"{readable_name} ASR Radar ({group_label})",
            output_dir / f"{prefix}_{group_name}_radar",
            formats,
            dpi,
            top_radar_models,
        )


def plot_subdiscipline_by_subject(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    top_radar_models: int,
    cell_width: float,
    cell_height: float,
) -> None:
    subject_root = output_dir / "subdiscipline_by_subject"
    for subject, subject_df in sorted(df.groupby("Subject"), key=lambda x: str(x[0]).lower()):
        subject_dir = subject_root / slugify(subject)
        plot_grouped_matrix_figures(
            subject_df,
            "Sub-discipline",
            "subdiscipline_model",
            f"{subject} Sub-discipline by Model",
            subject_dir,
            formats,
            dpi,
            top_radar_models,
            cell_width,
            cell_height,
        )


def plot_average_bar_figures(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    bar_dir = output_dir / "average_bar_charts"
    plot_average_bar(
        tables["subject_margin"],
        "Subject",
        "Average ASR by Scientific Domain",
        bar_dir / "subject_average_asr_bar",
        formats,
        dpi,
    )
    plot_average_bar(
        tables["risk_dimension_margin"],
        "Risk Dimension",
        "Average ASR by Risk Dimension",
        bar_dir / "risk_dimension_average_asr_bar",
        formats,
        dpi,
    )
    subject_root = bar_dir / "subdiscipline_by_subject"
    for subject, subject_df in sorted(tables["subject_subdiscipline"].groupby("Subject"), key=lambda x: str(x[0]).lower()):
        plot_average_bar(
            subject_df,
            "Sub-discipline",
            f"Average ASR by {subject} Sub-discipline",
            subject_root / slugify(subject) / "subdiscipline_average_asr_bar",
            formats,
            dpi,
        )


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
    formats = plot_cfg.get("formats", ["png", "pdf"])
    dpi = int(plot_cfg.get("dpi", 300))
    top_radar_models = int(plot_cfg.get("top_radar_models", 8))
    cell_width = float(plot_cfg.get("heatmap_cell_width", 0.34))
    cell_height = float(plot_cfg.get("heatmap_cell_height", 0.32))

    apply_style()
    tables = load_eval_tables(cfg["inputs"]["eval_dirs"])
    model_asr = compute_model_asr(tables["subject_margin"])
    model_asr.to_csv(output_dir / "model_asr.csv", index=False)

    plot_average_bar_figures(tables, output_dir, formats, dpi)

    release_df = join_release_metadata(model_asr, cfg["inputs"]["model_release"])
    release_df.to_csv(output_dir / "model_asr_with_release.csv", index=False)

    plot_release_scatter(release_df, output_dir, formats, dpi)

    plot_grouped_matrix_figures(
        tables["subject_margin"],
        "Subject",
        "subject_model",
        "Subject by Model",
        output_dir,
        formats,
        dpi,
        top_radar_models,
        cell_width,
        cell_height,
    )
    plot_subdiscipline_by_subject(
        tables["subject_subdiscipline"],
        output_dir,
        formats,
        dpi,
        top_radar_models,
        cell_width,
        cell_height,
    )
    plot_grouped_matrix_figures(
        tables["risk_dimension_margin"],
        "Risk Dimension",
        "risk_dimension_model",
        "Risk Dimension by Model",
        output_dir,
        formats,
        dpi,
        top_radar_models,
        cell_width,
        cell_height,
        skip_heatmap_groups={"base"},
    )

    print(f"[SAVED] figures and analysis tables under {output_dir}")


if __name__ == "__main__":
    main()
