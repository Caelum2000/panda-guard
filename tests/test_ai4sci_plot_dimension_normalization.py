import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from data.ai4sci_paper.normalize_202608 import (
    DISCIPLINE_MAPPING,
    RISK_DIMENSION_MAPPING,
    SUBDISCIPLINE_MAPPING,
    normalize_dimension_name,
    normalize_dimension_series,
)
from experiments.ai4sci_paper_draw_new.discipline_compare_radar.discipline_compare_radar import (
    average_group_asr,
    load_subject_tables,
)
from experiments.ai4sci_paper_draw_new.risk_heatmap_compare.risk_heatmap_compare import (
    load_risk_tables as load_heatmap_risk_tables,
)
from experiments.ai4sci_paper_draw_new.risk_heatmap_compare.risk_heatmap_compare import pivot_risk_heatmap
from experiments.ai4sci_paper_draw_new.risk_radar_compare.risk_radar_compare import (
    load_risk_tables as load_radar_risk_tables,
)
from experiments.ai4sci_paper_draw_new.risk_radar_compare.risk_radar_compare import pivot_asr
from experiments.ai4sci_paper_draw_new.subdiscipline_all.subdiscipline_all import (
    average_asr_by_subdiscipline,
    load_subdiscipline_tables,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRAW_1_2_PATH = REPO_ROOT / "experiments/ai4sci_paper/1_2_draw/ai4sci_paper_1_2_draw.py"
DRAW_1_2_SPEC = importlib.util.spec_from_file_location("ai4sci_paper_1_2_draw", DRAW_1_2_PATH)
assert DRAW_1_2_SPEC is not None and DRAW_1_2_SPEC.loader is not None
DRAW_1_2_MODULE = importlib.util.module_from_spec(DRAW_1_2_SPEC)
DRAW_1_2_SPEC.loader.exec_module(DRAW_1_2_MODULE)
load_eval_tables = DRAW_1_2_MODULE.load_eval_tables


@pytest.mark.parametrize(
    ("path", "column", "mapping"),
    [
        ("data/ai4sci_paper/20260423.csv", "Subject", DISCIPLINE_MAPPING),
        ("data/ai4sci_paper/20260423.csv", "Sub-discipline", SUBDISCIPLINE_MAPPING),
        ("data/ai4sci_paper/20260423.csv", "Risk Dimension", RISK_DIMENSION_MAPPING),
        ("data/ai4sci_paper/202608.csv", "discipline", DISCIPLINE_MAPPING),
        ("data/ai4sci_paper/202608.csv", "sub-discipline", SUBDISCIPLINE_MAPPING),
        ("data/ai4sci_paper/202608.csv", "risk dimension", RISK_DIMENSION_MAPPING),
    ],
)
def test_mappings_cover_both_source_taxonomies(path: str, column: str, mapping: dict[str, str]) -> None:
    values = pd.read_csv(REPO_ROOT / path, usecols=[column])[column]

    normalized = normalize_dimension_series(values, mapping, column)

    assert len(normalized) == len(values)
    assert normalized.notna().all()


@pytest.mark.parametrize(
    ("value", "mapping", "expected"),
    [
        (" Hallucinations AND misconceptions\n", RISK_DIMENSION_MAPPING, "Hallucinations and Misconceptions"),
        ("laboratory safety", RISK_DIMENSION_MAPPING, "Lab Safety"),
        ("privacy leakage", RISK_DIMENSION_MAPPING, "Privacy Leaks"),
        ("geopolitical sensitivity", RISK_DIMENSION_MAPPING, "Geopolitical Sensitive"),
        ("math", DISCIPLINE_MAPPING, "Mathematics"),
        ("MATHEMATICS", DISCIPLINE_MAPPING, "Mathematics"),
        (
            "Anatomy, physiology, and neuroscience",
            SUBDISCIPLINE_MAPPING,
            "Anatomy, Physiology & Neuroscience",
        ),
        ("applied & numerical mathematics", SUBDISCIPLINE_MAPPING, "Applied & Numerical Mathematics"),
    ],
)
def test_known_variants_have_one_canonical_name(value: str, mapping: dict[str, str], expected: str) -> None:
    assert normalize_dimension_name(value, mapping, "test dimension") == expected


def test_unmapped_values_are_reported_together() -> None:
    values = pd.Series(["Astronomy", "unexpected field", "another field"], name="Subject")

    with pytest.raises(ValueError) as exc_info:
        normalize_dimension_series(values, DISCIPLINE_MAPPING)

    message = str(exc_info.value)
    assert "Subject" in message
    assert "unexpected field" in message
    assert "another field" in message
    assert "normalize_202608.py" in message


def _write_csv(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> str:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("loader", [load_radar_risk_tables, load_heatmap_risk_tables])
def test_risk_loaders_merge_aliases_before_aggregation(tmp_path: Path, loader) -> None:
    path = _write_csv(
        tmp_path,
        "risk.csv",
        [
            {"model_name": "model-a", "Risk Dimension": "Hallucinations and Misconceptions", "total": 3, "attacked": 2},
            {
                "model_name": "model-a",
                "Risk Dimension": " hallucinations AND misconceptions\n",
                "total": 4,
                "attacked": 3,
            },
        ],
    )
    table = loader({"base": {"risk_dimension_margin": path}})

    if loader is load_radar_risk_tables:
        pivot = pivot_asr(table)
        assert list(pivot.columns) == ["Hallucinations and Misconceptions"]
        asr = pivot.loc["model-a", "Hallucinations and Misconceptions"]
    else:
        pivot = pivot_risk_heatmap(table, ["model-a"])
        assert list(pivot.index) == ["Hallucinations and Misconceptions"]
        asr = pivot.loc["Hallucinations and Misconceptions", "model-a"]

    assert asr == pytest.approx(5 / 7 * 100)


def test_discipline_loader_merges_math_aliases_before_aggregation(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "subjects.csv",
        [
            {"model_name": "model-a", "Subject": "math", "total": 10, "attacked": 4},
            {"model_name": "model-a", "Subject": "MATHEMATICS", "total": 10, "attacked": 6},
        ],
    )
    table = load_subject_tables({"base": {"subject_margin": path}})

    averaged = average_group_asr(table)

    assert averaged["Subject"].tolist() == ["Mathematics"]
    assert averaged.loc[0, "total"] == 20
    assert averaged.loc[0, "attacked"] == 10
    assert averaged.loc[0, "mean_asr"] == pytest.approx(50.0)


def test_subdiscipline_loader_merges_legacy_and_new_labels(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "subdisciplines.csv",
        [
            {
                "model_name": "model-a",
                "Subject": "math",
                "Sub-discipline": "Applied & numerical mathematics",
                "total": 10,
                "attacked": 4,
            },
            {
                "model_name": "model-a",
                "Subject": "mathematics",
                "Sub-discipline": "applied and numerical mathematics",
                "total": 10,
                "attacked": 6,
            },
        ],
    )
    table = load_subdiscipline_tables({"base": {"subject_subdiscipline": path}})

    averaged = average_asr_by_subdiscipline(table)

    assert averaged[["Subject", "Sub-discipline"]].to_dict(orient="records") == [
        {"Subject": "Mathematics", "Sub-discipline": "Applied & Numerical Mathematics"}
    ]
    assert averaged.loc[0, "total"] == 20
    assert averaged.loc[0, "attacked"] == 10
    assert averaged.loc[0, "mean_asr"] == pytest.approx(50.0)


def test_legacy_1_2_draw_loader_normalizes_every_dimension_table(tmp_path: Path) -> None:
    common = {"model_name": "model-a", "total": 10, "attacked": 5}
    subject_path = _write_csv(
        tmp_path,
        "all-subjects.csv",
        [{**common, "Subject": "math"}, {**common, "Subject": "mathematics"}],
    )
    subdiscipline_path = _write_csv(
        tmp_path,
        "all-subdisciplines.csv",
        [
            {**common, "Subject": "math", "Sub-discipline": "Applied & numerical mathematics"},
            {
                **common,
                "Subject": "mathematics",
                "Sub-discipline": "applied and numerical mathematics",
            },
        ],
    )
    risk_path = _write_csv(
        tmp_path,
        "all-risks.csv",
        [
            {**common, "Risk Dimension": "Hallucinations and Misconceptions"},
            {**common, "Risk Dimension": "hallucinations and misconceptions"},
        ],
    )

    tables = load_eval_tables(
        {
            "base": {
                "subject_margin": subject_path,
                "subject_subdiscipline": subdiscipline_path,
                "risk_dimension_margin": risk_path,
            }
        }
    )

    assert tables["subject_margin"]["Subject"].unique().tolist() == ["Mathematics"]
    assert tables["subject_subdiscipline"]["Subject"].unique().tolist() == ["Mathematics"]
    assert tables["subject_subdiscipline"]["Sub-discipline"].unique().tolist() == [
        "Applied & Numerical Mathematics"
    ]
    assert tables["risk_dimension_margin"]["Risk Dimension"].unique().tolist() == [
        "Hallucinations and Misconceptions"
    ]
