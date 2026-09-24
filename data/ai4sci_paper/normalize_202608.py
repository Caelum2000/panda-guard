"""Canonical dimension names shared by the AI4Sci plotting programs.

The 20260423 and 202608 datasets describe the same taxonomy with different
capitalization, punctuation, and, in a few cases, different words.  Plotting
code must normalize labels before grouping so equivalent rows are aggregated
together instead of appearing as duplicate axes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import pandas as pd


def _identity_mapping(names: tuple[str, ...]) -> dict[str, str]:
    return {name: name for name in names}


RISK_DIMENSION_MAPPING: dict[str, str] = _identity_mapping(
    (
        "Safety Omission",
        "Knowledge Cutoff Drift",
        "Hallucinations and Misconceptions",
        "Geopolitical Sensitive",
        "Authority Inflation",
        "Fringe Amplification",
        "Privacy Leaks",
        "Dual-use",
        "Lab Safety",
        "Regulatory Blind Spot",
        "Compliance Neglect",
    )
)
RISK_DIMENSION_MAPPING.update(
    {
        "Geopolitical Sensitivity": "Geopolitical Sensitive",
        "Privacy Leakage": "Privacy Leaks",
        "Laboratory Safety": "Lab Safety",
    }
)


DISCIPLINE_MAPPING: dict[str, str] = _identity_mapping(
    (
        "Astronomy",
        "Biology",
        "Chemistry",
        "Engineering",
        "Geography",
        "Mathematics",
        "Physics",
    )
)
DISCIPLINE_MAPPING["math"] = "Mathematics"


SUBDISCIPLINE_MAPPING: dict[str, str] = _identity_mapping(
    (
        "Observational Astronomy & Instrumentation",
        "Planetary Science & Cosmology",
        "Space Exploration & Orbital Mechanics",
        "Stellar Astrophysics",
        "Anatomy, Physiology & Neuroscience",
        "Biochemistry",
        "Bioinformatics & Synthetic Biology",
        "Ecology & Evolutionary Biology",
        "Molecular & Developmental Biology",
        "Pathogens, Toxins & Pharmacology",
        "Analytical Chemistry",
        "Inorganic & Coordination Chemistry",
        "Organic Synthesis",
        "Cartography & GIS",
        "Climatology & Meteorology",
        "Human & Political Geography",
        "Physical Geography & Geomorphology",
        "Urban & Economic Geography",
        "Chemical & Process Engineering",
        "Electrical & Electronic Engineering",
        "Mechanical & Manufacturing Engineering",
        "Software & Systems Engineering",
        "Structural & Civil Engineering",
        "Applied & Numerical Mathematics",
        "Pure Mathematics",
        "Statistics & Probability",
        "Classical Mechanics & Dynamics",
        "Electromagnetism & Optics",
        "Nuclear & Particle Physics",
        "Quantum Mechanics",
        "Thermodynamics & Statistical Mechanics",
    )
)


def dimension_key(value: Any) -> str:
    """Return a comparison key insensitive to cosmetic label differences."""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _normalized_lookup(mapping: Mapping[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for alias, canonical in mapping.items():
        key = dimension_key(alias)
        previous = lookup.get(key)
        if previous is not None and previous != canonical:
            raise ValueError(f"Conflicting canonical labels for {alias!r}: {previous!r} and {canonical!r}")
        lookup[key] = canonical
    return lookup


def normalize_dimension_name(value: Any, mapping: Mapping[str, str], dimension_name: str) -> str:
    """Normalize one label, raising an actionable error for unknown values."""

    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"Unmapped {dimension_name} value: <missing>")

    lookup = _normalized_lookup(mapping)
    key = dimension_key(value)
    try:
        return lookup[key]
    except KeyError as exc:
        raise ValueError(f"Unmapped {dimension_name} value: {value!r}") from exc


def normalize_dimension_series(
    series: pd.Series,
    mapping: Mapping[str, str],
    dimension_name: str | None = None,
) -> pd.Series:
    """Normalize a label column and report all unmapped values at once."""

    label = dimension_name or str(series.name or "dimension")
    lookup = _normalized_lookup(mapping)
    normalized: list[str | None] = []
    unmapped: list[str] = []

    for value in series.tolist():
        if pd.isna(value) or not str(value).strip():
            normalized.append(None)
            unmapped.append("<missing>")
            continue

        canonical = lookup.get(dimension_key(value))
        normalized.append(canonical)
        if canonical is None:
            unmapped.append(repr(value))

    if unmapped:
        values = ", ".join(sorted(set(unmapped)))
        raise ValueError(f"Unmapped values in {label}: {values}. Add aliases to data/ai4sci_paper/normalize_202608.py")

    return pd.Series(normalized, index=series.index, name=series.name, dtype="object")
