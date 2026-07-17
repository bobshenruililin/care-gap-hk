#!/usr/bin/env python3
"""Assemble district-level analysis dataset for The Care Gap paper."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DISTRICT_ORDER = [
    "Central & Western",
    "Wan Chai",
    "Eastern",
    "Southern",
    "Yau Tsim Mong",
    "Sham Shui Po",
    "Kowloon City",
    "Wong Tai Sin",
    "Kwun Tong",
    "Kwai Tsing",
    "Tsuen Wan",
    "Tuen Mun",
    "Yuen Long",
    "North",
    "Tai Po",
    "Sha Tin",
    "Sai Kung",
    "Islands",
]

# Paper Table 1 (C&SD Table 110-06833, 2024)
DATA_2024 = {
    "Central & Western": {"r_incl": 452, "r_excl": 517, "income_10k": 5.15, "floor_m2": 34.0},
    "Wan Chai": {"r_incl": 475, "r_excl": 566, "income_10k": 5.50, "floor_m2": 30.0},
    "Eastern": {"r_incl": 549, "r_excl": 603, "income_10k": 4.02, "floor_m2": 35.0},
    "Southern": {"r_incl": 506, "r_excl": 580, "income_10k": 4.12, "floor_m2": 37.0},
    "Yau Tsim Mong": {"r_incl": 429, "r_excl": 463, "income_10k": 3.50, "floor_m2": 27.0},
    "Sham Shui Po": {"r_incl": 496, "r_excl": 525, "income_10k": 3.00, "floor_m2": 30.0},
    "Kowloon City": {"r_incl": 482, "r_excl": 539, "income_10k": 3.83, "floor_m2": 33.0},
    "Wong Tai Sin": {"r_incl": 516, "r_excl": 536, "income_10k": 3.00, "floor_m2": 35.0},
    "Kwun Tong": {"r_incl": 502, "r_excl": 522, "income_10k": 2.91, "floor_m2": 35.0},
    "Kwai Tsing": {"r_incl": 506, "r_excl": 526, "income_10k": 2.96, "floor_m2": 37.0},
    "Tsuen Wan": {"r_incl": 473, "r_excl": 510, "income_10k": 4.01, "floor_m2": 37.0},
    "Tuen Mun": {"r_incl": 489, "r_excl": 509, "income_10k": 3.18, "floor_m2": 40.0},
    "Yuen Long": {"r_incl": 412, "r_excl": 435, "income_10k": 3.28, "floor_m2": 39.0},
    "North": {"r_incl": 463, "r_excl": 479, "income_10k": 3.02, "floor_m2": 41.0},
    "Tai Po": {"r_incl": 500, "r_excl": 540, "income_10k": 3.72, "floor_m2": 41.0},
    "Sha Tin": {"r_incl": 502, "r_excl": 537, "income_10k": 3.45, "floor_m2": 40.0},
    "Sai Kung": {"r_incl": 417, "r_excl": 451, "income_10k": 4.50, "floor_m2": 47.0},
    "Islands": {"r_incl": 406, "r_excl": 429, "income_10k": 3.50, "floor_m2": 50.0},
}

# 2021 GHS district profiles (B1130301 2021): (r_incl, r_excl)
DEP_2021 = [
    (410, 476),
    (449, 539),
    (485, 530),
    (449, 505),
    (406, 443),
    (455, 480),
    (461, 514),
    (456, 474),
    (471, 488),
    (457, 473),
    (414, 442),
    (420, 436),
    (355, 374),
    (394, 408),
    (404, 431),
    (454, 482),
    (371, 399),
    (375, 401),
]

# Female LFPR excl. FDH from 2021 district profiles (Overall by sex, female bracket)
FEMALE_LFPR_EXCL_2021 = [
    54.0,
    50.2,
    48.6,
    50.3,
    53.0,
    49.2,
    48.9,
    47.8,
    47.2,
    48.2,
    51.8,
    49.9,
    50.0,
    47.5,
    50.2,
    49.0,
    52.4,
    52.7,
]

# Table 1, B1130301 2021: % of 1-person + 2-person households
H_FRAG_2021 = {
    "Central & Western": 26.5 + 29.1,
    "Wan Chai": 27.5 + 30.9,
    "Eastern": 18.0 + 29.8,
    "Southern": 14.8 + 28.1,
    "Yau Tsim Mong": 28.1 + 30.7,
    "Sham Shui Po": 24.9 + 29.2,
    "Kowloon City": 21.8 + 28.5,
    "Wong Tai Sin": 18.8 + 28.3,
    "Kwun Tong": 19.6 + 29.6,
    "Kwai Tsing": 18.5 + 28.5,
    "Tsuen Wan": 16.4 + 31.5,
    "Tuen Mun": 20.1 + 30.6,
    "Yuen Long": 18.8 + 27.8,
    "North": 18.8 + 30.1,
    "Tai Po": 16.3 + 29.8,
    "Sha Tin": 16.2 + 31.0,
    "Sai Kung": 15.4 + 27.9,
    "Islands": 20.9 + 28.0,
}

# Mid-year LBNP 2024 ('000) and % aged 65+ (Tables 1.1–1.2, B1130301 2024 summary)
POP_2024 = {
    "Central & Western": {"pop_k": 229.4, "pct_0_14": 10.0, "pct_65": 21.2},
    "Wan Chai": {"pop_k": 162.0, "pct_0_14": 8.9, "pct_65": 23.3},
    "Eastern": {"pop_k": 514.4, "pct_0_14": 8.7, "pct_65": 26.7},
    "Southern": {"pop_k": 254.7, "pct_0_14": 9.7, "pct_65": 23.9},
    "Yau Tsim Mong": {"pop_k": 299.7, "pct_0_14": 10.7, "pct_65": 19.3},
    "Sham Shui Po": {"pop_k": 432.3, "pct_0_14": 11.2, "pct_65": 21.9},
    "Kowloon City": {"pop_k": 412.5, "pct_0_14": 10.7, "pct_65": 21.9},
    "Wong Tai Sin": {"pop_k": 406.7, "pct_0_14": 8.2, "pct_65": 25.8},
    "Kwun Tong": {"pop_k": 662.4, "pct_0_14": 9.3, "pct_65": 24.1},
    "Kwai Tsing": {"pop_k": 491.6, "pct_0_14": 9.2, "pct_65": 24.4},
    "Tsuen Wan": {"pop_k": 306.2, "pct_0_14": 10.6, "pct_65": 21.5},
    "Tuen Mun": {"pop_k": 531.0, "pct_0_14": 10.6, "pct_65": 22.2},
    "Yuen Long": {"pop_k": 671.1, "pct_0_14": 10.6, "pct_65": 18.6},
    "North": {"pop_k": 338.4, "pct_0_14": 10.4, "pct_65": 21.2},
    "Tai Po": {"pop_k": 327.9, "pct_0_14": 10.6, "pct_65": 22.7},
    "Sha Tin": {"pop_k": 698.9, "pct_0_14": 10.4, "pct_65": 23.1},
    "Sai Kung": {"pop_k": 498.2, "pct_0_14": 10.6, "pct_65": 18.8},
    "Islands": {"pop_k": 195.3, "pct_0_14": 11.8, "pct_65": 17.1},
}

# SWD subsidised RCHE capacity by district, 31.3.2024 (PDF extract)
RCHE_SUB_2024 = {
    "Eastern": 1189,
    "Wan Chai": 790,
    "Central & Western": 1126,
    "Islands": 447,
    "Southern": 2077,
    "Sham Shui Po": 1793,
    "Kowloon City": 2585,
    "Yau Tsim Mong": 1272,
    "Wong Tai Sin": 1983,
    "Sai Kung": 1277,
    "Kwun Tong": 2487,
    "Sha Tin": 1901,
    "Tai Po": 1654,
    "North": 2444,
    "Yuen Long": 2085,
    "Tuen Mun": 2109,
    "Tsuen Wan": 2015,
    "Kwai Tsing": 3363,
}

# Fallbacks if Excel parse fails (filled from SWD 31.3.2026 workbooks)
RCHE_SUB_2026 = {
    "Eastern": 1206,
    "Wan Chai": 792,
    "Central & Western": 1131,
    "Islands": 437,
    "Southern": 2063,
    "Sham Shui Po": 1814,
    "Kowloon City": 2528,
    "Yau Tsim Mong": 1278,
    "Wong Tai Sin": 2005,
    "Sai Kung": 1337,
    "Kwun Tong": 2539,
    "Sha Tin": 1961,
    "Tai Po": 1665,
    "North": 2636,
    "Yuen Long": 2185,
    "Tuen Mun": 2118,
    "Tsuen Wan": 2050,
    "Kwai Tsing": 3479,
}

RCHE_NONSUB_2026 = {
    "Eastern": 3700,
    "Wan Chai": 1636,
    "Central & Western": 1603,
    "Islands": 416,
    "Southern": 1984,
    "Sham Shui Po": 4057,
    "Kowloon City": 4029,
    "Yau Tsim Mong": 3736,
    "Wong Tai Sin": 1896,
    "Sai Kung": 955,
    "Kwun Tong": 3162,
    "Sha Tin": 1932,
    "Tai Po": 1948,
    "North": 2747,
    "Yuen Long": 3120,
    "Tuen Mun": 3095,
    "Tsuen Wan": 2261,
    "Kwai Tsing": 4374,
}

HA_CLUSTER = {
    "Central & Western": "HKWC",
    "Wan Chai": "HKEC",
    "Eastern": "HKEC",
    "Southern": "HKWC",
    "Yau Tsim Mong": "KWC",
    "Sham Shui Po": "KWC",
    "Kowloon City": "KCC",
    "Wong Tai Sin": "KCC",
    "Kwun Tong": "KEC",
    "Kwai Tsing": "KWC",
    "Tsuen Wan": "NTW",
    "Tuen Mun": "NTW",
    "Yuen Long": "NTW",
    "North": "NTE",
    "Tai Po": "NTE",
    "Sha Tin": "NTE",
    "Sai Kung": "KEC",
    "Islands": "HKWC",
}


def load_rche_2026_from_excel() -> tuple[dict[str, int], dict[str, int]]:
    """Override hardcoded 2026 totals from Excel if available."""
    try:
        import openpyxl
    except ImportError:
        return RCHE_SUB_2026, RCHE_NONSUB_2026

    name_map = {
        "Eastern": "Eastern",
        "Wanchai": "Wan Chai",
        "Wan Chai": "Wan Chai",
        "Central & Western": "Central & Western",
        "Islands": "Islands",
        "Southern": "Southern",
        "Shamshuipo": "Sham Shui Po",
        "Sham Shui Po": "Sham Shui Po",
        "Kowloon City": "Kowloon City",
        "Yau Tsim Mong": "Yau Tsim Mong",
        "Wong Tai Sin": "Wong Tai Sin",
        "Sai Kung": "Sai Kung",
        "Kwun Tong": "Kwun Tong",
        "Shatin": "Sha Tin",
        "Sha Tin": "Sha Tin",
        "Tai Po": "Tai Po",
        "North": "North",
        "Yuen Long": "Yuen Long",
        "Tuen Mun": "Tuen Mun",
        "Tsuen Wan": "Tsuen Wan",
        "Kwai Tsing": "Kwai Tsing",
        "Kwai Chung": "Kwai Tsing",
    }

    def parse(path: Path, total_col_candidates: list[int]) -> dict[str, int]:
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        out: dict[str, int] = {}
        for row in ws.iter_rows(values_only=True):
            if not row or row[0] is None:
                continue
            raw = str(row[0]).strip().rstrip(":")
            key = None
            for k, v in name_map.items():
                if raw.lower().startswith(k.lower()):
                    key = v
                    break
            if key is None:
                continue
            total = None
            for c in total_col_candidates:
                if c < len(row) and isinstance(row[c], (int, float)):
                    total = int(row[c])
                    break
            if total is not None:
                out[key] = total
        return out

    sub = parse(ROOT / "data/raw/swd_sub_2026.xlsx", [8, 7])
    non = parse(ROOT / "data/raw/swd_nonsub_2026.xlsx", [7, 6])
    # Fill any missing from hardcoded
    for d in DISTRICT_ORDER:
        sub.setdefault(d, RCHE_SUB_2026.get(d, 0))
        non.setdefault(d, RCHE_NONSUB_2026.get(d, 0))
    return sub, non


def main() -> None:
    sub26, non26 = load_rche_2026_from_excel()

    rows = []
    for i, d in enumerate(DISTRICT_ORDER):
        r24 = DATA_2024[d]
        r21_incl, r21_excl = DEP_2021[i]
        pop = POP_2024[d]
        elderly = pop["pop_k"] * 1000 * pop["pct_65"] / 100.0
        children = pop["pop_k"] * 1000 * pop["pct_0_14"] / 100.0
        fw24 = r24["r_excl"] / r24["r_incl"] - 1.0
        fw21 = r21_excl / r21_incl - 1.0
        theta = children / (children + elderly)
        i_adj = (1 - theta) * fw24 * 100
        h_frag = H_FRAG_2021[d]
        rche_sub = RCHE_SUB_2024[d]
        rche_total = sub26[d] + non26[d]
        rows.append(
            {
                "district": d,
                "ha_cluster": HA_CLUSTER[d],
                "r_incl_2024": r24["r_incl"],
                "r_excl_2024": r24["r_excl"],
                "fw_2024_pct": fw24 * 100,
                "r_incl_2021": r21_incl,
                "r_excl_2021": r21_excl,
                "fw_2021_pct": fw21 * 100,
                "income_10k": r24["income_10k"],
                "floor_m2": r24["floor_m2"],
                "pop_2024": pop["pop_k"] * 1000,
                "pct_0_14_2024": pop["pct_0_14"],
                "pct_65_2024": pop["pct_65"],
                "elderly_2024": elderly,
                "children_2024": children,
                "theta_childcare": theta,
                "fw_adj_elder_pct": i_adj,
                "h_frag_2021_pct": h_frag,
                "female_lfpr_excl_2021": FEMALE_LFPR_EXCL_2021[i],
                "rche_sub_places_2024": rche_sub,
                "rche_sub_per_1000_elderly": rche_sub / elderly * 1000,
                "rche_total_places_2026": rche_total,
                "rche_total_per_1000_elderly": rche_total / elderly * 1000,
                "rche_sub_places_2026": sub26[d],
                "rche_nonsub_places_2026": non26[d],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "district_master.csv", index=False)

    meta = {
        "n_districts": len(df),
        "sources": {
            "dependency_2024": "C&SD Table 110-06833 (as compiled in paper)",
            "dependency_2021": "C&SD B1130301 2021 district profiles (incl/excl FDH)",
            "income_floor_2021": "2021 Population Census (paper Appendices A–B)",
            "population_age_2024": "C&SD B1130301 2024 summary Tables 1.1–1.2",
            "h_frag_2021": "C&SD B1130301 2021 Table 1 (1+2 person HH %)",
            "female_lfpr_2021": "C&SD B1130301 2021 district profiles (excl FDH)",
            "rche_sub_2024": "SWD subsidised RCHE capacity by district, 31.3.2024",
            "rche_2026": "SWD Excel capacity files, 31.3.2026",
        },
        "notes": [
            "Income/floor area are 2021 census; dependency is 2024 GHS — rank stability tested in analysis.",
            "RCHE places are capacity (stock), not admission flows; interpret as institutional supply exposure.",
            "Female LFPR excl. FDH is a proxy for unpaid/local care pressure, not direct carer counts.",
        ],
    }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(df.to_string(index=False))
    print("Wrote", OUT / "district_master.csv")


if __name__ == "__main__":
    main()
