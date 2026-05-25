"""
Thailand-Cambodia Escalation Risk Dashboard
============================================
Multi-signal precursor detection model with interactive dashboard.

Analytical methods:
  1. Category-shift detection    - civilian targeting as violence precursor
  2. Cross-correlation scoring   - lag analysis
  3. IQR-based anomaly detection - robust to heavy tails
  4. Temporal momentum           - acceleration + jerk (2nd derivative)
  5. Volatility regime detection - 6mo vs 24mo rolling std
  6. Violence-protest divergence - suppression indicator
  7. Geographic clustering       - displacement event clustering
  8. Composite 8-signal score    - weighted sigmoid normalization

Data sources:
  - 6 ACLED monthly xlsx (demonstrations, civilian targeting, political violence)
  - 2 IDMC displacement CSVs (lat/lng event-level)
  - Gemini API for AI analyst narrative (optional — set GEMINI_API_KEY env var)

Usage:
    export GEMINI_API_KEY=your_key_here   # optional
    python3 risk_dashboard.py

Output:
    risk_dashboard_outputs/risk_dashboard.html
"""

from __future__ import annotations

import os
import json
import base64
import html as html_mod
from pathlib import Path

import pandas as pd
import numpy as np

# Optional Gemini dependency
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DATASETS = {
    "thai_demo":    ("thailand_demonstration_events_by_month-year_as-of-11mar2026.xlsx",         "Demonstrations"),
    "thai_civil":   ("thailand_civilian_targeting_events_and_fatalities_by_month-year_as-of-11mar2026.xlsx", "Civilian Targeting"),
    "thai_violence":("thailand_political_violence_events_and_fatalities_by_month-year_as-of-11mar2026.xlsx", "Political Violence"),
    "cam_demo":     ("cambodia_demonstration_events_by_month-year_as-of-11mar2026.xlsx",          "Demonstrations"),
    "cam_civil":    ("cambodia_civilian_targeting_events_and_fatalities_by_month-year_as-of-11mar2026.xlsx", "Civilian Targeting"),
    "cam_violence": ("cambodia_political_violence_events_and_fatalities_by_month-year_as-of-11mar2026.xlsx", "Political Violence"),
}

EVENT_CSVS = [
    "event_data_tha.csv",
    "event_data_khm.csv",
]

OUT_DIR = Path("risk_dashboard_outputs")
OUT_DIR.mkdir(exist_ok=True)
DASHBOARD_PATH = OUT_DIR / "risk_dashboard.html"

# Signal weights for composite risk score
SIGNAL_WEIGHTS = {
    "cat_shift":    0.15,
    "intensity":    0.15,
    "anomaly_iqr":  0.12,
    "acceleration": 0.15,
    "jerk":         0.08,
    "vol_ratio":    0.10,
    "fat_accel":    0.15,
    "divergence":   0.10,
}


# ──────────────────────────────────────────────
# GEMINI SETUP
# ──────────────────────────────────────────────

def configure_gemini() -> bool:
  
    if genai is None:
        return False
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  Warning: GEMINI_API_KEY not set. AI summary will be skipped.")
        print("  To enable: export GEMINI_API_KEY=your_key_here")
        return False
    genai.configure(api_key=api_key)
    return True


# ──────────────────────────────────────────────
# MATH HELPERS
# ──────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard sigmoid function, maps any real value to (0, 1)."""
    return 1 / (1 + np.exp(-x))


def minmax(s: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1]. Returns zeros if constant."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_acled_data() -> pd.DataFrame:
    """
    Load and concatenate all six ACLED monthly xlsx files.
    Returns a DataFrame with columns: country, date, events, fatalities, category.
    """
    frames = []
    for name, (path, category) in DATASETS.items():
        print(f"  Reading: {path}")
        df = pd.read_excel(path, sheet_name="Data")
        df.columns = [c.strip().lower() for c in df.columns]
        if "fatalities" not in df.columns:
            df["fatalities"] = 0
        df["date"] = pd.to_datetime(
            df["month"].astype(str) + " " + df["year"].astype(str),
            format="mixed", dayfirst=False,
        )
        df["category"] = category
        frames.append(df[["country", "date", "events", "fatalities", "category"]])
    return pd.concat(frames, ignore_index=True).sort_values("date")


def load_event_csvs() -> tuple[list[dict], list[dict]]:
    """
    Load IDMC displacement CSVs and expand each row into per-location points.
    Also computes geographic clusters on a 0.5-degree grid.

    Returns:
        expanded: list of individual displacement event dicts with lat/lng
        clusters: list of aggregated cluster dicts (0.5-degree grid cells)
    """
    frames = []
    for path in EVENT_CSVS:
        p = Path(path)
        if p.exists():
            print(f"  Reading events: {path}")
            frames.append(pd.read_csv(path))
        else:
            print(f"  Warning: {path} not found, skipping")

    if not frames:
        return [], []

    all_events = pd.concat(frames, ignore_index=True)
    expanded = []

    for _, row in all_events.iterrows():
        names  = str(row.get("locations_name", "")).split("; ")
        coords = str(row.get("locations_coordinates", "")).split("; ")
        n = min(len(names), len(coords))
        if n == 0:
            continue
        for i in range(n):
            try:
                lat_s, lng_s = coords[i].split(", ")
                lat, lng = float(lat_s), float(lng_s)
                desc = str(row.get("description", ""))
                if len(desc) > 180:
                    desc = desc[:177] + "..."
                expanded.append({
                    "country":    row["country"],
                    "lat":        round(lat, 5),
                    "lng":        round(lng, 5),
                    "location":   names[i].strip(),
                    "figure":     int(row.get("figure", 0)) // max(n, 1),
                    "date":       str(row.get("displacement_date", "")),
                    "event_type": str(row.get("combined_type", "Unknown")),
                    "desc":       desc,
                })
            except (ValueError, TypeError):
                continue

    print(f"  Expanded to {len(expanded)} map points")

    # Geographic clustering on a 0.5-degree grid
    pts_df = pd.DataFrame(expanded) if expanded else pd.DataFrame()
    clusters = []
    if len(pts_df) > 0:
        pts_df["lat_bin"] = (pts_df["lat"] * 2).round() / 2
        pts_df["lng_bin"] = (pts_df["lng"] * 2).round() / 2
        for (country, lat_b, lng_b), grp in pts_df.groupby(["country", "lat_bin", "lng_bin"]):
            clusters.append({
                "country":         country,
                "lat":             round(float(lat_b), 1),
                "lng":             round(float(lng_b), 1),
                "n_events":        len(grp),
                "total_displaced": int(grp["figure"].sum()),
                "locations":       "; ".join(grp["location"].unique()),
                "types":           "; ".join(grp["event_type"].unique()),
            })
        print(f"  Geographic clusters: {len(clusters)}")

    return expanded, clusters


# ──────────────────────────────────────────────
# PRECURSOR DETECTION MODEL
# ──────────────────────────────────────────────

def build_precursor_model(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute all 8 precursor signals for Thailand and Cambodia.

    Signals:
      1. cat_shift    - civilian targeting / (demonstrations + 1), 3mo smoothed
      2. intensity    - fatalities per event, 3mo smoothed
      3. anomaly_iqr  - IQR-based outlier score vs 12mo rolling window
      4. acceleration - 1st derivative of 3mo event sum
      5. jerk         - 2nd derivative (change in acceleration)
      6. vol_ratio    - 6mo vs 24mo volatility ratio
      7. fat_accel    - 3mo fatality acceleration
      8. divergence   - violence z-score minus protest z-score

    Returns:
        Dict mapping country name to DataFrame of signals indexed by date.
    """
    results = {}

    for country in ["Thailand", "Cambodia"]:
        c = raw[raw["country"] == country]

        cat_pivot = c.pivot_table(
            index="date", columns="category",
            values="events", aggfunc="sum", fill_value=0,
        ).sort_index()

        total = c.groupby("date").agg(
            events=("events", "sum"),
            fatalities=("fatalities", "sum"),
        ).sort_index()

        civil    = cat_pivot.get("Civilian Targeting", pd.Series(0, index=cat_pivot.index))
        demos    = cat_pivot.get("Demonstrations",     pd.Series(0, index=cat_pivot.index))
        violence = cat_pivot.get("Political Violence", pd.Series(0, index=cat_pivot.index))

        # Signal 1: Category shift (civilian targeting relative to protests)
        cat_shift = (civil / (demos + 1)).rolling(3, min_periods=1).mean()

        # Signal 2: Violence intensity (fatalities per event)
        intensity = (total["fatalities"] / total["events"].replace(0, np.nan)).fillna(0)
        intensity_3m = intensity.rolling(3, min_periods=1).mean()

        # Signal 3: IQR anomaly (robust to heavy-tailed conflict data)
        roll_med = total["events"].rolling(12, min_periods=3).median()
        roll_q1  = total["events"].rolling(12, min_periods=3).quantile(0.25)
        roll_q3  = total["events"].rolling(12, min_periods=3).quantile(0.75)
        iqr      = roll_q3 - roll_q1
        anomaly  = ((total["events"] - roll_med) / iqr.replace(0, np.nan)).replace(
            [np.inf, -np.inf], 0
        ).fillna(0)

        # Signal 4: Event acceleration (1st derivative of 3mo rolling sum)
        events_3m = total["events"].rolling(3, min_periods=1).sum()
        prev_3m   = events_3m.shift(3).fillna(0)
        accel     = events_3m - prev_3m

        # Signal 5: Jerk (2nd derivative — rate of change in acceleration)
        jerk = accel - accel.shift(1).fillna(0)

        # Signal 6: Volatility regime (short-term vs long-term instability)
        vol_6m  = total["events"].rolling(6,  min_periods=2).std().fillna(0)
        vol_24m = total["events"].rolling(24, min_periods=6).std().fillna(vol_6m)
        vol_ratio = (vol_6m / vol_24m.replace(0, np.nan)).replace(
            [np.inf, -np.inf], 1
        ).fillna(1)

        # Signal 7: Fatality acceleration
        fat_3m   = total["fatalities"].rolling(3, min_periods=1).sum()
        fat_accel = fat_3m - fat_3m.shift(3).fillna(0)

        # Signal 8: Violence-protest divergence (suppression indicator)
        demo_z = ((demos - demos.rolling(12, min_periods=3).mean())
                  / demos.rolling(12, min_periods=3).std().replace(0, np.nan)).fillna(0)
        viol_z = ((violence - violence.rolling(12, min_periods=3).mean())
                  / violence.rolling(12, min_periods=3).std().replace(0, np.nan)).fillna(0)
        divergence = viol_z - demo_z

        results[country] = pd.DataFrame({
            "events":       total["events"],
            "fatalities":   total["fatalities"],
            "cat_shift":    cat_shift,
            "intensity":    intensity_3m,
            "anomaly_iqr":  anomaly,
            "acceleration": accel,
            "jerk":         jerk,
            "vol_ratio":    vol_ratio,
            "fat_accel":    fat_accel,
            "divergence":   divergence,
        })

    return results


def compute_composite_risk(signals_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Combine 8 normalized signals into a composite risk score via weighted sigmoid.
    Also flags months where multiple leading indicators converge (precursor alert).

    Risk zones:
        < 0.30  → Low
        0.30-0.50 → Elevated
        0.50-0.70 → High
        >= 0.70 → Critical

    Returns:
        Dict mapping country name to DataFrame with risk scores and zone labels.
    """
    results = {}

    for country, df in signals_dict.items():
        r = df.copy()

        # Normalize each signal to [0, 1]
        for col in SIGNAL_WEIGHTS:
            r[f"n_{col}"] = minmax(r[col])

        # Weighted sum → sigmoid → final risk score
        r["risk_raw"]   = sum(SIGNAL_WEIGHTS[c] * r[f"n_{c}"] for c in SIGNAL_WEIGHTS)
        r["risk_score"] = sigmoid(5 * (r["risk_raw"] - 0.35))

        # Risk zone classification
        r["risk_zone"] = "Low"
        r.loc[r["risk_score"] >= 0.30, "risk_zone"] = "Elevated"
        r.loc[r["risk_score"] >= 0.50, "risk_zone"] = "High"
        r.loc[r["risk_score"] >= 0.70, "risk_zone"] = "Critical"

        # Precursor alert: three leading indicators converging simultaneously
        r["precursor_alert"] = (
            (r["cat_shift"]  > r["cat_shift"].rolling(6, min_periods=2).mean())
            & (r["vol_ratio"] > 1.2)
            & (r["acceleration"] > 0)
        ).fillna(False)

        results[country] = r

    return results


# ──────────────────────────────────────────────
# OVERVIEW PANEL (5-signal summary)
# ──────────────────────────────────────────────

def build_basic_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the simplified 5-signal panel used by overview charts.
    Also returns per-category monthly breakdown for stacked bar charts.

    Returns:
        monthly:     DataFrame with monthly totals + risk scores
        cat_monthly: DataFrame with per-category monthly breakdown
    """
    monthly = raw.groupby(["country", "date"], as_index=False).agg(
        events=("events", "sum"), fatalities=("fatalities", "sum"),
    ).sort_values(["country", "date"])

    cat_monthly = raw.groupby(["country", "date", "category"], as_index=False).agg(
        events=("events", "sum"), fatalities=("fatalities", "sum"),
    )

    g = monthly.groupby("country")
    monthly["inc_3m"]    = g["events"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    monthly["fat_3m"]    = g["fatalities"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    monthly["prev_3m"]   = g["inc_3m"].shift(3).fillna(0)
    monthly["acceleration"] = monthly["inc_3m"] - monthly["prev_3m"]

    roll_mean = g["events"].rolling(6, min_periods=2).mean().reset_index(0, drop=True)
    roll_std  = g["events"].rolling(6, min_periods=2).std().reset_index(0, drop=True).fillna(0)
    monthly["anomaly"] = ((monthly["events"] - roll_mean) / roll_std.replace(0, np.nan)).replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    # Naive weighted forecast: 50% current, 30% lag-1, 20% lag-2
    monthly["forecast"] = (
        0.5 * monthly["events"]
        + 0.3 * g["events"].shift(1).fillna(0)
        + 0.2 * g["events"].shift(2).fillna(0)
    )

    for col in ["inc_3m", "acceleration", "anomaly", "fat_3m", "forecast"]:
        monthly[f"n_{col}"] = minmax(monthly[col])

    monthly["risk_raw"] = (
        0.30 * monthly["n_inc_3m"]
        + 0.20 * monthly["n_acceleration"]
        + 0.15 * monthly["n_anomaly"]
        + 0.20 * monthly["n_fat_3m"]
        + 0.15 * monthly["n_forecast"]
    )
    monthly["risk_score"] = sigmoid(4 * (monthly["risk_raw"] - 0.35))

    return monthly, cat_monthly


# ──────────────────────────────────────────────
# SERIALIZATION
# ──────────────────────────────────────────────

def last_14_months(df):
    """Filter a DataFrame or dict of DataFrames to the most recent 14 months."""
    if isinstance(df, dict):
        out = {}
        for k, v in df.items():
            max_date = v.index.max()
            cutoff   = max_date - pd.DateOffset(months=13)
            out[k]   = v[v.index >= cutoff].copy()
        return out
    max_date = df["date"].max()
    cutoff   = max_date - pd.DateOffset(months=13)
    return df[df["date"] >= cutoff].copy()


def to_panel_json(recent: pd.DataFrame) -> list[dict]:
    """Serialize the overview panel DataFrame to a list of JSON-safe dicts."""
    recent = recent.copy()
    recent["date_str"] = recent["date"].dt.strftime("%Y-%m")
    records = []
    for _, row in recent.iterrows():
        records.append({
            "country":      row["country"],
            "date":         row["date_str"],
            "events":       int(row["events"]),
            "fatalities":   int(row["fatalities"]),
            "risk_score":   round(float(row["risk_score"]), 4),
            "risk_raw":     round(float(row["risk_raw"]), 4),
            "inc_3m":       round(float(row["inc_3m"]), 1),
            "acceleration": round(float(row["acceleration"]), 1),
            "anomaly":      round(float(row["anomaly"]), 2),
            "forecast":     round(float(row["forecast"]), 1),
        })
    return records


def to_cat_json(recent_cat: pd.DataFrame) -> list[dict]:
    """Serialize the per-category breakdown to a list of JSON-safe dicts."""
    recent_cat = recent_cat.copy()
    recent_cat["date_str"] = recent_cat["date"].dt.strftime("%Y-%m")
    records = []
    for _, row in recent_cat.iterrows():
        records.append({
            "country":    row["country"],
            "date":       row["date_str"],
            "category":   row["category"],
            "events":     int(row["events"]),
            "fatalities": int(row["fatalities"]),
        })
    return records


def to_precursor_json(risk_dict_14m: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    """Serialize the 8-signal precursor model results to JSON-safe dicts."""
    out = {}
    for country, df in risk_dict_14m.items():
        records = []
        for dt, row in df.iterrows():
            records.append({
                "date":             dt.strftime("%Y-%m"),
                "risk_score":       round(float(row["risk_score"]), 4),
                "risk_zone":        str(row["risk_zone"]),
                "cat_shift":        round(float(row["cat_shift"]), 4),
                "intensity":        round(float(row["intensity"]), 4),
                "anomaly_iqr":      round(float(row["anomaly_iqr"]), 2),
                "acceleration":     round(float(row["acceleration"]), 1),
                "jerk":             round(float(row["jerk"]), 1),
                "vol_ratio":        round(float(row["vol_ratio"]), 3),
                "fat_accel":        round(float(row["fat_accel"]), 1),
                "divergence":       round(float(row["divergence"]), 3),
                "precursor_alert":  bool(row["precursor_alert"]),
            })
        out[country] = records
    return out


# ──────────────────────────────────────────────
# AI ANALYST SUMMARY
# ──────────────────────────────────────────────

def generate_ai_summary(
    panel_json: list[dict],
    precursor_json: dict,
    map_points: list[dict],
    geo_clusters: list[dict],
) -> str:
    """
    Generate an AI analyst briefing via the Gemini API.
    Returns a placeholder string if the API key is not configured.
    """
    if not configure_gemini():
        return (
            "AI analyst summary unavailable.\n\n"
            "To enable: set the GEMINI_API_KEY environment variable "
            "and re-run the dashboard."
        )

    prompt = (
        "You are a geopolitical risk analyst. Analyze escalation risk between "
        "Thailand and Cambodia using the data below.\n\n"
        f"Monthly panel (last 14 months):\n{json.dumps(panel_json[:6], indent=2)}...\n\n"
        f"Precursor model signals (last 3 months each country):\n"
        f"Thailand: {json.dumps(precursor_json.get('Thailand', [])[-3:], indent=2)}\n"
        f"Cambodia: {json.dumps(precursor_json.get('Cambodia', [])[-3:], indent=2)}\n\n"
        f"Displacement events: {len(map_points)} locations, "
        f"total displaced: {sum(p.get('figure', 0) for p in map_points):,}\n\n"
        f"Geographic clusters: {len(geo_clusters)} zones, "
        f"border zone events: {sum(1 for c in geo_clusters if 12 <= c['lat'] <= 16 and 101 <= c['lng'] <= 107)}\n\n"
        "Write a concise analyst briefing:\n"
        "1. Bottom line up front\n"
        "2. Key precursor signals and what they indicate\n"
        "3. Violence trend assessment\n"
        "4. Displacement impact\n"
        "5. Geographic concentration analysis\n"
        "6. 90-day forecast with confidence level\n\n"
        "Be specific with numbers. Plain English."
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"


# ──────────────────────────────────────────────
# HTML TEMPLATE (base64-encoded frontend)
# ──────────────────────────────────────────────

_HTML_TEMPLATE_B64 = (
    "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVU"
    "Ri04Ij4KPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwg"
    "aW5pdGlhbC1zY2FsZT0xLjAiPgo8dGl0bGU+VGhhaWxhbmTigJNDYW1ib2RpYSBFc2NhbGF0aW9u"
    "IFJpc2sgRGFzaGJvYXJkPC90aXRsZT4KPGxpbmsgaHJlZj0iaHR0cHM6Ly9mb250cy5nb29nbGVh"
    "cGlzLmNvbS9jc3MyP2ZhbWlseT1ETStTYW5zOndnaHRANDAwOzUwMDs3MDAmZmFtaWx5PUpldEJy"
    "YWlucytNb25vOndnaHRANDAwOzUwMCZkaXNwbGF5PXN3YXAiIHJlbD0ic3R5bGVzaGVldCI+Cjxs"
    "aW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9h"
    "amF4L2xpYnMvbGVhZmxldC8xLjkuNC9sZWFmbGV0Lm1pbi5jc3MiIC8+CjxzdHlsZT4KICA6cm9v"
    "dCB7CiAgICAtLWJnOiAjMGEwZTE3OwogICAgLS1zdXJmYWNlOiAjMTExODI3OwogICAgLS1zdXJm"
    "YWNlMjogIzFhMjIzNTsKICAgIC0tYm9yZGVyOiAjMWUyOTNiOwogICAgLS10ZXh0OiAjZTJlOGYw"
    "OwogICAgLS10ZXh0LWRpbTogIzk0YTNiODsKICAgIC0tdGV4dC1tdXRlZDogIzY0NzQ4YjsKICAg"
    "IC0tYWNjZW50OiAjM2I4MmY2OwogICAgLS1kYW5nZXI6ICNlZjQ0NDQ7CiAgICAtLWRhbmdlci1z"
    "b2Z0OiAjOTkxYjFiOwogICAgLS13YXJuaW5nOiAjZjU5ZTBiOwogICAgLS13YXJuaW5nLXNvZnQ6"
    "ICM5MjQwMGU7CiAgICAtLXNhZmU6ICMxMGI5ODE7CiAgICAtLXNhZmUtc29mdDogIzA2NWY0NjsK"
    "ICAgIC0tdGhhaWxhbmQ6ICNmOTczMTY7CiAgICAtLWNhbWJvZGlhOiAjOGI1Y2Y2OwogIH0KCiAg"
    "KiB7IG1hcmdpbjogMDsgcGFkZGluZzogMDsgYm94LXNpemluZzogYm9yZGVyLWJveDsgfQoKICBi"
    "b2R5IHsKICAgIGJhY2tncm91bmQ6IHZhcigtLWJnKTsKICAgIGNvbG9yOiB2YXIoLS10ZXh0KTsK"
    "ICAgIGZvbnQtZmFtaWx5OiAnRE0gU2FucycsIHNhbnMtc2VyaWY7CiAgICBsaW5lLWhlaWdodDog"
    "MS42OwogICAgb3ZlcmZsb3cteDogaGlkZGVuOwogIH0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4K"
    "PHA+RGFzaGJvYXJkIHRlbXBsYXRlIGxvYWRlZC48L3A+CjwvYm9keT4KPC9odG1sPgo="
)


def get_html_template() -> str:
    return base64.b64decode(_HTML_TEMPLATE_B64).decode("utf-8")


# ──────────────────────────────────────────────
# DASHBOARD BUILDER
# ──────────────────────────────────────────────

def build_dashboard(
    panel_json:     list[dict],
    cat_json:       list[dict],
    map_points:     list[dict],
    geo_clusters:   list[dict],
    precursor_json: dict,
    ai_summary:     str,
) -> None:
    """
    Inject all data into the HTML template and write the final dashboard file.
    Uses __PLACEHOLDER__ tokens so no eval() or exec() is needed.
    """
    html = get_html_template()
    html = html.replace("__PANEL_DATA__",     json.dumps(panel_json))
    html = html.replace("__CAT_DATA__",       json.dumps(cat_json))
    html = html.replace("__MAP_POINTS__",     json.dumps(map_points))
    html = html.replace("__GEO_CLUSTERS__",   json.dumps(geo_clusters))
    html = html.replace("__PRECURSOR_DATA__", json.dumps(precursor_json))
    html = html.replace("__AI_SUMMARY__",     html_mod.escape(ai_summary))
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"  Dashboard saved: {DASHBOARD_PATH.resolve()}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 56)
    print("  Thailand-Cambodia Escalation Risk Dashboard")
    print("  8-Signal Precursor Detection Model")
    print("=" * 56)

    print("\n[1/7] Loading ACLED datasets...")
    raw = load_acled_data()

    print("\n[2/7] Loading IDMC displacement CSVs...")
    map_points, geo_clusters = load_event_csvs()

    print("\n[3/7] Building precursor detection model (8 signals)...")
    signals      = build_precursor_model(raw)
    risk_results = compute_composite_risk(signals)

    for country, df in risk_results.items():
        latest = df.iloc[-1]
        peak   = df["risk_score"].max()
        alerts = df.tail(14)["precursor_alert"].sum()
        print(f"  {country}: current={latest['risk_score']:.3f} ({latest['risk_zone']}), "
              f"14mo_peak={peak:.3f}, precursor_alerts={alerts}")

    print("\n[4/7] Building overview panel...")
    monthly, cat_monthly = build_basic_panel(raw)

    print("\n[5/7] Filtering to last 14 months & serializing...")
    recent      = last_14_months(monthly)
    recent_cat  = last_14_months(cat_monthly)
    risk_14m    = last_14_months(risk_results)

    panel_json     = to_panel_json(recent)
    cat_json       = to_cat_json(recent_cat)
    precursor_json = to_precursor_json(risk_14m)

    print("\n[6/7] Generating AI analysis (Gemini)...")
    ai_summary = generate_ai_summary(panel_json, precursor_json, map_points, geo_clusters)
    print(f"  AI summary: {len(ai_summary)} chars")

    print("\n[7/7] Building dashboard...")
    build_dashboard(panel_json, cat_json, map_points, geo_clusters,
                    precursor_json, ai_summary)

    print("\n" + "=" * 56)
    print("  Done! Open risk_dashboard_outputs/risk_dashboard.html")
    print("=" * 56)


if __name__ == "__main__":
    main()