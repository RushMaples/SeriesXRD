#!/usr/bin/env python3
"""Streamlit dashboard for browsing XRD correlation suite outputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - shown when launched with plain python.
    raise SystemExit(
        "Streamlit is not installed in this Python environment.\n"
        "Install it with:\n"
        "  /opt/anaconda3/envs/uotexrd/bin/pip install streamlit\n"
        "Then run:\n"
        "  /opt/anaconda3/envs/uotexrd/bin/streamlit run scripts/xrd_results_dashboard.py"
    ) from exc


DEFAULT_SUITE = Path("outputs/correlation_suite_20260621_high_recall_scored_v2")
DEFAULT_XDI = Path("outputs/xdi_pressure_window_maps_20260703")
PRESSURE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)GPa")


st.set_page_config(page_title="XRD Result Browser", layout="wide")


def app_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
        div[data-testid="stMetric"] { background: #f7f7f8; border: 1px solid #e6e6ea; padding: 0.6rem 0.8rem; border-radius: 6px; }
        div[data-testid="stDataFrame"] { border: 1px solid #e6e6ea; border-radius: 6px; }
        .xrd-path { color: #60646c; font-size: 0.86rem; margin-top: -0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_suite(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def resolve_local_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.set_index("frame")
    return df


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    return Path(path).read_text(errors="replace")


@st.cache_data(show_spinner=False)
def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def show_image(path: Path, caption: str | None = None) -> None:
    if not path.exists():
        st.warning(f"Missing image: {path}")
        return
    st.image(str(path), caption=caption or path.name, use_container_width=True)
    width, height = image_size(str(path))
    st.markdown(f"<div class='xrd-path'>{path} | {width} x {height}px</div>", unsafe_allow_html=True)


def pressure_key(label: str) -> tuple[float, str]:
    match = PRESSURE_RE.search(str(label))
    if not match:
        return (float("inf"), str(label))
    return (float(match.group("value")), str(label))


def sorted_frames(labels: list[str]) -> list[str]:
    return sorted(labels, key=pressure_key)


def list_pngs(folder: Path, pattern: str = "*.png") -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern), key=lambda path: path.name)


def folder_status(suite: Path) -> dict[str, Path]:
    return {
        "01_per_peak": suite / "01_per_peak_frame_correlation",
        "02_same_window": suite / "02_same_window_acf_across_frames",
        "03_single_frame": suite / "03_single_frame_window_acf",
        "04_all_window": suite / "04_all_window_to_all_window_acf",
        "05_static_overall": suite / "05_static_peaks_and_overall_correlation",
        "06_scientific_validation": suite / "06_scientific_peak_group_validation",
        "07_manual_review_links": suite / "07_manual_review_linked_table",
    }


def load_key_tables(suite: Path) -> dict[str, pd.DataFrame]:
    folders = folder_status(suite)
    tables: dict[str, pd.DataFrame] = {}
    candidates = {
        "per_peak_index": folders["01_per_peak"] / "per_peak_map_index.csv",
        "peak_groups": folders["01_per_peak"] / "peak_group_table.csv",
        "peak_group_summary": folders["01_per_peak"] / "peak_group_summary.csv",
        "peak_table": folders["01_per_peak"] / "peak_table.csv",
        "all_candidates": folders["01_per_peak"] / "all_candidate_table.csv",
        "tier_c_candidates": folders["01_per_peak"] / "tier_c_candidate_table.csv",
        "presence": folders["01_per_peak"] / "peak_presence_features.csv",
        "roi_area": folders["01_per_peak"] / "peak_roi_area_features.csv",
        "static_groups": folders["05_static_overall"] / "suspected_static_peak_groups.csv",
        "reliable_groups": folders["05_static_overall"] / "reliable_peak_groups.csv",
        "static_scores": folders["05_static_overall"] / "all_peak_static_scores.csv",
        "weights": folders["05_static_overall"] / "overall_component_weights.csv",
        "same_window_summary": folders["02_same_window"] / "same_window_summary.csv",
        "scientific_metrics": folders["06_scientific_validation"] / "all_group_scientific_metrics.csv",
        "scientific_recurring": folders["06_scientific_validation"] / "recurring_peak_groups.csv",
        "scientific_appearing": folders["06_scientific_validation"] / "appearing_disappearing_peak_groups.csv",
        "scientific_shifting": folders["06_scientific_validation"] / "shifting_peak_groups.csv",
        "scientific_isolated": folders["06_scientific_validation"] / "isolated_strong_peak_groups.csv",
        "scientific_static": folders["06_scientific_validation"] / "likely_static_background_groups.csv",
        "scientific_tier_compare": folders["06_scientific_validation"] / "tier_ab_vs_tier_a_comparison.csv",
        "scientific_linkages": folders["06_scientific_validation"] / "possible_group_linkages.csv",
        "scientific_shortlist": folders["06_scientific_validation"] / "manual_review_shortlist.csv",
        "manual_review_linked": folders["07_manual_review_links"] / "manual_review_linked_table.csv",
    }
    for name, path in candidates.items():
        if path.exists():
            tables[name] = read_csv(str(path))
    return tables


def overall_modes(folder: Path) -> list[str]:
    modes = []
    for path in sorted(folder.glob("overall_*_correlation_heatmap.png")):
        mode = path.name.removeprefix("overall_").removesuffix("_correlation_heatmap.png")
        modes.append(mode)
    preferred = ["reliable_static_removed", "static_removed", "reliable_peaks", "all_peaks"]
    ordered = [mode for mode in preferred if mode in modes]
    ordered.extend(mode for mode in modes if mode not in ordered)
    return ordered


def image_gallery(paths: list[Path], columns: int = 3) -> None:
    if not paths:
        st.info("No images found.")
        return
    cols = st.columns(columns)
    for index, path in enumerate(paths):
        with cols[index % columns]:
            st.image(str(path), caption=path.name, use_container_width=True)


def all_suite_pngs(suite: Path) -> list[Path]:
    return sorted(suite.rglob("*.png"), key=lambda path: str(path.relative_to(suite)))


def metric_row(items: list[tuple[str, object]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def overview_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("XRD Result Browser")
    st.caption("Local dashboard for correlation suite outputs.")

    folders = folder_status(suite)
    metric_row(
        [
            ("Peak groups", len(tables.get("per_peak_index", []))),
            ("Detected peaks", len(tables.get("peak_table", []))),
            ("Static candidates", len(tables.get("static_groups", []))),
            ("Same-window maps", len(list_pngs(folders["02_same_window"] / "heatmaps"))),
            ("Per-peak maps", len(list_pngs(folders["01_per_peak"] / "per_peak_heatmaps"))),
        ]
    )

    st.subheader("Suite folders")
    folder_df = pd.DataFrame(
        [
            {"section": name, "exists": path.exists(), "path": str(path)}
            for name, path in folders.items()
        ]
    )
    st.dataframe(folder_df, use_container_width=True, hide_index=True)

    readme = suite / "README.txt"
    if readme.exists():
        with st.expander("README", expanded=False):
            st.text(read_text(str(readme)))

    st.subheader("Quick views")
    quick = [
        folders["05_static_overall"] / "overall_reliable_static_removed_correlation_heatmap.png",
        folders["05_static_overall"] / "suspected_static_peak_summary.png",
        folders["04_all_window"] / "all_window_to_all_window_acf_heatmap.png",
    ]
    image_gallery([path for path in quick if path.exists()], columns=3)


def search_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Search")
    st.caption("Search CSV tables and PNG outputs inside the selected correlation suite.")

    query = st.text_input("Search text, pressure, 2theta, peak group, or filename", "")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        search_tables = st.checkbox("Search tables", value=True)
    with c2:
        search_images = st.checkbox("Search images", value=True)
    with c3:
        max_rows = st.slider("Rows/images per section", 5, 100, 25, 5)

    if not query.strip():
        st.info("Try examples: 7.4GPa, 10.603, peak_group_049, static, same_window.")
        return

    query_text = query.strip().lower()
    numeric_query: float | None = None
    try:
        numeric_query = float(query_text.replace("deg", "").strip())
    except ValueError:
        numeric_query = None

    if search_tables:
        st.subheader("Table matches")
        found_table = False
        for table_name, table in tables.items():
            if table.empty:
                continue
            text_mask = table.astype(str).apply(
                lambda column: column.str.lower().str.contains(query_text, regex=False, na=False)
            ).any(axis=1)
            numeric_mask = pd.Series(False, index=table.index)
            if numeric_query is not None:
                for column in table.columns:
                    numeric = pd.to_numeric(table[column], errors="coerce")
                    if numeric.notna().any():
                        numeric_mask |= (numeric - numeric_query).abs() <= 0.05
            matches = table[text_mask | numeric_mask].head(max_rows)
            if not matches.empty:
                found_table = True
                with st.expander(f"{table_name}: {len(matches)} shown", expanded=True):
                    st.dataframe(matches, use_container_width=True, hide_index=True)
        if not found_table:
            st.write("No table matches.")

    if search_images:
        st.subheader("Image matches")
        pngs = all_suite_pngs(suite)
        image_matches = [
            path
            for path in pngs
            if query_text in path.name.lower()
            or query_text in str(path.relative_to(suite)).lower()
        ]
        if numeric_query is not None:
            numeric_pattern = f"{numeric_query:.3f}"
            image_matches.extend(
                path
                for path in pngs
                if numeric_pattern in path.name or numeric_pattern in str(path.relative_to(suite))
            )
        image_matches = list(dict.fromkeys(image_matches))[:max_rows]
        if image_matches:
            result_df = pd.DataFrame(
                [{"file": path.name, "relative_path": str(path.relative_to(suite))} for path in image_matches]
            )
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            image_gallery(image_matches, columns=3)
        else:
            st.write("No image matches.")


def overall_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Overall Correlation")
    folder = folder_status(suite)["05_static_overall"]
    modes = overall_modes(folder)
    if not modes:
        st.warning("No overall correlation heatmaps found.")
        return
    mode = st.radio("Overall mode", modes, horizontal=True)
    heatmap = folder / f"overall_{mode}_correlation_heatmap.png"
    matrix_path = folder / f"overall_{mode}_correlation_matrix.csv"
    component_path = folder / f"overall_{mode}_pair_components.csv"

    left, right = st.columns([1.1, 0.9])
    with left:
        show_image(heatmap, f"Overall correlation: {mode}")
    with right:
        if "weights" in tables:
            st.subheader("Weights")
            st.dataframe(tables["weights"], use_container_width=True, hide_index=True)
        if component_path.exists():
            components = read_csv(str(component_path))
            st.subheader("Frame pair scores")
            sorted_components = components.sort_values("overall", ascending=False, na_position="last")
            st.dataframe(sorted_components, use_container_width=True, hide_index=True, height=430)

    if matrix_path.exists():
        st.subheader("Matrix")
        st.dataframe(read_matrix(str(matrix_path)), use_container_width=True)

    st.subheader("Component heatmaps")
    component_images = sorted(folder.glob(f"component_{mode}_*.png")) + sorted(folder.glob("component_*acf*.png"))
    image_gallery(component_images, columns=3)


def static_peaks_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Static / Diamond-like Peak Candidates")
    folder = folder_status(suite)["05_static_overall"]
    col1, col2 = st.columns(2)
    with col1:
        show_image(folder / "suspected_static_peak_summary.png", "Static peak summary")
    with col2:
        show_image(folder / "suspected_static_peak_positions_vs_pressure.png", "Static peak positions")

    if "static_groups" not in tables:
        st.warning("No suspected_static_peak_groups.csv found.")
        return

    static = tables["static_groups"].copy()
    all_scores = tables.get("static_scores", pd.DataFrame())
    min_coverage = st.slider("Minimum coverage fraction", 0.0, 1.0, 0.0, 0.05)
    max_mad = st.slider("Maximum position MAD shown", 0.0, 0.2, 0.2, 0.005)
    filtered = static[
        (static["coverage_fraction"] >= min_coverage)
        & (static["position_mad_deg"] <= max_mad)
    ]
    st.subheader("Suspected static peak groups")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    with st.expander("All peak static scores", expanded=False):
        st.dataframe(all_scores, use_container_width=True, hide_index=True)


def scientific_validation_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Scientific Peak Validation")
    folder = folder_status(suite)["06_scientific_validation"]
    if "scientific_metrics" not in tables:
        st.warning(
            "No scientific validation outputs found. Run: "
            "python scripts/scientific_peak_group_validation.py"
        )
        return

    metrics = tables["scientific_metrics"].copy()
    metric_row(
        [
            ("Groups analyzed", len(metrics)),
            ("Recurring", int(metrics["scientific_category_labels"].fillna("").str.contains("recurring").sum())),
            ("Shifting", int(metrics["scientific_category_labels"].fillna("").str.contains("systematic_shift").sum())),
            ("Static-like", int(metrics["scientific_category_labels"].fillna("").str.contains("likely_static_background").sum())),
            ("Manual shortlist", len(tables.get("scientific_shortlist", []))),
        ]
    )

    report = folder / "scientific_peak_group_report.md"
    if report.exists():
        with st.expander("Scientific report", expanded=False):
            st.markdown(read_text(str(report)))

    linked_folder = folder_status(suite)["07_manual_review_links"]
    linked_html = linked_folder / "manual_review_linked_table.html"
    linked_table = tables.get("manual_review_linked", pd.DataFrame())
    if linked_html.exists() or not linked_table.empty:
        st.subheader("Linked 1D/2D manual review table")
        if linked_html.exists():
            st.markdown(f"Clickable HTML table: `{linked_html}`")
        if not linked_table.empty:
            display_cols = [
                "peak_group",
                "selection_category",
                "frame",
                "group_two_theta",
                "two_theta",
                "confidence_tier",
                "source_methods",
                "raw_tiff",
                "path",
            ]
            st.dataframe(
                linked_table[[col for col in display_cols if col in linked_table.columns]],
                use_container_width=True,
                hide_index=True,
                height=240,
            )
            labels = [
                f"group {int(row.peak_group):03d} | {row.frame} | {row.group_two_theta:.4f} deg"
                for row in linked_table.itertuples()
            ]
            selected_linked = st.selectbox("Preview linked row", labels)
            idx = labels.index(selected_linked)
            linked_row = linked_table.iloc[idx]
            cols = st.columns(3)
            preview_specs = [
                ("xy_window", "1D local .xy window"),
                ("raw_ring_full", "2D raw ring overlay"),
                ("raw_ring_crop", "2D annular crop"),
            ]
            for col, (path_col, caption) in zip(cols, preview_specs):
                with col:
                    path = Path(str(linked_row.get(path_col, "")))
                    if path.exists():
                        show_image(path, caption)
                    else:
                        st.caption(f"{caption}: missing")

    st.subheader("Category plots")
    quick = [
        folder / "plots" / "category_summary.png",
        folder / "plots" / "peak_group_persistence.png",
        folder / "plots" / "peak_presence_map_tier_ab.png",
        folder / "plots" / "peak_presence_map_tier_a_only.png",
    ]
    image_gallery([path for path in quick if path.exists()], columns=2)

    st.subheader("Browse scientific groups")
    c1, c2, c3 = st.columns(3)
    with c1:
        categories = [
            "all",
            "recurring",
            "low_pressure_only",
            "high_pressure_only",
            "decompression_only",
            "systematic_shift",
            "isolated_strong",
            "likely_static_background",
            "tier_b_dependent",
            "uncertain",
        ]
        category = st.selectbox("Scientific category", categories)
        min_frames = st.slider("Minimum frame count", 1, int(metrics["frame_count"].max()), 1)
    with c2:
        pressure_min = float(pd.to_numeric(metrics["pressure_min"], errors="coerce").min())
        pressure_max = float(pd.to_numeric(metrics["pressure_max"], errors="coerce").max())
        pressure_range = st.slider("Pressure range", pressure_min, pressure_max, (pressure_min, pressure_max), 0.1)
        tier_mode = st.selectbox("Tier mode", ["A+B", "Tier A only emphasis", "Tier-B-dependent"])
    with c3:
        max_abs_slope = float(pd.to_numeric(metrics["pressure_position_slope_deg_per_gpa"], errors="coerce").abs().max())
        min_abs_slope = st.slider("Minimum abs slope", 0.0, max_abs_slope, 0.0, 0.0005)
        min_r2 = st.slider("Minimum R-squared", 0.0, 1.0, 0.0, 0.05)

    filtered = metrics.copy()
    filtered = filtered[filtered["frame_count"] >= min_frames]
    filtered = filtered[
        (pd.to_numeric(filtered["pressure_max"], errors="coerce") >= pressure_range[0])
        & (pd.to_numeric(filtered["pressure_min"], errors="coerce") <= pressure_range[1])
    ]
    if category != "all":
        filtered = filtered[filtered["scientific_category_labels"].fillna("").str.contains(category, regex=False)]
    if tier_mode == "Tier A only emphasis":
        filtered = filtered[pd.to_numeric(filtered["tier_a_count"], errors="coerce").fillna(0) > 0]
    elif tier_mode == "Tier-B-dependent":
        filtered = filtered[filtered["scientific_category_labels"].fillna("").str.contains("tier_b_dependent", regex=False)]
    filtered = filtered[
        pd.to_numeric(filtered["pressure_position_slope_deg_per_gpa"], errors="coerce").abs().fillna(0) >= min_abs_slope
    ]
    filtered = filtered[
        pd.to_numeric(filtered["position_fit_r_squared"], errors="coerce").fillna(0) >= min_r2
    ]

    if "suspected_static" in filtered.columns:
        static_choice = st.selectbox("Static score filter", ["all", "exclude static-like", "only static-like"])
        if static_choice == "exclude static-like":
            filtered = filtered[~filtered["suspected_static"].astype(bool)]
        elif static_choice == "only static-like":
            filtered = filtered[filtered["suspected_static"].astype(bool)]

    columns = [
        "peak_group",
        "median_2theta",
        "frame_count",
        "coverage_fraction",
        "longest_consecutive_run",
        "pressure_min",
        "pressure_max",
        "tier_a_count",
        "tier_b_count",
        "max_roi_area",
        "pressure_position_slope_deg_per_gpa",
        "position_fit_r_squared",
        "suspected_static",
        "scientific_category_labels",
        "notes",
    ]
    st.dataframe(filtered[[col for col in columns if col in filtered.columns]], use_container_width=True, hide_index=True, height=320)

    if not filtered.empty:
        choices = [
            f"group {int(row.peak_group):03d} | {row.median_2theta:.3f} deg | {row.scientific_category_labels}"
            for row in filtered.head(300).itertuples()
        ]
        selected = st.selectbox("Inspect group", choices)
        group_id = int(re.search(r"group\s+(\d+)", selected).group(1))
        row = metrics[metrics["peak_group"].astype(int) == group_id].iloc[0]
        left, right = st.columns([1.1, 0.9])
        with left:
            heatmap_match = tables.get("per_peak_index", pd.DataFrame())
            heatmap = None
            if not heatmap_match.empty:
                match = heatmap_match[heatmap_match["peak_group"].astype(int) == group_id]
                if not match.empty and "heatmap" in match.columns:
                    heatmap = folder_status(suite)["01_per_peak"] / str(match.iloc[0]["heatmap"])
            if heatmap:
                show_image(heatmap, "Per-peak correlation heatmap")
            contact = folder / "manual_review_contact_sheets" / f"group_{group_id:03d}_contact_sheet.png"
            if contact.exists():
                show_image(contact, "Manual-review local .xy contact sheet")
        with right:
            st.subheader("Full metric row")
            st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)
            if "peak_table" in tables:
                st.subheader("Detections by pressure")
                detections = tables["peak_table"][tables["peak_table"]["peak_group"].astype(int) == group_id]
                st.dataframe(detections, use_container_width=True, hide_index=True, height=280)

    st.subheader("Manual review shortlist")
    shortlist = tables.get("scientific_shortlist", pd.DataFrame())
    if not shortlist.empty:
        st.dataframe(shortlist, use_container_width=True, hide_index=True)
        contact_images = [
            Path(path)
            for path in shortlist.get("contact_sheet", pd.Series(dtype=str)).dropna().astype(str)
            if Path(path).exists()
        ]
        image_gallery(contact_images[:12], columns=2)


def per_peak_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Per-Peak Correlation Browser")
    folder = folder_status(suite)["01_per_peak"]
    if "per_peak_index" not in tables:
        st.warning("No per_peak_map_index.csv found.")
        return

    index = tables["per_peak_index"].copy()
    static_ids: set[int] = set()
    if "static_groups" in tables and "peak_group" in tables["static_groups"].columns:
        static_ids = set(tables["static_groups"]["peak_group"].astype(int).tolist())
    index["suspected_static"] = index["peak_group"].astype(int).isin(static_ids)

    c1, c2, c3 = st.columns([1, 1, 1])
    theta_min, theta_max = float(index["group_two_theta"].min()), float(index["group_two_theta"].max())
    with c1:
        theta_range = st.slider("2theta range", theta_min, theta_max, (theta_min, theta_max), 0.05)
    with c2:
        min_frames = st.slider("Minimum frame count", 1, int(index["frame_count"].max()), 1)
    with c3:
        static_filter = st.selectbox("Static filter", ["all", "exclude static", "only static"])

    search = st.text_input("Search peak group, 2theta, or frame label", "")
    filtered = index[
        (index["group_two_theta"] >= theta_range[0])
        & (index["group_two_theta"] <= theta_range[1])
        & (index["frame_count"] >= min_frames)
    ].copy()
    if static_filter == "exclude static":
        filtered = filtered[~filtered["suspected_static"]]
    elif static_filter == "only static":
        filtered = filtered[filtered["suspected_static"]]

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        tier_view = st.selectbox("Tier view", ["A+B/C optional", "A only", "contains B", "contains C"])
    with c5:
        available_sources = sorted(
            {
                source
                for text in index.get("source_methods", pd.Series(dtype=str)).dropna().astype(str)
                for source in text.split(";")
                if source
            }
        )
        source_filter = st.selectbox("Source method", ["all", *available_sources])
    with c6:
        available_scales = sorted(
            {
                scale
                for text in index.get("matched_filter_scales", pd.Series(dtype=str)).dropna().astype(str)
                for scale in text.split(";")
                if scale and scale != "nan"
            },
            key=lambda value: float(value),
        )
        scale_filter = st.selectbox("Matched scale", ["all", *available_scales])

    for column in ["tier_a_count", "tier_b_count", "tier_c_count"]:
        if column not in filtered.columns:
            filtered[column] = 0
    if tier_view == "A only":
        filtered = filtered[(filtered["tier_a_count"] > 0) & (filtered["tier_b_count"] == 0) & (filtered["tier_c_count"] == 0)]
    elif tier_view == "contains B":
        filtered = filtered[filtered["tier_b_count"] > 0]
    elif tier_view == "contains C":
        filtered = filtered[filtered["tier_c_count"] > 0]
    if source_filter != "all" and "source_methods" in filtered.columns:
        filtered = filtered[filtered["source_methods"].fillna("").astype(str).str.contains(source_filter, regex=False)]
    if scale_filter != "all" and "matched_filter_scales" in filtered.columns:
        filtered = filtered[filtered["matched_filter_scales"].fillna("").astype(str).str.split(";").apply(lambda values: scale_filter in values)]

    if search:
        text = search.lower()
        filtered = filtered[
            filtered.apply(lambda row: text in " ".join(str(value).lower() for value in row.values), axis=1)
        ]

    st.subheader("Peak groups")
    st.dataframe(
        filtered[
            [
                column
                for column in [
                    "peak_group",
                    "group_two_theta",
                    "frame_count",
                    "frame_coverage_fraction",
                    "max_roi_area",
                    "tier_a_count",
                    "tier_b_count",
                    "tier_c_count",
                    "source_methods",
                    "matched_filter_scales",
                    "suspected_static",
                    "frames_present",
                ]
                if column in filtered.columns
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=260,
    )
    if filtered.empty:
        return

    choices = [
        f"group {int(row.peak_group):03d} | {row.group_two_theta:.3f} deg | n={int(row.frame_count)}"
        for row in filtered.itertuples()
    ]
    selected_label = st.selectbox("Select peak group", choices)
    selected_group = int(re.search(r"group\s+(\d+)", selected_label).group(1))
    row = index[index["peak_group"].astype(int) == selected_group].iloc[0]

    left, right = st.columns([1.15, 0.85])
    with left:
        show_image(folder / str(row["heatmap"]), selected_label)
    with right:
        st.subheader("Group metadata")
        st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)
        matrix_path = folder / str(row["matrix"])
        if matrix_path.exists():
            st.subheader("Correlation matrix")
            st.dataframe(read_matrix(str(matrix_path)), use_container_width=True, height=360)

    if "peak_table" in tables:
        st.subheader("Detected peaks in this group")
        peaks = tables["peak_table"]
        group_peaks = peaks[peaks["peak_group"].astype(int) == selected_group]
        st.dataframe(group_peaks, use_container_width=True, hide_index=True)

    if "tier_c_candidates" in tables and not tables["tier_c_candidates"].empty:
        with st.expander("Tier C diagnostic candidates", expanded=False):
            st.dataframe(tables["tier_c_candidates"], use_container_width=True, hide_index=True, height=260)


def window_acf_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Window ACF Browser")
    folders = folder_status(suite)
    mode = st.radio(
        "View",
        ["same-window across frames", "single-frame windows", "all-window overview"],
        horizontal=True,
    )

    if mode == "same-window across frames":
        folder = folders["02_same_window"]
        if "same_window_summary" in tables:
            st.dataframe(tables["same_window_summary"], use_container_width=True, hide_index=True)
        images = list_pngs(folder / "heatmaps")
        selected = st.selectbox("Select same-window map", [path.name for path in images])
        path = folder / "heatmaps" / selected
        show_image(path)
        matrix_path = folder / "matrices" / selected.replace(".png", ".csv")
        if matrix_path.exists():
            st.dataframe(read_matrix(str(matrix_path)), use_container_width=True)

    elif mode == "single-frame windows":
        folder = folders["03_single_frame"]
        images = list_pngs(folder / "heatmaps")
        selected = st.selectbox("Select frame", [path.name for path in images])
        show_image(folder / "heatmaps" / selected)
        matrix_path = folder / "matrices" / selected.replace(".png", ".csv")
        if matrix_path.exists():
            st.dataframe(read_matrix(str(matrix_path)), use_container_width=True)

    else:
        folder = folders["04_all_window"]
        show_image(folder / "all_window_to_all_window_acf_heatmap.png")
        matrix_path = folder / "all_window_to_all_window_acf_matrix.csv"
        if matrix_path.exists():
            st.dataframe(read_matrix(str(matrix_path)), use_container_width=True, height=520)


def frame_pair_page(suite: Path, tables: dict[str, pd.DataFrame]) -> None:
    st.title("Frame Pair Inspector")
    folder = folder_status(suite)["05_static_overall"]
    modes = overall_modes(folder)
    if not modes:
        st.warning("No overall pair component files found.")
        return
    pair_file = folder / f"overall_{modes[0]}_pair_components.csv"
    if not pair_file.exists():
        st.warning(f"No {pair_file.name} found.")
        return

    pairs = read_csv(str(pair_file))
    frames = sorted_frames(sorted(set(pairs["frame_a"]).union(set(pairs["frame_b"]))))
    c1, c2, c3 = st.columns(3)
    with c1:
        frame_a = st.selectbox("Frame A", frames, index=0)
    with c2:
        frame_b = st.selectbox("Frame B", frames, index=min(1, len(frames) - 1))
    with c3:
        mode = st.radio("Peak set", modes, horizontal=True)

    pair_path = folder / f"overall_{mode}_pair_components.csv"
    pair_table = read_csv(str(pair_path))
    match = pair_table[
        ((pair_table["frame_a"] == frame_a) & (pair_table["frame_b"] == frame_b))
        | ((pair_table["frame_a"] == frame_b) & (pair_table["frame_b"] == frame_a))
    ]
    if match.empty:
        st.info("No pair entry for this combination.")
    else:
        row = match.iloc[0]
        metric_row(
            [
                ("Overall", f"{row['overall']:.3f}"),
                ("Presence", f"{row['peak_presence_jaccard']:.3f}"),
                ("Peak area", f"{row['peak_roi_area_cosine']:.3f}"),
                ("Same-window ACF", f"{row['same_window_acf_shifted_median']:.3f}"),
                ("All-window ACF", f"{row['all_window_acf_frame_median']:.3f}"),
            ]
        )
        st.dataframe(match, use_container_width=True, hide_index=True)

    show_pair_peak_differences(tables, frame_a, frame_b)


def show_pair_peak_differences(tables: dict[str, pd.DataFrame], frame_a: str, frame_b: str) -> None:
    if "presence" not in tables or "roi_area" not in tables:
        return
    presence = tables["presence"].set_index("frame")
    roi = tables["roi_area"].set_index("frame")
    if frame_a not in presence.index or frame_b not in presence.index:
        return

    static_groups = set()
    if "static_groups" in tables:
        static_groups = set(tables["static_groups"]["group_two_theta"].round(4).astype(str))

    values = []
    for col in presence.columns:
        present_a = bool(presence.loc[frame_a, col] > 0)
        present_b = bool(presence.loc[frame_b, col] > 0)
        area_a = float(roi.loc[frame_a, col]) if col in roi.columns else 0.0
        area_b = float(roi.loc[frame_b, col]) if col in roi.columns else 0.0
        values.append(
            {
                "2theta": float(col),
                "present_a": present_a,
                "present_b": present_b,
                "area_a": area_a,
                "area_b": area_b,
                "abs_area_delta": abs(area_a - area_b),
                "presence_class": "both" if present_a and present_b else "A only" if present_a else "B only" if present_b else "neither",
                "suspected_static": f"{float(col):.4f}" in static_groups,
            }
        )
    diff = pd.DataFrame(values)
    st.subheader("Peak differences")
    option = st.selectbox("Show", ["largest area differences", "A only", "B only", "both present", "suspected static"])
    if option == "A only":
        shown = diff[diff["presence_class"] == "A only"]
    elif option == "B only":
        shown = diff[diff["presence_class"] == "B only"]
    elif option == "both present":
        shown = diff[diff["presence_class"] == "both"]
    elif option == "suspected static":
        shown = diff[diff["suspected_static"]]
    else:
        shown = diff
    shown = shown.sort_values("abs_area_delta", ascending=False).head(80)
    st.dataframe(shown, use_container_width=True, hide_index=True)


def file_gallery_page(suite: Path) -> None:
    st.title("PNG Gallery")
    section_map = folder_status(suite)
    section = st.selectbox("Section", list(section_map))
    query = st.text_input("Filename contains", "")
    images = list_pngs(section_map[section], "**/*.png")
    if query:
        images = [path for path in images if query.lower() in path.name.lower()]
    max_images = st.slider("Max images", 6, 120, 36, 6)
    image_gallery(images[:max_images], columns=3)


@st.cache_data(show_spinner=False)
def read_xy_file(path: str) -> pd.DataFrame:
    arr = np.loadtxt(path)
    return pd.DataFrame({"two_theta": arr[:, 0], "intensity": arr[:, 1]})


def show_xdi_local_cell(row: pd.Series) -> None:
    st.subheader("Selected Pressure-Window Cell")
    numeric_cols = [
        "roi_area",
        "max_intensity",
        "local_baseline_corrected_intensity",
        "tier_a_peak_count",
        "tier_ab_peak_count",
        "tier_c_candidate_count",
        "dominant_peak_position",
        "dominant_peak_fwhm_deg",
        "ncc_zero_to_previous",
        "ncc_to_previous",
        "best_shift_deg_to_previous",
        "ncc_zero_to_reference",
        "ncc_to_reference",
        "best_shift_deg_to_reference",
        "acf_similarity_change",
        "static_background_score",
        "raw2d_static_tracks",
        "raw2d_moving_tracks",
    ]
    display_cols = [
        "cell",
        "frame_label",
        "window_label",
        "xy_path",
        "raw_2d_path",
        *[col for col in numeric_cols if col in row.index],
    ]
    st.dataframe(pd.DataFrame([row[display_cols]]), use_container_width=True, hide_index=True)

    xy_path = resolve_local_path(str(row.get("xy_path", "")))
    if xy_path.exists():
        xy = read_xy_file(str(xy_path))
        lo = float(row["window_start"])
        hi = float(row["window_end"])
        local = xy[(xy["two_theta"] >= lo) & (xy["two_theta"] <= hi)].copy()
        if not local.empty:
            st.line_chart(local.set_index("two_theta")["intensity"], height=260)
            st.caption(f"Local .xy window: {xy_path} | {lo:.2f}-{hi:.2f} deg")
    else:
        st.warning(f"Missing .xy file: {xy_path}")

    raw_path_text = str(row.get("raw_2d_path", "") or "")
    if raw_path_text:
        raw_path = resolve_local_path(raw_path_text)
        st.markdown(f"Raw 2D image path: `{raw_path}`")
        if raw_path.exists() and st.checkbox("Preview raw 2D image", value=False):
            try:
                show_image(raw_path, "Relevant raw 2D detector image")
            except Exception as exc:  # pragma: no cover - depends on TIFF backend.
                st.warning(f"Could not preview raw 2D image: {exc}")
    else:
        st.caption("No raw 2D image path was matched for this pressure.")


def xdi_pressure_window_page(xdi_dir: Path) -> None:
    st.title("XDI-Inspired Pressure-Window Maps")
    st.caption("Pressure-series adaptation of SXDM/XDI: pressure x sliding 2theta ROI feature maps.")
    if not xdi_dir.exists():
        st.warning(
            "XDI output folder not found. Run: "
            "`python scripts/xdi_pressure_window_maps.py`"
        )
        return

    readme = xdi_dir / "README.md"
    report = xdi_dir / "xdi_pressure_window_report.md"
    if readme.exists():
        with st.expander("Method README", expanded=False):
            st.markdown(read_text(str(readme)))
    if report.exists():
        with st.expander("Concise report", expanded=True):
            st.markdown(read_text(str(report)))

    feature_path = xdi_dir / "tables" / "pressure_window_feature_table.csv"
    heatmap_index_path = xdi_dir / "heatmap_index.csv"
    if not feature_path.exists() or not heatmap_index_path.exists():
        st.warning("Missing XDI feature table or heatmap index.")
        return

    features = read_csv(str(feature_path))
    heatmaps = read_csv(str(heatmap_index_path))
    cells = sorted(features["cell"].dropna().unique().tolist())
    if not cells:
        st.warning("No cells found in XDI feature table.")
        return

    tab_heatmap, tab_cross, tab_acf, tab_tables = st.tabs(
        ["Heatmap Viewer", "Cross-Pressure NCC", "Within-Frame ACF", "Candidate Tables"]
    )

    with tab_heatmap:
        c1, c2 = st.columns([1, 1])
        with c1:
            cell = st.selectbox("Cell", cells, key="xdi_cell")
        cell_heatmaps = heatmaps[heatmaps["cell"] == cell].copy()
        features_available = cell_heatmaps["feature"].tolist()
        preferred = "static_background_score" if "static_background_score" in features_available else features_available[0]
        with c2:
            feature = st.selectbox(
                "Feature",
                features_available,
                index=features_available.index(preferred),
                key="xdi_feature",
            )
        selected_heatmap = cell_heatmaps[cell_heatmaps["feature"] == feature].iloc[0]
        show_image(resolve_local_path(selected_heatmap["heatmap"]), selected_heatmap.get("title", feature))
        with st.expander("Heatmap matrix CSV", expanded=False):
            matrix_path = resolve_local_path(selected_heatmap["csv"])
            if matrix_path.exists():
                st.dataframe(read_csv(str(matrix_path)), use_container_width=True)

        cell_features = features[features["cell"] == cell].copy()
        pressure_order = (
            cell_features[["pressure_gpa", "frame_label"]]
            .drop_duplicates()
            .sort_values("pressure_gpa")["frame_label"]
            .tolist()
        )
        window_order = (
            cell_features[["window_start", "window_label"]]
            .drop_duplicates()
            .sort_values("window_start")["window_label"]
            .tolist()
        )
        pcol, wcol = st.columns([1, 1])
        with pcol:
            frame_label = st.selectbox("Pressure", pressure_order, key="xdi_pressure")
        with wcol:
            window_label = st.selectbox("Window", window_order, key="xdi_window")
        match = cell_features[
            (cell_features["frame_label"] == frame_label)
            & (cell_features["window_label"] == window_label)
        ]
        if not match.empty:
            show_xdi_local_cell(match.iloc[0])

    with tab_cross:
        cross_path = xdi_dir / "same_window_cross_pressure_index.csv"
        if not cross_path.exists():
            st.info("No same-window cross-pressure index found.")
        else:
            cross = read_csv(str(cross_path))
            cell = st.selectbox("Cell", sorted(cross["cell"].unique()), key="xdi_cross_cell")
            subset = cross[cross["cell"] == cell].copy()
            matrix_type = st.selectbox(
                "Matrix",
                ["zero_shift_ncc", "best_shift_ncc", "best_shift_deg"],
                key="xdi_cross_matrix",
            )
            subset = subset[subset["matrix_type"] == matrix_type].sort_values("window_start")
            selected_window = st.selectbox("Window", subset["window_label"].tolist(), key="xdi_cross_window")
            row = subset[subset["window_label"] == selected_window].iloc[0]
            show_image(resolve_local_path(row["heatmap"]), f"{cell} {selected_window}: {matrix_type}")
            with st.expander("Matrix CSV", expanded=False):
                st.dataframe(read_csv(str(resolve_local_path(row["csv"]))), use_container_width=True)

    with tab_acf:
        acf_path = xdi_dir / "within_frame_acf_index.csv"
        if not acf_path.exists():
            st.info("No within-frame ACF index found.")
        else:
            acf = read_csv(str(acf_path))
            cell = st.selectbox("Cell", sorted(acf["cell"].unique()), key="xdi_acf_cell")
            subset = acf[acf["cell"] == cell].sort_values("pressure_gpa").copy()
            mode = st.radio("Overlap mode", ["all_windows", "nonoverlap_only"], horizontal=True, key="xdi_acf_mode")
            subset = subset[subset["mode"] == mode]
            labels = subset["frame_label"].tolist()
            frame = st.selectbox("Pressure", labels, key="xdi_acf_pressure")
            row = subset[subset["frame_label"] == frame].iloc[0]
            show_image(resolve_local_path(row["heatmap"]), f"{cell} {frame}: {mode}")
            with st.expander("ACF matrix CSV", expanded=False):
                st.dataframe(read_csv(str(resolve_local_path(row["csv"]))), use_container_width=True)

    with tab_tables:
        table_specs = [
            ("Static/background candidates", xdi_dir / "tables" / "static_background_window_candidates.csv"),
            ("Systematically shifting windows", xdi_dir / "tables" / "systematically_shifting_windows.csv"),
            ("Appearance/disappearance candidates", xdi_dir / "tables" / "appearance_disappearance_windows.csv"),
            ("Windows to check in raw 2D", xdi_dir / "tables" / "windows_to_check_in_raw2d.csv"),
        ]
        for title, path in table_specs:
            if path.exists():
                st.subheader(title)
                st.dataframe(read_csv(str(path)).head(80), use_container_width=True, hide_index=True)


def main() -> None:
    app_css()
    st.sidebar.title("XRD Browser")
    suite_raw = st.sidebar.text_input("Correlation suite folder", str(DEFAULT_SUITE))
    suite = resolve_suite(suite_raw)
    st.sidebar.caption(str(suite))
    xdi_raw = st.sidebar.text_input("XDI maps folder", str(DEFAULT_XDI))
    xdi_dir = resolve_local_path(xdi_raw)
    if not suite.exists():
        st.error(f"Suite folder does not exist: {suite}")
        return

    tables = load_key_tables(suite)
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Search",
            "Overall Correlation",
            "Static Peaks",
            "Scientific Peak Validation",
            "Per-Peak Browser",
            "Window ACF",
            "Frame Pair Inspector",
            "XDI Pressure-Window Maps",
            "PNG Gallery",
        ],
    )

    if page == "Overview":
        overview_page(suite, tables)
    elif page == "Search":
        search_page(suite, tables)
    elif page == "Overall Correlation":
        overall_page(suite, tables)
    elif page == "Static Peaks":
        static_peaks_page(suite, tables)
    elif page == "Scientific Peak Validation":
        scientific_validation_page(suite, tables)
    elif page == "Per-Peak Browser":
        per_peak_page(suite, tables)
    elif page == "Window ACF":
        window_acf_page(suite, tables)
    elif page == "Frame Pair Inspector":
        frame_pair_page(suite, tables)
    elif page == "XDI Pressure-Window Maps":
        xdi_pressure_window_page(xdi_dir)
    else:
        file_gallery_page(suite)


if __name__ == "__main__":
    main()
