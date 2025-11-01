#!/usr/bin/env python3
"""
Basic statistics and visualization for the Trump–Xi meeting news dataset.

Actions:
- Read the CSV input file
- Count number of articles (rows)
- Add a 'word_count' column for each article's body
- Generate charts (histogram of word counts, top sources, articles per day)
- Write a markdown report summarizing the results

Outputs are written to the output directory (default: this demo folder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute basic stats and plots for the CSV file.")
    parser.add_argument(
        "--input",
        default="/workspaces/ainewsdemo/data/trump_xi_meeting_fulltext_dedup-1657.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--outdir",
        default="/workspaces/ainewsdemo/demo",
        help="Directory to write outputs (CSV with word_count, charts, report)",
    )
    return parser.parse_args()


def safe_word_count(text: object) -> int:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    s = str(text).strip()
    if not s:
        return 0
    # Basic tokenization on whitespace; robust enough for quick stats
    return len(s.split())


def load_data(path: Path) -> pd.DataFrame:
    # Ensure consistent dtypes; keep 'body' as string/object
    df = pd.read_csv(path)
    # Normalize expected columns
    expected_cols = {"title", "authors", "source", "url", "published", "language", "sentiment", "body"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"Warning: missing expected columns: {sorted(missing)}", file=sys.stderr)
    return df


def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df["body"].apply(safe_word_count)
    return df


def parse_published(df: pd.DataFrame) -> pd.Series:
    # Convert to datetime; coerce errors to NaT
    return pd.to_datetime(df.get("published", pd.Series([], dtype=str)), errors="coerce", utc=True)


def make_plots(df: pd.DataFrame, outdir: Path) -> Tuple[Path, Path, Path]:
    sns.set_theme(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Histogram of word counts
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    sns.histplot(df["word_count"], bins=50, kde=False, ax=ax1, color="#1f77b4")
    ax1.set_title("Distribution of Article Word Counts")
    ax1.set_xlabel("Word count")
    ax1.set_ylabel("Articles")
    fig1.tight_layout()
    hist_path = outdir / "word_count_hist.png"
    fig1.savefig(hist_path, dpi=150)
    plt.close(fig1)

    # 2) Top sources by article count (top 10)
    top_sources = (
        df["source"].fillna("(unknown)").astype(str).value_counts().head(10).sort_values(ascending=True)
    )
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.barh(top_sources.index, top_sources.values, color="#2ca02c")
    ax2.set_title("Top 10 Sources by Article Count")
    ax2.set_xlabel("Articles")
    ax2.set_ylabel("Source")
    fig2.tight_layout()
    sources_path = outdir / "top_sources_bar.png"
    fig2.savefig(sources_path, dpi=150)
    plt.close(fig2)

    # 3) Articles per day line chart
    published_dt = parse_published(df)
    per_day = published_dt.dt.tz_convert(None).dt.date.value_counts().sort_index()
    per_day_index = pd.to_datetime(pd.Series(list(per_day.index)))
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(per_day_index, per_day.values, marker="o", linestyle="-", color="#d62728")
    ax3.set_title("Articles per Day")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Articles")
    fig3.autofmt_xdate()
    fig3.tight_layout()
    per_day_path = outdir / "articles_per_day.png"
    fig3.savefig(per_day_path, dpi=150)
    plt.close(fig3)

    return hist_path, sources_path, per_day_path


def write_report(
    df: pd.DataFrame,
    outdir: Path,
    plots: Tuple[Path, Path, Path],
    output_csv_path: Path,
) -> Path:
    n_articles = len(df)
    wc = df["word_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    top_sources = df["source"].fillna("(unknown)").astype(str).value_counts().head(10)

    report_path = outdir / "basic_stats_report.md"
    rel_plot_paths = [p.name for p in plots]

    lines = []
    lines.append(f"# Basic Stats Report\n")
    lines.append("")
    lines.append(f"- Total articles: {n_articles}")
    lines.append(f"- Output CSV with word counts: `{output_csv_path.name}`")
    lines.append("")
    lines.append("## Word count summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for label, value in (
        ("min", int(wc["min"]) if not np.isnan(wc["min"]) else 0),
        ("25%", int(wc["25%" ] if "25%" in wc else 0)),
        ("median", int(wc["50%"] if "50%" in wc else 0)),
        ("75%", int(wc["75%"] if "75%" in wc else 0)),
        ("90%", int(wc.get("90%", 0))),
        ("95%", int(wc.get("95%", 0))),
        ("max", int(wc["max"]) if not np.isnan(wc["max"]) else 0),
        ("mean", round(float(wc["mean"]) if not np.isnan(wc["mean"]) else 0.0, 2)),
    ):
        lines.append(f"| {label} | {value} |")
    lines.append("")
    lines.append("### Histogram of word counts")
    lines.append("")
    lines.append(f"![Word count histogram]({rel_plot_paths[0]})")
    lines.append("")
    lines.append("## Top sources")
    lines.append("")
    lines.append("| source | articles |")
    lines.append("|---|---:|")
    for src, cnt in top_sources.items():
        lines.append(f"| {src} | {int(cnt)} |")
    lines.append("")
    lines.append(f"![Top sources bar chart]({rel_plot_paths[1]})")
    lines.append("")
    lines.append("## Articles per day")
    lines.append("")
    lines.append(f"![Articles per day]({rel_plot_paths[2]})")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    df = load_data(input_path)
    df_wc = add_word_count(df)

    # Save augmented CSV in outdir (do not overwrite original)
    output_csv_path = outdir / "trump_xi_meeting_fulltext_with_wordcount.csv"
    df_wc.to_csv(output_csv_path, index=False)

    # Generate plots
    plots = make_plots(df_wc, outdir)

    # Write report
    report_path = write_report(df_wc, outdir, plots, output_csv_path)

    # Console summary
    print(f"Articles: {len(df)}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Report: {report_path}")
    for p in plots:
        print(f"Chart: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
