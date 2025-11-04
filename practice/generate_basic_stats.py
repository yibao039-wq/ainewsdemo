#!/usr/bin/env python3
"""Generate basic statistics and charts for the provided CSV.

Reads /workspaces/ainewsdemo/data/trump_xi_meeting_fulltext_dedup-1657.csv,
computes a word count for each article (column `body`), writes a new CSV
with the `word_count` column, and saves plots and a small markdown report
into /workspaces/ainewsdemo/practice/output.
"""
import os
from pathlib import Path
import sys
import statistics

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print("Missing required packages. Please install pandas, matplotlib, seaborn.")
    raise


INPUT_CSV = "/workspaces/ainewsdemo/data/trump_xi_meeting_fulltext_dedup-1657.csv"
OUTPUT_DIR = "/workspaces/ainewsdemo/practice/output"


def choose_text_column(df):
    # prefer common names, otherwise fall back to the last column
    candidates = ["body", "fulltext", "full_text", "content", "text", "article"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback
    return df.columns[-1]


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Rows read: {total}")

    text_col = choose_text_column(df)
    print(f"Using text column: {text_col}")

    # compute word count robustly
    df[text_col] = df[text_col].fillna("").astype(str)
    df["word_count"] = df[text_col].str.split().str.len()

    out_csv = out / "trump_xi_meeting_fulltext_with_wordcount.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV with word counts to {out_csv}")

    # Basic statistics
    wc = df["word_count"]
    stats = {
        "n_articles": int(len(wc)),
        "mean": float(wc.mean()),
        "median": float(wc.median()),
        "min": int(wc.min()) if len(wc) else 0,
        "max": int(wc.max()) if len(wc) else 0,
    }

    print("Generating plots...")
    sns.set(style="whitegrid")

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(wc, bins=50, kde=False)
    plt.title("Distribution of article word counts")
    plt.xlabel("Word count")
    plt.ylabel("Number of articles")
    hist_path = out / "wordcount_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # Boxplot (log scale could help if extremely skewed)
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=wc)
    plt.title("Boxplot of article word counts")
    box_path = out / "wordcount_boxplot.png"
    plt.tight_layout()
    plt.savefig(box_path)
    plt.close()

    # Top-10 longest articles
    top10 = df.nlargest(10, "word_count")[["title", "word_count"]].copy()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top10, x="word_count", y="title")
    plt.title("Top 10 longest articles by word count")
    plt.xlabel("Word count")
    plt.ylabel("")
    top10_path = out / "top10_longest.png"
    plt.tight_layout()
    plt.savefig(top10_path)
    plt.close()

    # Write a small markdown report
    report_path = out / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Basic stats report\n\n")
        f.write(f"Input CSV: `{INPUT_CSV}`\n\n")
        f.write(f"Total articles: **{stats['n_articles']}**\n\n")
        f.write("## Word count summary\n\n")
        f.write(f"- Mean: {stats['mean']:.1f}\n")
        f.write(f"- Median: {stats['median']:.1f}\n")
        f.write(f"- Min: {stats['min']}\n")
        f.write(f"- Max: {stats['max']}\n\n")
        f.write("## Generated files\n\n")
        f.write(f"- `{out_csv}`\n")
        f.write(f"- `{hist_path}`\n")
        f.write(f"- `{box_path}`\n")
        f.write(f"- `{top10_path}`\n")

    print(f"Report written to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
