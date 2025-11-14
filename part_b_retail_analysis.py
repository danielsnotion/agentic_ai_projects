#!/usr/bin/env python3
"""
part_b_retail_analysis.py
- Usage:
    python part_b_retail_analysis.py --csv requirements/Retail_Transactions_Dataset.csv --out retail_outputs
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import math
import warnings
import ast
import re
import numbers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.max_open_warning": 0})

# Show all columns without trimming
pd.set_option('display.max_columns', None)  # None means unlimited
pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping

# -------------------- Config / Helpers -------------------- #
EXPECTED_COLS = {
    "transaction_id": ["Transaction_ID", "Transaction Id", "transaction_id", "invoice_id", "Invoice"],
    "date": ["Date", "Transaction_Date", "transaction_date", "Invoice Date", "datetime", "Timestamp"],
    "customer": ["Customer_ID", "Customer Id", "Customer Name", "Customer_Name", "Customer"],
    "product": ["Product", "Product Name", "Items", "Item"],
    "total_items": ["Total_Items", "Total Items", "No_of_Items", "Quantity"],
    "total_cost": ["Total_Cost", "Total Cost", "Total", "Amount", "Grand_Total"],
    "payment": ["Payment_Method", "Payment Method", "payment_type", "Payment"],
    "city": ["City", "Store_City", "Location", "Store"],
    "store_type": ["Store_Type", "Store Type", "StoreType"],
    "discount": ["Discount_Applied", "Discount Applied", "Discount"],
    "customer_category": ["Customer_Category", "Customer Category", "Segment"],
    "promotion": ["Promotion", "Promo", "Promotion_Type"],
    "season": ["Season"]
}


def find_first_col(df: pd.DataFrame, candidates):
    """
    Find the first matching column name in the DataFrame from a list of candidate names.

    Tries exact (case-insensitive) matches first, then substring matches.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy by substring
    for cand in candidates:
        for c in df.columns:
            if cand.lower().replace("_", " ") in c.lower().replace("_", " "):
                return c
    return None


def detect_columns(df: pd.DataFrame):
    """Detect dataset columns using EXPECTED_COLS mapping."""
    mapping = {}
    for key, candidates in EXPECTED_COLS.items():
        mapping[key] = find_first_col(df, candidates)
    return mapping


def safe_parse_dates(series: pd.Series) -> pd.Series:
    """Robustly parse a pandas Series to datetimes."""
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        try:
            return pd.to_datetime(series, dayfirst=True, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=series.index)


def month_to_season(month: int) -> str:
    """Map numeric month (1-12) to season string."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Fall"
    return "Unknown"


def explode_products(product_series: pd.Series) -> list:
    """
    Flatten a series of product entries into a list of product names.

    Handles:
      - Proper CSV strings: "['A', 'B', 'C']"
      - Unquoted lists: ['Spinach']
      - Comma-separated values "A, B, C"
      - Single-item strings

    Uses ast.literal_eval when the value looks like a Python list; otherwise falls back to regex split.
    """
    products = []
    list_like_re = re.compile(r"^\s*\[.*\]\s*$")
    for v in product_series.fillna("").astype(str):
        s = v.strip()
        if s == "":
            continue
        # If looks like a Python list, try literal_eval
        if list_like_re.match(s):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    for item in parsed:
                        if item is None:
                            continue
                        item_str = str(item).strip()
                        if item_str:
                            products.append(item_str)
                    continue
            except Exception:
                # fall through to fallback parsing
                pass
        # Fallbacks:
        # 1) If contains commas, split on commas (handles "A, B, C")
        if "," in s:
            parts = [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
            products.extend([p for p in parts if p])
            continue
        # 2) If surrounded by quotes (e.g., "'Spinach'") strip them
        m = re.match(r"^['\"]?(.*?)['\"]?$", s)
        if m:
            item = m.group(1).strip()
            if item:
                products.append(item)
            continue
        # 3) last resort: add raw string
        products.append(s)
    return products


def ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Safely coerce a DataFrame column to numeric (in-place)."""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def try_save_excel(dfs: dict, out_path: Path) -> tuple:
    """Try to write multiple DataFrames into a single Excel workbook."""
    try:
        with pd.ExcelWriter(out_path) as writer:
            for name, df in dfs.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        return True, None
    except Exception as e:
        return False, str(e)


def save_plot(fig, path: Path) -> None:
    """Save a matplotlib figure to disk ensuring target directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -------------------- Plot annotation utilities -------------------- #
def annotate_bar_values(ax, fmt="{:.0f}", fontsize=8, va="bottom"):
    """
    Add numeric labels on top of bar containers in an Axes.
    Works for both simple bar plots and bar charts produced from pandas.
    """
    for p in ax.patches:
        try:
            height = p.get_height()
            if height is None or (isinstance(height, float) and math.isnan(height)):
                continue
            # only show labels for visible bars
            if abs(height) < 1e-12:
                continue
            x = p.get_x() + p.get_width() / 2
            y = height
            ax.text(x, y, fmt.format(height), ha="center", va=va, fontsize=fontsize, rotation=0)
        except Exception:
            continue


def annotate_stacked_totals(ax, df_pivot, fmt="{:.0f}", fontsize=8):
    """
    For stacked bar charts (pivot table plotted), annotate total at top of each stacked bar.
    df_pivot: DataFrame indexed by x, columns are stacks (values numeric)
    """
    totals = df_pivot.sum(axis=1)
    x_positions = np.arange(len(totals))
    for x, total in zip(x_positions, totals):
        ax.text(x, total, fmt.format(total), ha="center", va="bottom", fontsize=fontsize)


def annotate_line_points(ax, x_vals, y_vals, fmt="{:.0f}", fontsize=8):
    """Annotate line chart markers with numeric y-values."""
    for x, y in zip(x_vals, y_vals):
        try:
            if y is None or (isinstance(y, float) and math.isnan(y)):
                continue
            ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=fontsize)
        except Exception:
            continue


def annotate_heatmap_cells(ax, data, fmt="{:.0f}", fontsize=8):
    """Annotate seaborn heatmap cells with formatted values (data: DataFrame)."""
    # data assumed to be numeric DataFrame
    for (i, j), val in np.ndenumerate(data.values):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                txt = "0"
            else:
                txt = fmt.format(val) if isinstance(val, numbers.Number) else str(val)
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=fontsize, color="black")
        except Exception:
            continue


# -------------------- Load & Clean -------------------- #
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV dataset and perform cleaning & canonicalization.

    Ensures canonical columns:
      Customer_ID, Total_Cost, Total_Items, City, Payment_Method, Store_Type,
      Discount_Applied, Promotion, Customer_Category, Product_Raw, Year, Month, Season, etc.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    col_map = detect_columns(df)

    # Date
    if col_map["date"]:
        df["Date"] = safe_parse_dates(df[col_map["date"]])
    else:
        df["Date"] = pd.NaT

    # Customer ID (fallback to name or synthetic)
    if col_map["customer"]:
        df["Customer_ID"] = df[col_map["customer"]].astype(str).str.strip()
    else:
        alt = find_first_col(df, ["Customer_Name", "Name", "Client", "Buyer"])
        if alt:
            df["Customer_ID"] = df[alt].astype(str).str.strip()
        else:
            df["Customer_ID"] = pd.Series([f"cust_{i}" for i in range(len(df))]).astype(str)

    # Total cost
    if col_map["total_cost"]:
        df["Total_Cost"] = pd.to_numeric(df[col_map["total_cost"]], errors="coerce")
    else:
        alt = find_first_col(df, ["Amount", "Price", "Total", "Grand_Total"])
        if alt:
            df["Total_Cost"] = pd.to_numeric(df[alt], errors="coerce")
        else:
            df["Total_Cost"] = np.nan

    # Total items
    if col_map["total_items"]:
        df["Total_Items"] = pd.to_numeric(df[col_map["total_items"]], errors="coerce")
    else:
        df["Total_Items"] = np.nan

    # Canonicalize categorical/text columns
    df["City"] = df[col_map["city"]].astype(str).str.strip() if col_map["city"] else "Unknown"
    df["Payment_Method"] = df[col_map["payment"]].astype(str).str.strip() if col_map["payment"] else "Unknown"
    df["Store_Type"] = df[col_map["store_type"]].astype(str).str.strip() if col_map["store_type"] else None
    df["Discount_Applied"] = df[col_map["discount"]].astype(str).str.strip() if col_map["discount"] else ""
    df["Promotion"] = df[col_map["promotion"]].astype(str).str.strip() if col_map["promotion"] else ""
    df["Customer_Category"] = df[col_map["customer_category"]].astype(str).str.strip() if col_map["customer_category"] else None

    # Product column: normalize into Product_Raw (string) and a parsed items list for downstream counts
    if col_map["product"]:
        df["Product_Raw"] = df[col_map["product"]].astype(str).str.strip()
    else:
        df["Product_Raw"] = ""

    # Derived fields
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month.fillna(0).astype(int)
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["Season"] = df["Month"].apply(lambda m: month_to_season(m) if (m and not math.isnan(m)) else "Unknown")

    # Normalize some values
    df["Discount_Applied"] = df["Discount_Applied"].replace({"True": "Yes", "False": "No", "nan": ""}).fillna("")
    df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
    df["City"] = df["City"].fillna("Unknown")

    # Ensure numeric columns coerced
    df = ensure_numeric(df, "Total_Cost")
    df = ensure_numeric(df, "Total_Items")

    # --- NEW: remove exact duplicate rows (full-row duplicates) to reduce duplicates in analysis
    df = df.drop_duplicates().reset_index(drop=True)

    return df


# -------------------- Metrics & Analyses -------------------- #
def exploratory_metrics(df: pd.DataFrame) -> dict:
    """
    Compute basic exploratory metrics.

    Returns a dict with:
      - total_tx, unique_customers, top_products, tx_by_city, rev_by_city
    """
    total_tx = len(df)
    unique_customers = int(df["Customer_ID"].nunique(dropna=True))

    products = explode_products(df["Product_Raw"]) if "Product_Raw" in df.columns else []
    top_products = Counter(products).most_common(10)

    # Transactions by city
    if "City" in df.columns:
        tx_by_city = df["City"].fillna("Unknown").value_counts().reset_index()
        tx_by_city.columns = ["City", "Transactions"]
    else:
        possible_city = find_first_col(df, ["city", "store", "location"])
        if possible_city:
            tx_by_city = df[possible_city].fillna("Unknown").value_counts().reset_index()
            tx_by_city.columns = ["City", "Transactions"]
        else:
            tx_by_city = pd.DataFrame(columns=["City", "Transactions"])

    if "City" in df.columns and "Total_Cost" in df.columns:
        rev_by_city = df.groupby("City", dropna=False)["Total_Cost"].sum().reset_index().sort_values("Total_Cost", ascending=False)
    else:
        rev_by_city = pd.DataFrame(columns=["City", "Total_Cost"])

    return {
        "total_tx": total_tx,
        "unique_customers": unique_customers,
        "top_products": top_products,
        "tx_by_city": tx_by_city,
        "rev_by_city": rev_by_city
    }


def customer_behavior(df: pd.DataFrame) -> dict:
    """
    Analyze customer behaviour.

    Returns avg_spend_cat, payment_pref (crosstab), avg_items_store.
    """
    avg_spend_cat = None
    if "Customer_Category" in df.columns and df["Customer_Category"].notna().any():
        avg_spend_cat = df.groupby("Customer_Category")["Total_Cost"].mean().reset_index().sort_values("Total_Cost", ascending=False)

    payment_pref = None
    if "Customer_Category" in df.columns and "Payment_Method" in df.columns:
        payment_pref = pd.crosstab(df["Customer_Category"].fillna("Unknown"), df["Payment_Method"].fillna("Unknown"), margins=False)

    avg_items_store = None
    if "Store_Type" in df.columns and df["Store_Type"].notna().any():
        avg_items_store = df.groupby("Store_Type")["Total_Items"].mean().reset_index().sort_values("Total_Items", ascending=False)

    return {
        "avg_spend_cat": avg_spend_cat,
        "payment_pref": payment_pref,
        "avg_items_store": avg_items_store
    }


def promotion_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze promotion and discount impact.
    """
    results = {}
    if "Discount_Applied" in df.columns:
        df_discount = df.copy()
        df_discount["Discount_Flag"] = df_discount["Discount_Applied"].apply(lambda v: "Yes" if str(v).strip().lower() in ("yes", "true", "1") else "No")
        grp = df_discount.groupby("Discount_Flag").agg(
            avg_total_cost=("Total_Cost", "mean"),
            avg_items=("Total_Items", "mean"),
            count=("Total_Cost", "count")
        ).reset_index()
        results["discount_summary"] = grp

    if "Promotion" in df.columns:
        promo = df.groupby("Promotion").agg(avg_total_cost=("Total_Cost", "mean"), count=("Total_Cost", "count")).reset_index().sort_values("avg_total_cost", ascending=False)
        results["promotion_summary"] = promo

    return results


def seasonality_analysis(df: pd.DataFrame) -> dict:
    """
    Compute seasonality metrics.
    """
    rev_by_season = df.groupby("Season", dropna=False)["Total_Cost"].sum().reset_index().sort_values("Total_Cost", ascending=False)
    avg_by_season = df.groupby("Season", dropna=False)["Total_Cost"].mean().reset_index().sort_values("Total_Cost", ascending=False)
    pop_by_season = {}
    for season in df["Season"].dropna().unique():
        subset = df[df["Season"] == season]
        prods = explode_products(subset["Product_Raw"])
        pop_by_season[season] = Counter(prods).most_common(5)
    return {
        "rev_by_season": rev_by_season,
        "avg_by_season": avg_by_season,
        "pop_by_season": pop_by_season
    }


# -------------------- Visualisations (defensive) -------------------- #
def create_visualisations(df: pd.DataFrame, metrics: dict, behavior: dict, promo: dict, season: dict, out_dir: Path) -> list:
    """
    Create and save required visualisations to out_dir/plots.
    """
    plots = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Helper: collect existing numbered plot suffixes (without leading number and underscore)
    existing_numbered_suffixes = set()
    existing_numbers = set()
    for f in plots_dir.glob("*.png"):
        m = re.match(r"^(\d+)_([^.]+)\.png$", f.name)
        if m:
            num = int(m.group(1))
            suffix = m.group(2)
            existing_numbered_suffixes.add(suffix)
            existing_numbers.add(num)
    next_number = max(existing_numbers) + 1 if existing_numbers else 1

    def next_num_name(stem: str) -> Path:
        nonlocal next_number
        # ensure two-digit prefix
        name = f"{next_number:02d}_{stem}.png"
        next_number += 1
        return plots_dir / name

    # ---------------- Core plots (keep numbering and existing stems) ---------------- #
    # 1) Transactions by city
    tx_by_city = metrics.get("tx_by_city", pd.DataFrame(columns=["City", "Transactions"]))
    if isinstance(tx_by_city, pd.DataFrame) and not tx_by_city.empty:
        try:
            display_df = tx_by_city.copy()
            if "City" not in display_df.columns or "Transactions" not in display_df.columns:
                cols = list(display_df.columns)
                if len(cols) >= 2:
                    display_df = display_df.rename(columns={cols[0]: "City", cols[1]: "Transactions"})
                else:
                    display_df = pd.DataFrame(columns=["City", "Transactions"])
            display_df["Transactions"] = pd.to_numeric(display_df["Transactions"], errors="coerce").fillna(0)
            fig, ax = plt.subplots(figsize=(10, 6))
            display_df.head(15).set_index("City")["Transactions"].plot(kind="bar", ax=ax)
            ax.set_title("Top 15 Cities by Transaction Count")
            ax.set_ylabel("Transactions")
            ax.set_xlabel("")
            plt.xticks(rotation=45, ha="right")
            annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
            p = plots_dir / "01_transactions_by_city.png"
            save_plot(fig, p)
            plots.append(p)
            existing_numbered_suffixes.add("transactions_by_city")
            existing_numbers.add(1)
        except Exception as e:
            print("Warning: could not plot transactions by city:", e)

    # 2) Payment method distribution (pie) with counts in legend
    try:
        if "Payment_Method" in df.columns:
            counts = df["Payment_Method"].fillna("Unknown").value_counts()
            if not counts.empty:
                fig, ax = plt.subplots(figsize=(7, 7))
                wedges, texts, autotexts = ax.pie(counts.values, labels=None, autopct="%1.1f%%", startangle=90)
                ax.set_title("Payment Method Distribution")
                # create legend with counts & percent
                legend_labels = []
                for lab, cnt, pct in zip(counts.index, counts.values, autotexts):
                    legend_labels.append(f"{lab}: {int(cnt)} ({pct.get_text()})")
                ax.legend(legend_labels, bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8)
                p = plots_dir / "02_payment_distribution.png"
                save_plot(fig, p)
                plots.append(p)
                existing_numbered_suffixes.add("payment_distribution")
                existing_numbers.add(2)
    except Exception as e:
        print("Warning: could not plot payment distribution:", e)

    # 3) Monthly revenue trend (line with annotated points)
    try:
        monthly = df.dropna(subset=["Date", "Total_Cost"]).set_index("Date").resample("ME")["Total_Cost"].sum().reset_index().rename(columns={"Total_Cost": "Revenue"})
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly["Date"], monthly["Revenue"], marker="o")
            ax.set_title("Monthly Revenue Trend")
            ax.set_ylabel("Revenue")
            ax.set_xlabel("")
            plt.xticks(rotation=45)
            annotate_line_points(ax, monthly["Date"], monthly["Revenue"], fmt="{:.0f}", fontsize=8)
            p = plots_dir / "03_monthly_revenue_trend.png"
            save_plot(fig, p)
            plots.append(p)
            existing_numbered_suffixes.add("monthly_revenue_trend")
            existing_numbers.add(3)
    except Exception as e:
        print("Warning: could not plot monthly revenue trend:", e)

    # 4) Revenue by season (bar + labels)
    try:
        rev_by_season = season.get("rev_by_season", pd.DataFrame())
        if isinstance(rev_by_season, pd.DataFrame) and not rev_by_season.empty:
            if "Season" in rev_by_season.columns and "Total_Cost" in rev_by_season.columns:
                order = ["Winter", "Spring", "Summer", "Fall", "Unknown"]
                rev_idx = rev_by_season.set_index("Season").reindex([s for s in order if s in rev_by_season["Season"].values or True]).fillna(0)
                fig, ax = plt.subplots(figsize=(8, 5))
                rev_idx["Total_Cost"].plot(kind="bar", ax=ax)
                ax.set_title("Total Revenue by Season")
                ax.set_ylabel("Revenue")
                annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
                p = plots_dir / "04_revenue_by_season.png"
                save_plot(fig, p)
                plots.append(p)
                existing_numbered_suffixes.add("revenue_by_season")
                existing_numbers.add(4)
    except Exception as e:
        print("Warning: could not plot revenue by season:", e)

    # 5) Stacked season x customer-category (stacked bars + totals)
    try:
        if "Customer_Category" in df.columns:
            pivot = df.groupby(["Season", "Customer_Category"])["Total_Cost"].sum().reset_index().pivot(index="Season", columns="Customer_Category", values="Total_Cost").fillna(0)
            if not pivot.empty:
                order = [s for s in ["Winter", "Spring", "Summer", "Fall", "Unknown"] if s in pivot.index] + [s for s in pivot.index if s not in ["Winter", "Spring", "Summer", "Fall", "Unknown"]]
                pivot = pivot.reindex(order).fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                pivot.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title("Revenue by Season and Customer Category (stacked)")
                ax.set_ylabel("Revenue")
                annotate_stacked_totals(ax, pivot, fmt="{:.0f}", fontsize=8)
                p = plots_dir / "05_season_customer_category_stacked.png"
                save_plot(fig, p)
                plots.append(p)
                existing_numbered_suffixes.add("season_customer_category_stacked")
                existing_numbers.add(5)
    except Exception as e:
        print("Warning: could not plot season x customer-category stacked:", e)

    # 6) Top products bar (annotated)
    top_products = metrics.get("top_products", [])
    if top_products:
        try:
            prod_df = pd.DataFrame(top_products, columns=["Product", "Count"])
            fig, ax = plt.subplots(figsize=(12, 6))
            prod_df.set_index("Product")["Count"].plot(kind="bar", ax=ax)
            ax.set_title("Top Products by Transaction Count")
            annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
            p = plots_dir / "06_top_products.png"
            save_plot(fig, p)
            plots.append(p)
            existing_numbered_suffixes.add("top_products")
            existing_numbers.add(6)
        except Exception as e:
            print("Warning: could not plot top products:", e)

    # 7) Average spend per season (bar + annotated)
    try:
        avg_by_season = season.get("avg_by_season", pd.DataFrame())
        if isinstance(avg_by_season, pd.DataFrame) and not avg_by_season.empty:
            order = ["Winter", "Spring", "Summer", "Fall", "Unknown"]
            avg_ordered = avg_by_season.set_index("Season").reindex([s for s in order if s in avg_by_season["Season"].values or True]).fillna(0)
            fig, ax = plt.subplots(figsize=(8, 5))
            avg_ordered["Total_Cost"].plot(kind="bar", ax=ax)
            ax.set_title("Average Spend per Season")
            ax.set_ylabel("Average Total Cost")
            annotate_bar_values(ax, fmt="{:.1f}", fontsize=8)
            p = plots_dir / "07_avg_spend_by_season.png"
            save_plot(fig, p)
            plots.append(p)
            existing_numbered_suffixes.add("avg_spend_by_season")
            existing_numbers.add(7)
    except Exception as e:
        print("Warning: could not plot average spend per season:", e)

    # 8) Season x Store_Type stacked bar (annotated totals)
    try:
        if "Store_Type" in df.columns and df["Store_Type"].notna().any():
            pivot = df.groupby(["Season", "Store_Type"])["Total_Cost"].sum().reset_index().pivot(index="Season", columns="Store_Type", values="Total_Cost").fillna(0)
            if not pivot.empty:
                order = [s for s in ["Winter", "Spring", "Summer", "Fall", "Unknown"] if s in pivot.index] + [s for s in pivot.index if s not in ["Winter", "Spring", "Summer", "Fall", "Unknown"]]
                pivot = pivot.reindex(order).fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                pivot.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title("Revenue by Season and Store Type (stacked)")
                ax.set_ylabel("Revenue")
                annotate_stacked_totals(ax, pivot, fmt="{:.0f}", fontsize=8)
                p = plots_dir / "08_season_storetype_stacked.png"
                save_plot(fig, p)
                plots.append(p)
                existing_numbered_suffixes.add("season_storetype_stacked")
                existing_numbers.add(8)
    except Exception as e:
        print("Warning: could not plot season x store-type stacked:", e)

    # 9) Average number of items by Store_Type (bar + labels)
    try:
        avg_items_store = behavior.get("avg_items_store")
        if avg_items_store is not None and not avg_items_store.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # ensure ordering from highest to lowest; column name is "Total_Items"
            avg_items_store_sorted = avg_items_store.sort_values("Total_Items", ascending=False)
            avg_items_store_sorted.set_index("Store_Type")["Total_Items"].plot(kind="bar", ax=ax)
            ax.set_title("Average Number of Items per Transaction by Store Type")
            ax.set_ylabel("Average Total Items")
            ax.set_xlabel("")
            plt.xticks(rotation=45, ha="right")
            annotate_bar_values(ax, fmt="{:.3f}", fontsize=8)
            p = plots_dir / "09_avg_items_by_storetype.png"
            save_plot(fig, p)
            plots.append(p)
            existing_numbered_suffixes.add("avg_items_by_storetype")
            existing_numbers.add(9)
    except Exception as e:
        print("Warning: could not plot avg items by store type:", e)

    # 10) Payment Preference heatmap (counts + % annotation)
    try:
        payment_pref = behavior.get("payment_pref")

        if payment_pref is not None and not payment_pref.empty:
            heatmap_df = payment_pref.copy().astype(float)
            percent_df = heatmap_df.div(heatmap_df.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(heatmap_df, annot=False, fmt="g", cmap="Blues", cbar_kws={'label': 'Count'}, ax=ax)

            # annotate cells with "count\nxx.x%"
            for i, row in enumerate(heatmap_df.index):
                for j, col in enumerate(heatmap_df.columns):
                    count_val = int(heatmap_df.iloc[i, j]) if not np.isnan(heatmap_df.iloc[i, j]) else 0
                    pct_val = percent_df.iloc[i, j] if not np.isnan(percent_df.iloc[i, j]) else 0.0
                    ax.text(j + 0.5, i + 0.5, f"{count_val}\n{pct_val:.1f}%", ha="center", va="center", color="black", fontsize=8)

            ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
            ax.set_yticklabels(heatmap_df.index)
            ax.set_title("Payment Method Preference vs Customer Category\n(Count and %)")
            p = plots_dir / "10_payment_pref_heatmap.png"
            save_plot(fig, p)
            plots.append(p)
            existing_numbered_suffixes.add("payment_pref_heatmap")
            existing_numbers.add(10)

    except Exception as e:
        print("Warning: could not generate Payment Preference heatmap:", e)

    # ---------------- Aggregates CSVs -> charts (force a chart for each CSV) ---------------- #
    # Heuristic approach:
    # - If df has a numeric column and a categorical column -> bar (if multiple numeric columns -> grouped/stacked)
    # - If df has only numeric columns -> line (index on x)
    # - If df has one column categorical -> value_counts bar
    # - If all non-numeric, fallback to value_counts on first column
    try:
        agg_dir = out_dir / "aggregates"
        if agg_dir.exists():
            # recompute highest existing number to continue numbering cleanly
            if existing_numbers:
                next_number = max(existing_numbers) + 1
            else:
                next_number = 11

            def numbered_path_for(stem: str) -> Path:
                nonlocal next_number
                name = f"{next_number:02d}_{stem}.png"
                next_number += 1
                return plots_dir / name

            for csvf in sorted(agg_dir.glob("*.csv")):
                try:
                    stem = csvf.stem  # e.g., transactions_by_city, monthly_revenue, top_products...
                    # if a numbered plot with same stem already exists, skip to avoid duplicates
                    if stem in existing_numbered_suffixes:
                        continue

                    gdf = pd.read_csv(csvf)
                    if gdf.empty:
                        continue

                    plot_path = numbered_path_for(stem)
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Identify numeric columns
                    numeric_cols = [c for c in gdf.columns if np.issubdtype(gdf[c].dtype, np.number)]
                    non_numeric_cols = [c for c in gdf.columns if c not in numeric_cols]

                    plotted = False

                    # Case A: categorical key + one or more numeric columns -> grouped or stacked bar
                    if non_numeric_cols and numeric_cols:
                        keycol = non_numeric_cols[0]
                        # if single numeric -> simple bar
                        if len(numeric_cols) == 1:
                            valcol = numeric_cols[0]
                            try:
                                gdf.set_index(keycol)[valcol].plot(kind="bar", ax=ax)
                                ax.set_title(f"{stem} ({valcol})")
                                annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
                                plotted = True
                            except Exception:
                                pass
                        else:
                            # multiple numeric columns -> grouped bars
                            try:
                                gdf.set_index(keycol)[numeric_cols].plot(kind="bar", ax=ax)
                                ax.set_title(f"{stem} (grouped)")
                                annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
                                plotted = True
                            except Exception:
                                # try stacked
                                try:
                                    gdf.set_index(keycol)[numeric_cols].plot(kind="bar", stacked=True, ax=ax)
                                    ax.set_title(f"{stem} (stacked)")
                                    # annotate totals
                                    annotate_stacked_totals(ax, gdf.set_index(keycol)[numeric_cols], fmt="{:.0f}", fontsize=8)
                                    plotted = True
                                except Exception:
                                    pass

                    # Case B: only numeric cols -> line chart for each numeric column (annotated)
                    if not plotted and numeric_cols and not non_numeric_cols:
                        try:
                            if gdf.shape[1] == 1:
                                # single numeric series -> line with points
                                ax.plot(gdf.index, gdf.iloc[:, 0], marker='o')
                                ax.set_title(stem)
                                annotate_line_points(ax, gdf.index, gdf.iloc[:, 0], fmt="{:.0f}", fontsize=8)
                            else:
                                # multiple numeric columns -> plot each line
                                for col in numeric_cols:
                                    ax.plot(gdf.index, gdf[col], marker='o', label=str(col))
                                ax.set_title(stem)
                                ax.legend(fontsize=8)
                                # annotate last point of each line
                                for col in numeric_cols:
                                    annotate_line_points(ax, gdf.index.to_list(), gdf[col].to_list(), fmt="{:.0f}", fontsize=8)
                            plotted = True
                        except Exception:
                            pass

                    # Case C: first column categorical and rest non-numeric or mixed -> value_counts on first column
                    if not plotted and gdf.shape[1] >= 1:
                        first_col = gdf.columns[0]
                        try:
                            vc = gdf[first_col].value_counts()
                            if len(vc) > 0:
                                vc.plot(kind="bar", ax=ax)
                                ax.set_title(f"{stem} ({first_col} counts)")
                                annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
                                plotted = True
                        except Exception:
                            pass

                    # Case D: fallback to trying each pair: first non-numeric as x and first numeric as y
                    if not plotted and non_numeric_cols and numeric_cols:
                        try:
                            gdf.set_index(non_numeric_cols[0])[numeric_cols[0]].plot(kind="bar", ax=ax)
                            ax.set_title(f"{stem} ({numeric_cols[0]})")
                            annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
                            plotted = True
                        except Exception:
                            pass

                    # If still not plotted, as last resort try to convert columns to numbers where possible and plot heatmap if rectangular
                    if not plotted:
                        # Try heatmap if many numeric values
                        if gdf.shape[0] > 1 and gdf.shape[1] > 1:
                            try:
                                # try to coerce to numeric matrix (NaNs where not numeric)
                                numeric_matrix = gdf.apply(pd.to_numeric, errors="coerce")
                                if numeric_matrix.notna().sum().sum() > 0:
                                    sns.heatmap(numeric_matrix.fillna(0), annot=True, fmt=".0f", ax=ax, cbar_kws={'label': 'value'})
                                    ax.set_title(stem)
                                    plotted = True
                            except Exception:
                                pass

                    # If still not plotted, fallback to value_counts on first column (guarantee a chart)
                    if not plotted:
                        try:
                            first_col = gdf.columns[0]
                            vc = gdf[first_col].astype(str).value_counts().head(30)  # top 30 categories if many
                            vc.plot(kind="bar", ax=ax)
                            ax.set_title(f"{stem} ({first_col} counts)")
                            annotate_bar_values(ax, fmt="{:.0f}", fontsize=8)
                            plotted = True
                        except Exception:
                            # last fallback: show table but this should be rare now
                            ax.axis("off")
                            tbl = ax.table(cellText=gdf.head(20).values, colLabels=gdf.columns, loc='center')
                            tbl.auto_set_font_size(False)
                            tbl.set_fontsize(8)
                            ax.set_title(stem)
                            plotted = True

                    plt.xticks(rotation=45, ha="right")
                    save_plot(fig, plot_path)
                    plots.append(plot_path)
                    existing_numbered_suffixes.add(stem)
                except Exception:
                    continue
    except Exception:
        pass

    return plots


# -------------------- Aggregates Save -------------------- #
def save_aggregates(out_dir: Path, metrics: dict, behavior: dict, promo: dict, season: dict, df: pd.DataFrame) -> tuple:
    """
    Save aggregate CSVs and attempt to write an Excel workbook.
    """
    agg_dir = out_dir / "aggregates"
    agg_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics["tx_by_city"].to_csv(agg_dir / "transactions_by_city.csv", index=False)
    except Exception:
        pass
    try:
        metrics["rev_by_city"].to_csv(agg_dir / "revenue_by_city.csv", index=False)
    except Exception:
        pass

    try:
        pd.DataFrame(metrics["top_products"], columns=["Product", "Count"]).to_csv(agg_dir / "top_products.csv", index=False)
    except Exception:
        pass

    try:
        monthly = df.dropna(subset=["Date", "Total_Cost"]).set_index("Date").resample("ME")["Total_Cost"].sum().reset_index().rename(columns={"Total_Cost": "Revenue"})
        monthly.to_csv(agg_dir / "monthly_revenue.csv", index=False)
    except Exception:
        pass

    if "discount_summary" in promo:
        try:
            promo["discount_summary"].to_csv(agg_dir / "discount_summary.csv", index=False)
        except Exception:
            pass
    if "promotion_summary" in promo:
        try:
            promo["promotion_summary"].to_csv(agg_dir / "promotion_summary.csv", index=False)
        except Exception:
            pass

    try:
        season["rev_by_season"].to_csv(agg_dir / "revenue_by_season.csv", index=False)
        season["avg_by_season"].to_csv(agg_dir / "avg_by_season.csv", index=False)
    except Exception:
        pass

    # payment_pref export
    if behavior.get("payment_pref") is not None:
        try:
            behavior["payment_pref"].to_csv(agg_dir / "payment_pref.csv")
        except Exception:
            pass

    # avg_spend_cat export
    if behavior.get("avg_spend_cat") is not None:
        try:
            behavior["avg_spend_cat"].to_csv(agg_dir / "avg_spend_cat.csv", index=False)
        except Exception:
            pass

    # season x storetype pivot
    try:
        if "Store_Type" in df.columns and df["Store_Type"].notna().any():
            pivot = df.groupby(["Season", "Store_Type"])["Total_Cost"].sum().reset_index().pivot(index="Season", columns="Store_Type", values="Total_Cost").fillna(0)
            pivot.reset_index().to_csv(agg_dir / "season_storetype_revenue.csv", index=False)
    except Exception:
        pass

    # attempt Excel
    dfs_for_excel = {}
    try:
        if not metrics.get("tx_by_city", pd.DataFrame()).empty:
            dfs_for_excel["transactions_by_city"] = metrics["tx_by_city"]
        if not metrics.get("rev_by_city", pd.DataFrame()).empty:
            dfs_for_excel["revenue_by_city"] = metrics["rev_by_city"]
        if not pd.DataFrame(metrics.get("top_products", [])).empty:
            dfs_for_excel["top_products"] = pd.DataFrame(metrics.get("top_products", []), columns=["Product", "Count"])
        if not season.get("rev_by_season", pd.DataFrame()).empty:
            dfs_for_excel["revenue_by_season"] = season["rev_by_season"]
        ok, err = try_save_excel(dfs_for_excel, agg_dir / "key_aggregates.xlsx")
        return ok, err
    except Exception as e:
        return False, str(e)


# -------------------- Orchestrator -------------------- #
def analyze(csv_path: str, out: str = "retail_outputs") -> dict:
    """
    Full analysis orchestration.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning dataset...")
    df = load_and_clean(csv_path)
    # Print dataset after cleaning
    print("------------------------------------------------------Dataset After Cleaning (Read the CSV file)------------------------------------------------------------")
    print(df.head(5))
    print("------------------------------------------------------Parse and convert the Date column into an appropriate format------------------------------------------------------------")
    print(df['Date'].head(5))
    print("------------------------------------------------------Extract additional useful information like Year, Month, or DayOfWeek from the Date------------------------------------------------------------")
    print(df[['Date','Year','Month','DayOfWeek','YearMonth']].head(5))
    print("------------------------------------------------------Clean and preprocess the data if required.------------------------------------------------------------")
    print(df[['Total_Cost','Total_Items','City','Payment_Method','Store_Type','Discount_Applied','Promotion', 'Customer_Category','Product_Raw','Season']].head(5))
    print("----------------------------------------------------------------------------------------------------------------------------------------")

    print("Computing exploratory metrics...")
    metrics = exploratory_metrics(df)

    print("Analysing customer behaviour...")
    global behavior
    behavior = customer_behavior(df)

    print("Analysing promotions...")
    promo = promotion_analysis(df)

    print("Analysing seasonality...")
    season = seasonality_analysis(df)

    print("Creating visualisations...")
    plots = create_visualisations(df, metrics, behavior, promo, season, out_dir)

    print("Saving aggregates...")
    ok, err = save_aggregates(out_dir, metrics, behavior, promo, season, df)
    if not ok:
        print("Note: Excel write failed (openpyxl may be missing). CSV fallbacks created. Error:", err)

    # ---- concise summary (defensive) ----
    print("\n=== Part B: Summary ===")
    print(f"Total transactions: {int(metrics.get('total_tx', 0))}")
    print(f"Unique customers: {int(metrics.get('unique_customers', 0))}")

    # defensive top city extraction
    top_city = None
    tx_df = metrics.get("tx_by_city")
    if isinstance(tx_df, pd.DataFrame) and not tx_df.empty:
        if "City" in tx_df.columns:
            top_city = tx_df.iloc[0]["City"]
        else:
            top_city = tx_df.iloc[0][tx_df.columns[0]]
    if top_city:
        print(f"Highest volume city: {top_city}")

    if behavior.get("avg_spend_cat") is not None and not behavior["avg_spend_cat"].empty:
        top_cat = behavior["avg_spend_cat"].iloc[0]["Customer_Category"]
        print(f"Highest average spend customer category: {top_cat}")

    if not season.get("rev_by_season", pd.DataFrame()).empty:
        top_season = season["rev_by_season"].iloc[0]["Season"]
        print(f"Highest total revenue season: {top_season}")

    if metrics.get("top_products"):
        top5 = ", ".join([f"{p} ({c})" for p, c in metrics["top_products"][:5]])
        print(f"Top 5 products: {top5}")

    print("\nRecommended actions:")
    print("- Staff & stock top cities more heavily in peak season.")
    if behavior.get("avg_spend_cat") is not None:
        print("- Personalise offers for the highest-spend customer categories.")
    else:
        print("- Capture customer segment data to enable targeted campaigns.")
    print("- A/B test promotions to validate uplift vs margin impact.")
    print("- Streamline checkout for dominant payment methods; add popular payment rails.")

    print("\nSaved plots and aggregates to:", out_dir.resolve())
    if plots:
        print("Saved plots:")
        for p in plots:
            print(" -", Path(p).resolve())

    return {
        "total_transactions": metrics.get("total_tx"),
        "unique_customers": metrics.get("unique_customers"),
        "top_city": top_city,
        "top_season": season.get("rev_by_season").iloc[0]["Season"] if not season.get("rev_by_season", pd.DataFrame()).empty else None,
        "top_products": [p for p, _ in metrics.get("top_products", [])[:5]]
    }


# -------------------- CLI -------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part B: Retail Transaction Insights")
    parser.add_argument("--csv", default="Retail_Transactions_Dataset.csv", help="Path to Retail_Transactions_Dataset.csv")
    parser.add_argument("--out", default="retail_outputs", help="Output folder for plots & aggregates")
    args = parser.parse_args()

    analyze(args.csv, args.out)
