"""
Part B: Retail Transaction Insights
-----------------------------------
Tasks:
1) Load & clean Retail_Transactions_Dataset.csv (dates, numerics, categories).
2) Exploratory metrics (transactions, unique customers, top products, etc.).
3) Visualisations:
   - Bar: transactions by city
   - Pie: payment method distribution
   - Line: monthly revenue trend
   - Bar/stacked: seasonality and customer-category mix
4) Concise insights + recommended actions printed at the end.

How to run:
$ python part_b_retail_analysis.py --csv Retail_Transactions_Dataset.csv
"""

import argparse
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------- Helper utils --------------------------- #
def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df["Year"] = df[col].dt.year
        df["Month"] = df[col].dt.month
        df["MonthName"] = df[col].dt.strftime("%b")
        df["DayOfWeek"] = df[col].dt.day_name()
    return df


def normalize_categories(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def to_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_plot_save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Analysis ---------------------------- #
def main(csv_path: str):
    # ---- 1) Load & clean ----
    df = pd.read_csv(csv_path)
    df = ensure_datetime(df, "Date")
    df = normalize_categories(
        df,
        ["Payment_Method", "City", "Store_Type", "Customer_Category", "Season", "Promotion", "Discount_Applied"],
    )
    df = to_numeric(df, ["Total_Items", "Total_Cost"])

    # ---- 2) Exploration ----
    total_tx = len(df)
    unique_customers = df["Customer_Name"].nunique() if "Customer_Name" in df.columns else np.nan

    top_products = None
    if "Product" in df.columns:
        all_products = []
        for items in df["Product"].astype(str):
            all_products.extend([p.strip() for p in items.split(",") if p.strip()])
        top_products = Counter(all_products).most_common(5)

    transactions_by_city = df["City"].value_counts().sort_values(ascending=False) if "City" in df.columns else pd.Series(dtype=int)
    avg_spend_by_category = (
        df.groupby("Customer_Category")["Total_Cost"].mean().sort_values(ascending=False)
        if "Customer_Category" in df.columns and "Total_Cost" in df.columns
        else pd.Series(dtype=float)
    )

    payment_pref = (
        pd.crosstab(df["Customer_Category"], df["Payment_Method"])
        if set(["Customer_Category", "Payment_Method"]).issubset(df.columns)
        else pd.DataFrame()
    )

    monthly_revenue = None
    if "Date" in df.columns and "Total_Cost" in df.columns:
        monthly_revenue = (
            df.set_index("Date")
            .resample("M")["Total_Cost"]
            .sum()
            .reset_index()
            .rename(columns={"Total_Cost": "Revenue"})
        )

    revenue_by_season = (
        df.groupby("Season")["Total_Cost"].sum().sort_values(ascending=False)
        if set(["Season", "Total_Cost"]).issubset(df.columns)
        else pd.Series(dtype=float)
    )
    avg_spend_per_season = (
        df.groupby("Season")["Total_Cost"].mean().sort_values(ascending=False)
        if set(["Season", "Total_Cost"]).issubset(df.columns)
        else pd.Series(dtype=float)
    )

    # ---- 3) Visualisations ----
    out_dir = Path("retail_outputs")

    # 3a) Bar: transactions by city
    if not transactions_by_city.empty:
        fig = plt.figure()
        transactions_by_city.head(15).plot(kind="bar")
        plt.title("Number of Transactions per City (Top 15)")
        plt.xlabel("City")
        plt.ylabel("Transactions")
        safe_plot_save(fig, out_dir / "01_transactions_by_city.png")

    # 3b) Pie: payment method distribution
    if "Payment_Method" in df.columns:
        counts = df["Payment_Method"].value_counts()
        if len(counts) > 0:
            fig = plt.figure()
            plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            plt.title("Payment Method Distribution")
            plt.axis("equal")
            safe_plot_save(fig, out_dir / "02_payment_method_distribution.png")

    # 3c) Line: monthly revenue trend
    if monthly_revenue is not None and len(monthly_revenue) > 0:
        fig = plt.figure()
        plt.plot(monthly_revenue["Date"], monthly_revenue["Revenue"])
        plt.title("Monthly Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        safe_plot_save(fig, out_dir / "03_monthly_revenue_trend.png")

    # 3d) Seasonality: revenue by season
    if not revenue_by_season.empty:
        fig = plt.figure()
        revenue_by_season.plot(kind="bar")
        plt.title("Total Revenue by Season")
        plt.xlabel("Season")
        plt.ylabel("Revenue")
        safe_plot_save(fig, out_dir / "04_total_revenue_by_season.png")

    # 3e) Seasonality: average spend per season
    if not avg_spend_per_season.empty:
        fig = plt.figure()
        avg_spend_per_season.plot(kind="bar")
        plt.title("Average Spend per Season")
        plt.xlabel("Season")
        plt.ylabel("Average Total Cost")
        safe_plot_save(fig, out_dir / "05_avg_spend_per_season.png")

    # ---- 4) Print concise summary ----
    print("\n=== Part B: Summary ===")
    print(f"Total transactions: {total_tx}")
    print(f"Unique customers: {unique_customers}")

    if "City" in df.columns and not df["City"].value_counts().empty:
        top_city = df["City"].value_counts().idxmax()
        print(f"Highest volume city: {top_city}")

    if "Customer_Category" in df.columns and "Total_Cost" in df.columns:
        top_cat = df.groupby("Customer_Category")["Total_Cost"].mean().idxmax()
        print(f"Highest average spend customer category: {top_cat}")

    if "Season" in df.columns and "Total_Cost" in df.columns and not df["Season"].value_counts().empty:
        top_season = df.groupby("Season")["Total_Cost"].sum().idxmax()
        print(f"Highest total revenue season: {top_season}")

    if top_products:
        print("Top 5 products:", ", ".join([f"{p} ({c})" for p, c in top_products]))

    # Recommended actions (1-liners)
    print("\nRecommended actions:")
    print("- Staff & stock top cities more heavily in peak season.")
    print("- Personalise offers for the highest-spend customer segment.")
    print("- A/B test promotions to confirm uplift without margin erosion.")
    print("- Optimise checkout UX for dominant payment methods.")

    # Optional: save a small report CSV with key aggregates
    agg_out = out_dir / "key_aggregates.csv"
    out_frames = []
    if not transactions_by_city.empty:
        out_frames.append(transactions_by_city.rename("transactions").to_frame())
    if monthly_revenue is not None and len(monthly_revenue) > 0:
        out_frames.append(monthly_revenue.rename(columns={"Date": "month"}))
    if out_frames:
        # Align indices and write
        with pd.ExcelWriter(out_dir / "key_aggregates.xlsx") as xw:
            if not transactions_by_city.empty:
                transactions_by_city.rename("transactions").to_frame().to_excel(xw, sheet_name="Transactions_by_City")
            if monthly_revenue is not None and len(monthly_revenue) > 0:
                monthly_revenue.to_excel(xw, sheet_name="Monthly_Revenue", index=False)
        print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Retail_Transactions_Dataset.csv", help="Path to Retail_Transactions_Dataset.csv")
    args = parser.parse_args()
    main(args.csv)
