# ================================================================
#  SALES DATA ANALYSIS AND FORECASTING SYSTEM — main.py
#  Steps: Load → Clean → EDA → Features → Forecast → Dashboard
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "retail_sales.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
PALETTE = "tab10"


# ================================================================
# STEP 1 — DATA LOADING
# ================================================================
def load_data():
    print("\n" + "═" * 60)
    print("  STEP 1 ▶  Loading Dataset")
    print("═" * 60)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {list(df.columns)}")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"\n{df.head(3).to_string(index=False)}")
    return df


# ================================================================
# STEP 2 — DATA CLEANING & PREPROCESSING
# ================================================================
def preprocess(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  STEP 2 ▶  Data Cleaning & Preprocessing")
    print("═" * 60)

    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["Date", "Revenue"])

    # Derived time columns
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"]    = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["YearMonth"] = df["Date"].dt.to_period("M")

    # Profit margin
    df["Profit_Margin_%"] = (df["Profit"] / df["Revenue"] * 100).round(2)

    print(f"  Records after clean: {len(df):,}  (dropped {before - len(df)})")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string() or '  None'}")
    print(f"  New columns: Year, Month, Quarter, Week, Profit_Margin_%")
    return df


# ================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS
# ================================================================
def eda(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  STEP 3 ▶  Exploratory Data Analysis")
    print("═" * 60)

    # ── 3A  Monthly Revenue trend ───────────────────────────
    monthly = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(len(monthly)), monthly["Revenue"], alpha=0.25, color="steelblue")
    ax.plot(range(len(monthly)), monthly["Revenue"], "o-", color="steelblue",
            linewidth=2, markersize=5)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["YearMonth_str"], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Monthly Revenue Trend (2022–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue ($)")
    plt.tight_layout()
    _save("monthly_revenue_trend.png"); print("  [SAVED] monthly_revenue_trend.png")

    # ── 3B  Revenue by Category ─────────────────────────────
    cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    cat_prof = df.groupby("Category")["Profit"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bars = axes[0].bar(cat_rev.index, cat_rev.values,
                       color=sns.color_palette(PALETTE, len(cat_rev)))
    axes[0].bar_label(bars, fmt="${:,.0f}", fontsize=8, padding=3)
    axes[0].set_title("Total Revenue by Category", fontweight="bold")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    bars2 = axes[1].bar(cat_prof.index, cat_prof.values,
                        color=sns.color_palette("Set2", len(cat_prof)))
    axes[1].bar_label(bars2, fmt="${:,.0f}", fontsize=8, padding=3)
    axes[1].set_title("Total Profit by Category", fontweight="bold")
    axes[1].set_ylabel("Profit ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    _save("revenue_by_category.png"); print("  [SAVED] revenue_by_category.png")

    # ── 3C  Revenue by Region ────────────────────────────────
    region_monthly = df.groupby(["YearMonth", "Region"])["Revenue"].sum().reset_index()
    region_monthly["YearMonth_str"] = region_monthly["YearMonth"].astype(str)
    regions = df["Region"].unique()

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, reg in enumerate(regions):
        sub = region_monthly[region_monthly["Region"] == reg]
        ax.plot(sub["YearMonth_str"], sub["Revenue"], "o-",
                label=reg, linewidth=1.8, markersize=4)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Monthly Revenue by Region", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue ($)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save("revenue_by_region.png"); print("  [SAVED] revenue_by_region.png")

    # ── 3D  Top 10 Products ──────────────────────────────────
    top_prod = df.groupby("Product")["Revenue"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top_prod.index, top_prod.values,
                   color=sns.color_palette("coolwarm", len(top_prod)))
    ax.bar_label(bars, fmt="${:,.0f}", fontsize=8, padding=3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Top 10 Products by Revenue", fontsize=13, fontweight="bold")
    ax.set_xlabel("Revenue ($)")
    plt.tight_layout()
    _save("top_products.png"); print("  [SAVED] top_products.png")

    # ── 3E  Sales Rep Performance ────────────────────────────
    rep = df.groupby("Sales_Rep").agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "count")
    ).sort_values("Revenue", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes,
                               ["Revenue", "Profit", "Orders"],
                               ["steelblue", "green", "coral"]):
        axes_obj = ax
        bars = axes_obj.bar(rep.index, rep[col], color=color, edgecolor="black", alpha=0.85)
        axes_obj.bar_label(bars,
                           labels=[f"${v:,.0f}" if col != "Orders" else str(v)
                                   for v in rep[col]],
                           fontsize=8, padding=3)
        axes_obj.set_title(f"Sales Rep — {col}", fontweight="bold")
        axes_obj.set_ylabel(col)
        if col != "Orders":
            axes_obj.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    _save("sales_rep_performance.png"); print("  [SAVED] sales_rep_performance.png")

    # ── 3F  Quarterly heatmap ────────────────────────────────
    pivot = df.pivot_table(values="Revenue", index="Category",
                            columns="Quarter", aggfunc="sum")
    pivot.columns = [f"Q{c}" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"format": "${x:,.0f}"})
    ax.set_title("Revenue Heatmap — Category × Quarter", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("quarterly_heatmap.png"); print("  [SAVED] quarterly_heatmap.png")

    # ── 3G  Profit Margin distribution ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for cat in df["Category"].unique():
        vals = df[df["Category"] == cat]["Profit_Margin_%"]
        ax.hist(vals, bins=15, alpha=0.6, label=cat)
    ax.set_title("Profit Margin Distribution by Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Profit Margin (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    _save("profit_margin_dist.png"); print("  [SAVED] profit_margin_dist.png")

    print("\n  ✅  EDA complete — 7 charts saved.")


# ================================================================
# STEP 4 — FEATURE ENGINEERING
# ================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  STEP 4 ▶  Feature Engineering")
    print("═" * 60)

    # Lag & rolling features on monthly revenue
    monthly = (df.groupby("YearMonth")["Revenue"]
               .sum()
               .reset_index()
               .sort_values("YearMonth"))
    monthly["Revenue_Lag1"]    = monthly["Revenue"].shift(1)
    monthly["Revenue_Lag2"]    = monthly["Revenue"].shift(2)
    monthly["Revenue_Lag3"]    = monthly["Revenue"].shift(3)
    monthly["Rolling_3M_Avg"]  = monthly["Revenue"].rolling(3).mean()
    monthly["Rolling_3M_Std"]  = monthly["Revenue"].rolling(3).std()
    monthly["MoM_Growth_%"]    = monthly["Revenue"].pct_change() * 100
    monthly["Month_Num"]       = range(1, len(monthly) + 1)
    monthly["Month"]           = monthly["YearMonth"].apply(lambda x: x.month)
    monthly["Year"]            = monthly["YearMonth"].apply(lambda x: x.year)

    monthly.dropna(inplace=True)

    print(f"  Features created: Lag1, Lag2, Lag3, Rolling3M_Avg/Std, MoM_Growth_%")
    print(f"  Monthly records available for modelling: {len(monthly)}")
    print(f"\n{monthly[['YearMonth','Revenue','Revenue_Lag1','Rolling_3M_Avg','MoM_Growth_%']].tail(5).to_string(index=False)}")

    # Month-over-Month growth chart
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["green" if x >= 0 else "red" for x in monthly["MoM_Growth_%"]]
    ax.bar(monthly["Month_Num"], monthly["MoM_Growth_%"], color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Month-over-Month Revenue Growth (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month Index")
    ax.set_ylabel("Growth (%)")
    plt.tight_layout()
    _save("mom_growth.png"); print("\n  [SAVED] mom_growth.png")

    return monthly


# ================================================================
# STEP 5 — SALES FORECASTING
# ================================================================
def forecast(monthly: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  STEP 5 ▶  Sales Forecasting")
    print("═" * 60)

    feature_cols = ["Month_Num", "Month", "Year",
                    "Revenue_Lag1", "Revenue_Lag2", "Revenue_Lag3",
                    "Rolling_3M_Avg", "Rolling_3M_Std"]
    X = monthly[feature_cols]
    y = monthly["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    models = {
        "Linear Regression":       LinearRegression(),
        "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    best_model_name, best_r2 = None, -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2,
                         "model": model, "preds": preds}
        print(f"  [{name:25s}]  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_model_name = r2, name

    print(f"\n  🏆  Best model: {best_model_name}  (R²={best_r2:.4f})")

    # ── Actual vs Predicted ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["Month_Num"], monthly["Revenue"],
            "o-", color="steelblue", linewidth=2, label="Actual")
    test_idx = monthly["Month_Num"].values[-len(y_test):]
    for name, res in results.items():
        ax.plot(test_idx, res["preds"], "--", linewidth=1.5, label=f"{name} (pred)")
    ax.axvline(test_idx[0] - 0.5, color="red", linestyle=":", label="Train/Test split")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Actual vs Predicted Monthly Revenue", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month Index")
    ax.set_ylabel("Revenue ($)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("actual_vs_predicted.png"); print("  [SAVED] actual_vs_predicted.png")

    # ── Future 6-month forecast ──────────────────────────────
    best   = results[best_model_name]["model"]
    last   = monthly.iloc[-1]
    future = []
    lag1, lag2, lag3 = last["Revenue"], last["Revenue_Lag1"], last["Revenue_Lag2"]
    roll_avg = last["Rolling_3M_Avg"]
    roll_std = last["Rolling_3M_Std"]
    month_num = int(last["Month_Num"]) + 1
    month     = int(last["Month"])
    year      = int(last["Year"])

    for _ in range(6):
        month += 1
        if month > 12:
            month = 1
            year += 1
        row = {
            "Month_Num":      month_num,
            "Month":          month,
            "Year":           year,
            "Revenue_Lag1":   lag1,
            "Revenue_Lag2":   lag2,
            "Revenue_Lag3":   lag3,
            "Rolling_3M_Avg": roll_avg,
            "Rolling_3M_Std": roll_std,
        }
        pred = best.predict(pd.DataFrame([row]))[0]
        future.append({"Period": f"{year}-{month:02d}", "Forecast_Revenue": pred})
        lag3, lag2, lag1 = lag2, lag1, pred
        roll_avg = (roll_avg * 2 + pred) / 3

    future_df = pd.DataFrame(future)
    print(f"\n  📅  6-Month Revenue Forecast ({best_model_name}):")
    print(future_df.to_string(index=False))

    # ── Forecast chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["Month_Num"], monthly["Revenue"],
            "o-", color="steelblue", linewidth=2, label="Historical")

    future_x = list(range(int(last["Month_Num"]) + 1,
                           int(last["Month_Num"]) + 8))
    hist_last = [monthly["Revenue"].iloc[-1]]
    future_rev = hist_last + list(future_df["Forecast_Revenue"])

    ax.plot(range(int(last["Month_Num"]), int(last["Month_Num"]) + 7),
            future_rev, "s--", color="orangered", linewidth=2,
            markersize=8, label="6-Month Forecast")

    # Confidence band
    ax.fill_between(
        range(int(last["Month_Num"]), int(last["Month_Num"]) + 7),
        [v * 0.90 for v in future_rev],
        [v * 1.10 for v in future_rev],
        alpha=0.15, color="orangered", label="±10% Confidence Band"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(f"Sales Revenue Forecast — Next 6 Months ({best_model_name})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month Index")
    ax.set_ylabel("Revenue ($)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("revenue_forecast.png"); print("  [SAVED] revenue_forecast.png")

    # ── Model comparison bar ─────────────────────────────────
    names  = list(results.keys())
    r2s    = [results[n]["R²"]  for n in names]
    maes   = [results[n]["MAE"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(names, r2s,
                       color=["gold" if n == best_model_name else "steelblue" for n in names],
                       edgecolor="black")
    axes[0].bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    axes[0].set_title("Model Comparison — R² Score", fontweight="bold")
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0, 1.1)

    bars2 = axes[1].bar(names, maes,
                        color=["gold" if n == best_model_name else "coral" for n in names],
                        edgecolor="black")
    axes[1].bar_label(bars2, fmt="${:,.0f}", fontsize=9, padding=3)
    axes[1].set_title("Model Comparison — MAE ($)", fontweight="bold")
    axes[1].set_ylabel("MAE ($)")
    plt.tight_layout()
    _save("model_comparison.png"); print("  [SAVED] model_comparison.png")

    return future_df, best_model_name


# ================================================================
# STEP 6 — INTERACTIVE PLOTLY DASHBOARD
# ================================================================
def build_dashboard(df: pd.DataFrame, monthly: pd.DataFrame,
                    future_df: pd.DataFrame, best_model: str):
    print("\n" + "═" * 60)
    print("  STEP 6 ▶  Building Interactive Dashboard")
    print("═" * 60)

    colors = px.colors.qualitative.Bold

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📈 Monthly Revenue Trend",
            "🏪 Revenue by Category",
            "🗺️ Revenue by Region",
            "🧑‍💼 Sales Rep Performance",
            "🔮 6-Month Revenue Forecast",
            "📊 Quarterly Revenue Heatmap",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "heatmap"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: Monthly trend ────────────────────────────────
    monthly_plot = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly_plot["YM"] = monthly_plot["YearMonth"].astype(str)
    fig.add_trace(go.Scatter(
        x=monthly_plot["YM"], y=monthly_plot["Revenue"],
        mode="lines+markers", name="Revenue",
        line=dict(color="steelblue", width=2),
        fill="tozeroy", fillcolor="rgba(70,130,180,0.15)"
    ), row=1, col=1)

    # ── Panel 2: Category revenue ─────────────────────────────
    cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    fig.add_trace(go.Bar(
        x=cat.index, y=cat.values, name="Category Revenue",
        marker_color=colors[:len(cat)], showlegend=False
    ), row=1, col=2)

    # ── Panel 3: Region lines ─────────────────────────────────
    reg_m = df.groupby(["YearMonth", "Region"])["Revenue"].sum().reset_index()
    reg_m["YM"] = reg_m["YearMonth"].astype(str)
    for i, reg in enumerate(df["Region"].unique()):
        sub = reg_m[reg_m["Region"] == reg]
        fig.add_trace(go.Scatter(
            x=sub["YM"], y=sub["Revenue"], mode="lines+markers",
            name=reg, line=dict(width=1.8),
            marker=dict(size=5)
        ), row=2, col=1)

    # ── Panel 4: Sales Rep bars ───────────────────────────────
    rep = df.groupby("Sales_Rep")["Revenue"].sum().sort_values(ascending=False)
    fig.add_trace(go.Bar(
        x=rep.index, y=rep.values, name="Rep Revenue",
        marker_color=colors[:len(rep)], showlegend=False
    ), row=2, col=2)

    # ── Panel 5: Forecast ─────────────────────────────────────
    hist_ym = monthly_plot["YM"].tolist()
    hist_rev = monthly_plot["Revenue"].tolist()
    fig.add_trace(go.Scatter(
        x=hist_ym, y=hist_rev, mode="lines+markers",
        name="Historical", line=dict(color="steelblue", width=2)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[hist_ym[-1]] + future_df["Period"].tolist(),
        y=[hist_rev[-1]] + future_df["Forecast_Revenue"].tolist(),
        mode="lines+markers", name="Forecast",
        line=dict(color="orangered", dash="dash", width=2),
        marker=dict(size=8, symbol="square")
    ), row=3, col=1)

    # ── Panel 6: Heatmap ──────────────────────────────────────
    pivot = df.pivot_table(values="Revenue", index="Category",
                            columns="Quarter", aggfunc="sum")
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[f"Q{q}" for q in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        showscale=True,
        text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        showlegend=False,
    ), row=3, col=2)

    fig.update_layout(
        title=dict(
            text="🛒  Sales Data Analysis & Forecasting Dashboard",
            font=dict(size=22), x=0.5
        ),
        height=1050,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.03, x=0.5, xanchor="center"),
    )

    # Y-axis dollar formatting
    for r, c in [(1,1),(1,2),(2,1),(2,2),(3,1)]:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=r, col=c)

    out = os.path.join(OUTPUT_DIR, "dashboard.html")
    fig.write_html(out)
    print(f"  [SAVED] dashboard.html")
    print("  Open dashboard.html in any browser for interactive exploration.")


# ================================================================
# STEP 7 — SUMMARY REPORT
# ================================================================
def summary_report(df: pd.DataFrame, future_df: pd.DataFrame, best_model: str):
    print("\n" + "═" * 60)
    print("  STEP 7 ▶  Summary Report")
    print("═" * 60)

    total_rev   = df["Revenue"].sum()
    total_prof  = df["Profit"].sum()
    avg_margin  = df["Profit_Margin_%"].mean()
    total_units = df["Units_Sold"].sum()
    total_orders= df["Order_ID"].count()
    best_cat    = df.groupby("Category")["Revenue"].sum().idxmax()
    best_prod   = df.groupby("Product")["Revenue"].sum().idxmax()
    best_region = df.groupby("Region")["Revenue"].sum().idxmax()
    best_rep    = df.groupby("Sales_Rep")["Revenue"].sum().idxmax()

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │           BUSINESS PERFORMANCE SUMMARY          │
  ├─────────────────────────────────────────────────┤
  │  Total Revenue       : ${total_rev:>12,.2f}           │
  │  Total Profit        : ${total_prof:>12,.2f}           │
  │  Avg Profit Margin   : {avg_margin:>11.1f}%           │
  │  Total Units Sold    : {total_units:>12,}           │
  │  Total Orders        : {total_orders:>12,}           │
  ├─────────────────────────────────────────────────┤
  │  Best Category       : {best_cat:<26}│
  │  Best Product        : {best_prod:<26}│
  │  Best Region         : {best_region:<26}│
  │  Top Sales Rep       : {best_rep:<26}│
  ├─────────────────────────────────────────────────┤
  │  Forecast Model Used : {best_model:<26}│
  │  6-Month Forecast:                              │""")
    for _, row in future_df.iterrows():
        print(f"  │    {row['Period']}  →  ${row['Forecast_Revenue']:>10,.2f}                 │")
    print("  └─────────────────────────────────────────────────┘")


# ================================================================
# HELPER
# ================================================================
def _save(filename: str):
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n╔" + "═" * 58 + "╗")
    print("║   SALES DATA ANALYSIS AND FORECASTING SYSTEM          ║")
    print("╚" + "═" * 58 + "╝")

    df         = load_data()
    df         = preprocess(df)
    eda(df)
    monthly    = feature_engineering(df)
    future_df, best_model = forecast(monthly)
    build_dashboard(df, monthly, future_df, best_model)
    summary_report(df, future_df, best_model)

    print("\n" + "═" * 60)
    print(f"  ✅  All outputs saved to: {OUTPUT_DIR}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
