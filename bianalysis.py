# bianalysis.py
import pandas as pd

def analyze_generalized_financials(financial_data):
    """
    Computes YoY/Period-over-Period growth rates and profit margins dynamically.
    """
    if not financial_data:
        return None

    df = pd.DataFrame(financial_data)
    
    # Ensure numeric types for calculation
    for col in ["Revenue", "Net Profit", "Operating Profit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 1. Dynamic Margin Calculations
    if "Revenue" in df.columns and "Net Profit" in df.columns:
        df["Net Margin (%)"] = (df["Net Profit"] / df["Revenue"] * 100).replace([float('inf'), -float('inf')], 0.0).fillna(0.0).round(2)

    # 2. Dynamic Period-over-Period Growth Rates
    if "Revenue" in df.columns:
        df["Revenue Growth (%)"] = (df["Revenue"].pct_change() * 100).fillna(0.0).round(2)
    if "Net Profit" in df.columns:
        df["Net Profit Growth (%)"] = (df["Net Profit"].pct_change() * 100).fillna(0.0).round(2)

    # 3. Aggregate Summary Metrics (comparing latest vs previous period)
    latest_period = df.iloc[-1]["Period"]
    prev_period = df.iloc[-2]["Period"] if len(df) > 1 else None

    latest_rev_growth = df.iloc[-1].get("Revenue Growth (%)", 0.0) if len(df) > 1 else 0.0
    latest_profit_growth = df.iloc[-1].get("Net Profit Growth (%)", 0.0) if len(df) > 1 else 0.0

    return {
        "df": df,
        "latest_period": latest_period,
        "prev_period": prev_period,
        "latest_rev_growth": latest_rev_growth,
        "latest_profit_growth": latest_profit_growth,
        "total_periods": len(df)
    }