"""
Vercel auto-detects a FastAPI app named `app` here and routes matching
requests to it (see vercel.json). Reads from data/risk_scores.csv, which
gets overwritten by the weekly retrain workflow and pushed to main, which
triggers a Vercel redeploy. No db, just git as the data store.
"""

import os
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "risk_scores.csv")
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")

app = FastAPI(title="Churn Dashboard API", version="1.0.0")

PROFILE_COLUMNS = {
    "Contract": "contract_type", "tenure": "tenure", "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges", "InternetService": "internet_service",
    "PaymentMethod": "payment_method", "PaperlessBilling": "paperless_billing",
    "gender": "gender", "SeniorCitizen": "senior_citizen", "Partner": "partner",
    "Dependents": "dependents", "PhoneService": "phone_service", "MultipleLines": "multiple_lines",
    "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection", "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
}


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    risk = pd.read_csv(DATA_PATH)
    profile = pd.read_csv(PROFILE_PATH)
    profile["TotalCharges"] = pd.to_numeric(profile["TotalCharges"], errors="coerce").fillna(0)
    profile = profile.rename(columns=PROFILE_COLUMNS)[["customerID", *PROFILE_COLUMNS.values()]]
    merged = risk.merge(profile, left_on="customer_id", right_on="customerID", how="left").drop(columns="customerID")
    return merged


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/kpis")
def kpis():
    df = load_data()
    total = len(df)
    high = int((df["risk_tier"] == "High").sum())
    medium = int((df["risk_tier"] == "Medium").sum())
    low = int((df["risk_tier"] == "Low").sum())
    revenue_at_risk = round(float(df.loc[df["risk_tier"] == "High", "monthly_charges"].sum()), 2)
    return {
        "total_customers": total,
        "high_risk_count": high,
        "medium_risk_count": medium,
        "low_risk_count": low,
        "high_risk_pct": round(high / total, 4) if total else 0,
        "revenue_at_risk_monthly": revenue_at_risk,
    }


@app.get("/api/customers")
def customers(
    risk_tier: Optional[str] = Query(None, description="High, Medium, or Low"),
    search: Optional[str] = Query(None, description="Substring match on customer_id"),
    sort_by: str = Query("risk_score"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(0, ge=0),
    page_size: int = Query(25, ge=1, le=200),
):
    df = load_data()

    if risk_tier:
        df = df[df["risk_tier"].str.lower() == risk_tier.lower()]
    if search:
        df = df[df["customer_id"].str.contains(search, case=False, na=False)]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))

    total = len(df)
    start = page * page_size
    page_df = df.iloc[start:start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": page_df.to_dict(orient="records"),
    }


@app.get("/api/customers/{customer_id}")
def customer_detail(customer_id: str):
    df = load_data()
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    return row.iloc[0].to_dict()
