"""
KUSTAWI DIGITAL SOLUTIONS LTD - PROPRIETARY SOFTWARE
Product: PredictaKenya™ - AI Sales Forecasting Dashboard
Copyright © 2024 Kustawi Digital Solutions Ltd. All Rights Reserved.
CONFIDENTIAL AND PROPRIETARY
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import warnings

from kustawi_ml_engine import (
    PredictaKenyaEngine,
    ProductAnalytics
)

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="PredictaKenya™ | Kustawi Digital Solutions",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "ml_engine" not in st.session_state:
    st.session_state.ml_engine = PredictaKenyaEngine()

# ============================================================================
# DATA GENERATION (UNCHANGED)
# ============================================================================
def generate_kenyan_sample_data():
    np.random.seed(42)
    products = [
        'Maize Flour 2kg', 'Rice 5kg', 'Cooking Oil 1L', 'Sugar 2kg',
        'Milk 500ml', 'Bread', 'Tea Leaves 250g', 'Wheat Flour 2kg'
    ]
    regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')

    data = []
    for date in dates:
        for _ in range(np.random.randint(10, 20)):
            product = np.random.choice(products)
            region = np.random.choice(regions)
            quantity = np.random.randint(1, 6)
            price = np.random.uniform(50, 500)
            sales = quantity * price
            profit = sales * np.random.uniform(0.1, 0.3)

            data.append({
                "Date": date,
                "Product": product,
                "Region": region,
                "Quantity": quantity,
                "Sales": round(sales, 2),
                "Profit": round(profit, 2),
                "Days_To_Expiry": np.random.randint(1, 90)
            })

    return pd.DataFrame(data)

# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_comprehensive(df, engine):
    results = {}

    df_processed = engine.load_and_validate_data(df)
    metrics = engine.train_model(df_processed)
    forecast = engine.generate_forecast(df_processed, periods=12)

    results["metrics"] = metrics
    results["forecast"] = forecast

    results["top_products"] = (
        df.groupby("Product")[["Sales", "Quantity", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    results["slow_products"] = (
        df.groupby("Product")[["Sales", "Quantity"]]
        .sum()
        .sort_values("Sales", ascending=True)
        .head(10)
    )

    analytics = ProductAnalytics()
    results["expiring_goods"] = analytics.identify_expiring_inventory(df)

    return results

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use Sample Data", value=True)

    if st.button("🚀 Run Analysis"):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = generate_kenyan_sample_data()

        st.session_state.uploaded_data = df
        st.session_state.analysis_results = analyze_comprehensive(
            df, st.session_state.ml_engine
        )
        st.session_state.analysis_complete = True
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================
if not st.session_state.analysis_complete:
    st.title("🇰🇪 PredictaKenya™")
    st.info("Upload data or use sample to begin.")
else:
    results = st.session_state.analysis_results
    df = st.session_state.uploaded_data

    st.header("📊 Executive Dashboard")
    st.metric("Total Revenue", f"KES {df['Sales'].sum():,.0f}")

    # =========================================================================
    # PDF REPORT
    # =========================================================================
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    if st.button("📄 Generate PDF Report"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("PREDICTAKENYA™ SALES REPORT", styles["Title"]))

        # ==========================
        # ✅ FIXED TOP PRODUCTS TABLE
        # ==========================
        top_data = [["Rank", "Product", "Total Sales"]]

        for idx, row in results["top_products"].reset_index().iterrows():
            top_data.append([
                str(idx + 1),
                row["Product"],
                f"KES {row['Sales']:,.0f}"
            ])

        top_table = Table(top_data, colWidths=[1*inch, 3*inch, 2*inch])
        top_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.green),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(top_table)

        # ==========================
        # ✅ FIXED SLOW PRODUCTS TABLE
        # ==========================
        slow_data = [["Rank", "Product", "Total Sales"]]

        for idx, row in results["slow_products"].reset_index().iterrows():
            slow_data.append([
                str(idx + 1),
                row["Product"],
                f"KES {row['Sales']:,.0f}"
            ])

        slow_table = Table(slow_data, colWidths=[1*inch, 3*inch, 2*inch])
        slow_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.red),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(slow_table)

        doc.build(story)
        buffer.seek(0)

        st.download_button(
            "⬇️ Download PDF",
            buffer,
            file_name="PredictaKenya_Report.pdf",
            mime="application/pdf"
        )
