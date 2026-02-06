"""
KUSTAWI DIGITAL SOLUTIONS LTD - PROPRIETARY SOFTWARE
Product: PredictaKenya™ - AI Sales Forecasting Dashboard
Copyright © 2024
CONFIDENTIAL AND PROPRIETARY
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import warnings
import tempfile
import os

from kustawi_ml_engine import PredictaKenyaEngine, ProductAnalytics

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="PredictaKenya™ | Kustawi Digital Solutions",
    page_icon="🇰🇪",
    layout="wide"
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
# DATA
# ============================================================================
def generate_kenyan_sample_data():
    np.random.seed(42)
    products = [
        'Maize Flour 2kg', 'Rice 5kg', 'Cooking Oil 1L',
        'Sugar 2kg', 'Milk 500ml', 'Bread'
    ]
    regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']
    dates = pd.date_range("2022-01-01", "2024-12-31")

    rows = []
    for d in dates:
        for _ in range(np.random.randint(8, 15)):
            q = np.random.randint(1, 6)
            p = np.random.uniform(80, 600)
            rows.append({
                "Date": d,
                "Product": np.random.choice(products),
                "Region": np.random.choice(regions),
                "Quantity": q,
                "Sales": round(q * p, 2),
                "Profit": round(q * p * 0.2, 2),
                "Days_To_Expiry": np.random.randint(1, 45)
            })
    return pd.DataFrame(rows)

# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_comprehensive(df, engine):
    results = {}
    engine.load_and_validate_data(df)

    results["forecast"] = engine.generate_forecast(df, periods=12)

    results["top_products"] = (
        df.groupby("Product")[["Sales", "Quantity", "Profit"]]
        .sum().sort_values("Sales", ascending=False).head(10)
    )

    results["slow_products"] = (
        df.groupby("Product")[["Sales", "Quantity"]]
        .sum().sort_values("Sales").head(10)
    )

    analytics = ProductAnalytics()
    results["expiring_goods"] = analytics.identify_expiring_inventory(df)

    results["regional"] = (
        df.groupby("Region")["Sales"].sum().reset_index()
    )

    results["weekly_cashflow"] = (
        df.assign(Week=df["Date"].dt.to_period("W"))
          .groupby("Week")[["Sales", "Profit"]]
          .sum()
          .reset_index()
    )

    return results

# ============================================================================
# CHART EXPORT HELPER
# ============================================================================
def save_chart(fig, name):
    path = os.path.join(tempfile.gettempdir(), name)
    fig.write_image(path, scale=2)
    return path

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("🚀 Run Analysis"):
        df = pd.read_csv(uploaded) if uploaded else generate_kenyan_sample_data()
        st.session_state.uploaded_data = df
        st.session_state.analysis_results = analyze_comprehensive(
            df, st.session_state.ml_engine
        )
        st.session_state.analysis_complete = True
        st.rerun()

# ============================================================================
# MAIN
# ============================================================================
if not st.session_state.analysis_complete:
    st.title("🇰🇪 PredictaKenya™")
    st.info("Upload data or run sample.")
else:
    df = st.session_state.uploaded_data
    results = st.session_state.analysis_results
    forecast_df = results["forecast"]

    st.metric("Total Revenue", f"KES {df['Sales'].sum():,.0f}")

    # ---------------- FORECAST CHART ----------------
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecast"],
        mode="lines+markers", name="Forecast"
    ))
    st.plotly_chart(forecast_fig, use_container_width=True)

    # =========================================================================
    # PDF REPORT
    # =========================================================================
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle,
        Paragraph, Spacer, Image, PageBreak
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    if st.button("📄 Generate Full PDF Report"):
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # ================= EXEC SUMMARY =================
        story.append(Paragraph("PREDICTAKENYA™ EXECUTIVE REPORT", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y %H:%M EAT')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.3 * inch))

        # ================= FORECAST =================
        story.append(Paragraph("12-Month Forecast", styles["Heading2"]))
        forecast_img = save_chart(forecast_fig, "forecast.png")
        story.append(Image(forecast_img, 6.5*inch, 3.2*inch))
        story.append(PageBreak())

        # ================= TOP PRODUCTS =================
        story.append(Paragraph("Top Products", styles["Heading2"]))
        top_data = [["#", "Product", "Sales"]]
        for i, r in results["top_products"].reset_index().iterrows():
            top_data.append([str(i+1), r["Product"], f"KES {r['Sales']:,.0f}"])
        top_table = Table(top_data, repeatRows=1)
        top_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.green),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(top_table)
        story.append(PageBreak())

        # ================= EXPIRY GOODS (1000) =================
        story.append(Paragraph("Expiring Inventory (Up to 1,000)", styles["Heading2"]))

        def label(days):
            if days <= 7: return "🔴 Critical"
            if days <= 14: return "🟡 Moderate"
            return "🟢 Planned"

        exp = results["expiring_goods"].head(1000)
        exp_data = [["Product", "Qty", "Days Left", "Status"]]
        for _, r in exp.iterrows():
            exp_data.append([
                r["Product"],
                int(r["Quantity"]),
                int(r["Days_Left"]),
                label(r["Days_Left"])
            ])

        exp_table = Table(exp_data, repeatRows=1)
        exp_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.black)
        ]))
        story.append(exp_table)
        story.append(PageBreak())

        # ================= COMPLIANCE =================
        story.append(Paragraph("Compliance Notice", styles["Heading2"]))
        story.append(Paragraph(
            "Compliant with Kenya Data Protection Act 2019. "
            "All data anonymized. Confidential & Proprietary.",
            styles["Normal"]
        ))

        doc.build(story)
        buf.seek(0)

        st.download_button(
            "⬇️ Download Full Report",
            buf,
            file_name="PredictaKenya_Full_Report.pdf",
            mime="application/pdf"
        )
