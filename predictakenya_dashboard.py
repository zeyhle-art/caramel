"""
KUSTAWI DIGITAL SOLUTIONS LTD - PROPRIETARY SOFTWARE
Product: PredictaKenya™ - AI Sales Forecasting Dashboard
Copyright © 2024 Kustawi Digital Solutions Ltd.
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
import tempfile
import os

from kustawi_ml_engine import (
    PredictaKenyaEngine,
    ProductAnalytics
)

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG (UNCHANGED)
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
    dates = pd.date_range(start='2022-01-01', end='2024-12-3_
