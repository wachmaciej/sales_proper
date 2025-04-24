# main_dashboard.py (or app.py)
import streamlit as st
import pandas as pd
import datetime
import os
import traceback

# Import project modules - order matters for dependencies
from config import LOGO_PATH, CUSTOM_YEAR_COL # Import necessary config constants
from utils import format_currency, format_currency_int # Import if needed globally (less likely now)
from data_loader import load_data_from_gsheet
from processing import preprocess_data # Assuming this file exists and is correct

# Import tab display functions AFTER other modules they might depend on
from tabs import kpi, yoy_trends, daily_prices, sku_trends, pivot_table, unrecognised_sales
from tabs import seasonality_load
from tabs import category_summary # <<< ADD IMPORT FOR NEW TAB

# --- Page Configuration ---
st.set_page_config(page_title="YOY Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# --- Title and Logo ---
col1_title, col2_logo = st.columns([3, 1])
with col1_title:
    st.title("YOY Dashboard 📊") # Using st.title for consistency
with col2_logo:
    # Check if logo file exists before trying to display it
    if os.path.exists(LOGO_PATH):
          st.image(LOGO_PATH, width=300)
    else:
          st.write(" ") # Placeholder if logo not found


# =============================================================================
# Data Loading and Processing Orchestration
# =============================================================================
# Load Data
df_raw = load_data_from_gsheet() # Function handles caching and errors

if df_raw is None or df_raw.empty:
    st.warning("Failed to load data from Google Sheet or the sheet is empty. Dashboard cannot proceed.")
    st.stop() # Stop execution if data loading fails

# Preprocess Data
try:
    # Pass a copy to prevent modifying the cached raw data if preprocess_data alters it
    df = preprocess_data(df_raw.copy()) # Function handles caching and errors
except Exception as e:
    st.error(f"An error occurred during data preprocessing: {e}")
    st.error(traceback.format_exc())
    st.stop() # Stop execution if preprocessing fails

# Check if preprocessing returned valid data
if df is None or df.empty:
    st.error("Data is empty after preprocessing. Please check the 'processing.py' function logic and the source data.")
    st.stop()

# =============================================================================
# Prepare Common Filter Variables (Derived from Processed Data 'df')
# =============================================================================
# These are calculated once and passed to the relevant tabs

if CUSTOM_YEAR_COL not in df.columns:
    st.error(f"Critical Error: '{CUSTOM_YEAR_COL}' column not found after preprocessing.")
    st.stop()

available_custom_years = sorted(pd.to_numeric(df[CUSTOM_YEAR_COL], errors='coerce').dropna().unique().astype(int))

if not available_custom_years:
    st.error(f"No valid '{CUSTOM_YEAR_COL}' data found after preprocessing. Check calculations and sheet content.")
    st.stop()

# Determine current, previous, and default years for filters
current_custom_year = available_custom_years[-1]
prev_custom_year = available_custom_years[-2] if len(available_custom_years) >= 2 else None

# Default for YOY charts/comparisons: current and previous year if available
yoy_default_years = [prev_custom_year, current_custom_year] if prev_custom_year is not None else [current_custom_year]
# Default for single-year views (like Pivot table initially): current year
default_current_year = [current_custom_year]


# =============================================================================
# Define and Display Dashboard Tabs
# =============================================================================
# <<< ADD NEW TAB NAME >>>
tab_names = [
    "KPIs",
    "YOY Trends",
    "Daily Prices",
    "SKU Trends",
    "Pivot Table",
    "Category Summary",
    "Seasonality Load",
    "Unrecognised Sales"
]
# <<< CREATE TAB OBJECTS (add one more variable) >>>
tab_kpi, tab_yoy, tab_daily, tab_sku, tab_pivot, tab_category, tab_seasonality, tab_unrec = st.tabs(tab_names) # Added tab_category

# Render content for each tab by calling its display function
with tab_kpi:
    # Pass the processed dataframe and necessary pre-calculated variables
    kpi.display_tab(df, available_custom_years, current_custom_year)

with tab_yoy:
    yoy_trends.display_tab(df, available_custom_years, yoy_default_years)

with tab_daily:
    daily_prices.display_tab(df, available_custom_years, default_current_year) # Pass default current for comparison section maybe? Adjust as needed

with tab_sku:
    sku_trends.display_tab(df, available_custom_years, yoy_default_years) # Use YOY default for SKU year selection

with tab_pivot:
    pivot_table.display_tab(df, available_custom_years, default_current_year) # Default to current year

with tab_seasonality:
    seasonality_load.display_tab(df, available_custom_years) # Pass df and available years

# <<< ADD WITH BLOCK FOR NEW TAB >>>
with tab_category:
    category_summary.display_tab(df, available_custom_years, yoy_default_years) # Pass df, years, and default years for filters

with tab_unrec:
    unrecognised_sales.display_tab(df) # Likely only needs the dataframe