# preprocessing.py
import streamlit as st
import pandas as pd
import datetime
import traceback
# from utils import get_custom_week_date_range # If you need this specific function here
from config import (
    DATE_COL, SALES_VALUE_GBP_COL, WEEK_AS_INT_COL, QUARTER_COL,
    CUSTOM_YEAR_COL, CUSTOM_WEEK_COL, CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL,
    LISTING_COL, PRODUCT_COL, SKU_COL, SALES_CHANNEL_COL, SEASON_COL, # <<< Corrected import name
    ORDER_QTY_COL_RAW, SALES_VALUE_TRANS_CURRENCY_COL, ORIGINAL_CURRENCY_COL,
    YEAR_COL_RAW, REVENUE_COL_RAW, WEEK_COL_RAW # Added other potential imports from config
)

# =============================================================================
# Helper Functions for Sales Data (kept here as they are core to preprocessing)
# =============================================================================
def compute_custom_week(date):
    """Computes the custom week number (Sat-Fri), year, start, and end dates."""
    # Ensure input is a date object
    if not isinstance(date, datetime.date):
        # Handle cases where input might be NaT or other non-date types
        return None, None, None, None
    try:
        custom_dow = (date.weekday() + 2) % 7  # Saturday=0, Sunday=1, ..., Friday=6
        week_start = date - datetime.timedelta(days=custom_dow)
        week_end = week_start + datetime.timedelta(days=6)
        custom_year = week_end.year # Year is defined by the week's end date

        # Calculate the start of the first week of the custom_year
        first_day_of_year = datetime.date(custom_year, 1, 1)
        first_day_of_year_custom_dow = (first_day_of_year.weekday() + 2) % 7
        first_week_start_of_year = first_day_of_year - datetime.timedelta(days=first_day_of_year_custom_dow)

        # Handle edge case where the week_start might fall in the *previous* calendar year
        # but belongs to the first week of the custom_year if first_week_start_of_year is also in the previous year.
        if week_start < first_week_start_of_year:
            # This week likely belongs to the previous year's week numbering
            custom_year -= 1
            # Recalculate first week start for the adjusted previous year
            first_day_of_year = datetime.date(custom_year, 1, 1)
            first_day_of_year_custom_dow = (first_day_of_year.weekday() + 2) % 7
            first_week_start_of_year = first_day_of_year - datetime.timedelta(days=first_day_of_year_custom_dow)

        # Calculate the custom week number
        custom_week = ((week_start - first_week_start_of_year).days // 7) + 1

        # Optional: Sanity check/adjustment logic for week 53/1 transition if needed
        # (Current code keeps it simple, assigns based on calculation above)
        # if custom_week == 53: ...

        return custom_week, custom_year, week_start, week_end
    except Exception as e:
        # Log or handle error appropriately - using st.error here links it to Streamlit
        st.error(f"Error computing custom week for date {date}: {e}")
        return None, None, None, None

def get_quarter(week):
    """Determines the quarter based on the custom week number."""
    if pd.isna(week): return None
    try:
        week = int(week)
        if 1 <= week <= 13:
            return "Q1"
        elif 14 <= week <= 26:
            return "Q2"
        elif 27 <= week <= 39:
            return "Q3"
        elif 40 <= week <= 53: # Assuming up to 53 weeks possible
            return "Q4"
        else:
            return None
    except (ValueError, TypeError):
        return None # Handle non-integer weeks

# =============================================================================
# Main Preprocessing Function
# =============================================================================
@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(data):
    """Preprocesses the loaded data: converts types, calculates custom weeks/quarters."""
    df = data.copy() # Work on a copy to avoid modifying cached data
    # st.info("Starting data preprocessing...")

    # --- Initial Data Validation ---
    # Check for essential columns needed *before* processing
    required_input_cols = {DATE_COL, SALES_VALUE_GBP_COL} # Adjust if other raw cols are essential
    if not required_input_cols.issubset(set(df.columns)):
        missing_cols = required_input_cols.difference(set(df.columns))
        st.error(f"Input data is missing essential columns required for preprocessing: {missing_cols}")
        st.stop()

    # --- Type Conversion and Cleaning (Safeguard) ---
    # Ensure 'Date' is datetime (should be handled by load_data, but double-check)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=[DATE_COL], inplace=True)
        if len(df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(df)} rows during preprocessing due to invalid '{DATE_COL}' values.")
    else:
        st.error(f"Essential column '{DATE_COL}' not found during preprocessing.")
        st.stop()

    # Ensure 'Sales Value (£)' is numeric (should be handled by load_data)
    if SALES_VALUE_GBP_COL in df.columns:
        df[SALES_VALUE_GBP_COL] = pd.to_numeric(df[SALES_VALUE_GBP_COL], errors='coerce')
        initial_rows_sales = len(df)
        df.dropna(subset=[SALES_VALUE_GBP_COL], inplace=True)
        if len(df) < initial_rows_sales:
            st.warning(f"Removed {initial_rows_sales - len(df)} rows during preprocessing due to invalid '{SALES_VALUE_GBP_COL}' values.")
    else:
        st.error(f"Essential column '{SALES_VALUE_GBP_COL}' not found during preprocessing.")
        st.stop()

    if df.empty:
        st.error("No valid data remaining after initial cleaning during preprocessing.")
        st.stop()

    # --- Feature Engineering ---
    # Calculate custom week details
    try:
        # Ensure we are applying to the .date part of datetime objects
        week_results = df[DATE_COL].apply(lambda d: compute_custom_week(d.date()) if pd.notnull(d) else (None, None, None, None))
        # Unpack results into new columns using config names
        df[[CUSTOM_WEEK_COL, CUSTOM_YEAR_COL, CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL]] = pd.DataFrame(week_results.tolist(), index=df.index)

    except Exception as e:
        st.error(f"Error calculating custom week details during preprocessing: {e}")
        st.error(traceback.format_exc())
        st.stop()

    # Assign 'Week' and 'Quarter' based on calculated custom week
    # IMPORTANT: Overwriting/Assigning the calculated Custom_Week to the 'Week' column name used by filters/charts
    df[WEEK_AS_INT_COL] = df[CUSTOM_WEEK_COL]
    df[QUARTER_COL] = df[WEEK_AS_INT_COL].apply(get_quarter)

    # Convert calculated columns to appropriate integer types (allowing NAs)
    # Use Int64 to handle potential NaNs gracefully
    df[WEEK_AS_INT_COL] = pd.to_numeric(df[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
    df[CUSTOM_YEAR_COL] = pd.to_numeric(df[CUSTOM_YEAR_COL], errors='coerce').astype('Int64')
    df[CUSTOM_WEEK_COL] = pd.to_numeric(df[CUSTOM_WEEK_COL], errors='coerce').astype('Int64') # Also convert original Custom_Week

    # Convert week start/end dates back to datetime if needed for consistency
    df[CUSTOM_WEEK_START_COL] = pd.to_datetime(df[CUSTOM_WEEK_START_COL], errors='coerce')
    df[CUSTOM_WEEK_END_COL] = pd.to_datetime(df[CUSTOM_WEEK_END_COL], errors='coerce')

    # --- Final Validation ---
    # Drop rows where crucial calculated fields are missing
    initial_rows_final = len(df)
    # Ensure all columns needed for filtering/display exist and have valid values
    # Use config variables for column names
    required_calculated_cols = [WEEK_AS_INT_COL, CUSTOM_YEAR_COL, QUARTER_COL]
    df.dropna(subset=required_calculated_cols, inplace=True)
    if len(df) < initial_rows_final:
            st.warning(f"Removed {initial_rows_final - len(df)} rows during preprocessing due to missing calculated week/year/quarter.")

    # Check if essential columns for the dashboard exist AFTER processing
    # Add all columns expected by ANY tab or chart function
    required_output_cols = {
        WEEK_AS_INT_COL, CUSTOM_YEAR_COL, SALES_VALUE_GBP_COL, DATE_COL, QUARTER_COL,
        LISTING_COL, PRODUCT_COL, SKU_COL, SALES_CHANNEL_COL, ORDER_QTY_COL_RAW, # <<< Corrected variable name here too
        SALES_VALUE_TRANS_CURRENCY_COL, ORIGINAL_CURRENCY_COL, SEASON_COL,
        CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL # Needed for summary table date ranges
    }
    missing_cols_output = required_output_cols.difference(set(df.columns))
    if missing_cols_output:
            st.error(f"Preprocessing did not produce all required output columns: {missing_cols_output}")
            st.info("Please check config.py definitions and preprocessing.py logic.")
            # Check config.py and preprocessing logic if columns are unexpected
            st.stop()

    if df.empty:
        st.error("No data remaining after full preprocessing.")
        st.stop()

    # st.success(f"Preprocessing complete. {len(df)} rows ready for analysis.")
    return df
