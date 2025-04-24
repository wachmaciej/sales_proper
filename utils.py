# utils.py
import pandas as pd
import datetime
import calendar # Keep if needed, but get_custom_week_date_range doesn't use it directly
from config import CUSTOM_YEAR_COL, WEEK_AS_INT_COL # Import relevant config

# --- Formatting Functions ---
def format_currency(value):
    """Formats a numeric value as currency (£)."""
    if pd.isna(value): return "£N/A" # Handle NaN values
    try:
        return f"£{float(value):,.2f}"
    except (ValueError, TypeError):
        return "£Error" # Handle non-numeric input

def format_currency_int(value):
    """Formats a numeric value as integer currency (£)."""
    if pd.isna(value): return "£N/A" # Handle NaN values
    try:
        # Round before converting to int to handle decimals properly
        return f"£{int(round(float(value))):,}"
    except (ValueError, TypeError):
        return "£Error" # Handle non-numeric input

# --- ADDED: Dynamic Currency Formatter ---
def format_dynamic_currency(value, symbol=""):
    """Formats a numeric value as currency with a dynamic symbol."""
    if pd.isna(value): return "-" # Handle NaN values with a dash or N/A as preferred
    try:
        # Basic symbol placement, adjust if needed for specific currencies (e.g., EUR often has symbol after)
        return f"{symbol}{float(value):,.2f}"
    except (ValueError, TypeError):
        return "Error" # Handle non-numeric input
# --- END ADDED ---


# --- Date/Week Functions ---
def get_custom_week_date_range(week_year, week_number):
    """Gets the start and end date for a given custom week year and number (Sat-Fri)."""
    try:
        week_year = int(week_year)
        week_number = int(week_number)
        # Calculate the start of the first week of the week_year
        first_day = datetime.date(week_year, 1, 1)
        # Saturday=0, Sunday=1, ..., Friday=6
        first_day_custom_dow = (first_day.weekday() + 2) % 7
        first_week_start = first_day - datetime.timedelta(days=first_day_custom_dow)

        # Calculate the start date of the requested week
        # Subtract 1 because week numbers start from 1
        week_start = first_week_start + datetime.timedelta(weeks=week_number - 1)
        week_end = week_start + datetime.timedelta(days=6)
        return week_start, week_end
    except (ValueError, TypeError) as e:
        # Consider logging this warning instead of using st.warning if utils shouldn't depend on streamlit
        # print(f"Warning: Invalid input for get_custom_week_date_range: Year={week_year}, Week={week_number}. Error: {e}")
        return None, None
