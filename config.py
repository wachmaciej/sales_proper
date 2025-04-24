# config.py

# --- Google Sheet Configuration ---
# Constants for Sheet Name and Worksheet Name are now read from secrets.toml
# GOOGLE_SHEET_NAME = "RA_sales_dashboard_data"
# WORKSHEET_NAME = "SAP_DATA"
# GOOGLE_SHEET_KEY constant is also removed (read from secrets or extracted from URL)

# Path to your service account key file for local development fallback.
# Ensure this file exists if you're not using secrets for GCP auth locally.
LOCAL_KEY_PATH = 'service_account.json'

# --- Google API Scopes ---
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file' # Needed for discovery/listing sometimes
]

# --- File Paths ---
LOGO_PATH = "assets/logo.png" # Path relative to the main script location

# --- Dashboard Constants ---
# List of listings to feature specifically on the Daily Prices tab
MAIN_LISTINGS_FOR_DAILY_PRICE = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Patterned Polos"]

# --- Column Name Constants ---
# Define constants for all column names used in the project.
# This helps prevent typos and makes it easy to update if source names change.

# >> Raw Data Columns (match names in your Google Sheet) <<
DATE_COL = "Date"
REVENUE_COL_RAW = "Revenue" # Example if you have a raw 'Revenue' column
WEEK_COL_RAW = "Week" # Example if you have a raw 'Week' column from source
ORDER_QTY_COL_RAW = "Order Quantity"
SALES_VALUE_TRANS_CURRENCY_COL = "Sales Value in Transaction Currency"
SALES_VALUE_GBP_COL = "Sales Value (£)" # Primary sales value column used
YEAR_COL_RAW = "Year" # Example if you have a raw calendar 'Year' column
SALES_CHANNEL_COL = "Sales Channel"
LISTING_COL = "Listing"
PRODUCT_COL = "Product"
SKU_COL = "Product SKU"
ORIGINAL_CURRENCY_COL = "Original Currency"
SEASON_COL = "Season"

# >> Derived Columns (created during preprocessing) <<
CUSTOM_WEEK_COL = "Custom_Week" # The calculated Sat-Fri week number (1-53)
CUSTOM_YEAR_COL = "Custom_Week_Year" # The calculated year based on custom week logic
CUSTOM_WEEK_START_COL = "Custom_Week_Start" # Calculated start date of the custom week
CUSTOM_WEEK_END_COL = "Custom_Week_End" # Calculated end date of the custom week
QUARTER_COL = "Quarter" # Derived Quarter (Q1-Q4) based on custom week
WEEK_AS_INT_COL = "Week" # The column used for weekly filtering/display after assigning Custom_Week

# --- Type Conversion Configuration ---
# Use the defined constants above in these lists for data_loader.py

# List of columns to attempt conversion to numeric during data loading
# Add/remove constants based on your actual raw numeric columns
NUMERIC_COLS_CONFIG = [
    REVENUE_COL_RAW,
    WEEK_COL_RAW,
    ORDER_QTY_COL_RAW,
    SALES_VALUE_TRANS_CURRENCY_COL,
    SALES_VALUE_GBP_COL,
    YEAR_COL_RAW
]

# List of columns to attempt conversion to datetime during data loading
# Add/remove constants based on your actual raw date columns
DATE_COLS_CONFIG = [
    DATE_COL
]

# --- Add any other configuration constants needed below ---

