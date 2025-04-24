# tabs/unrecognised_sales.py
import streamlit as st
import pandas as pd
from config import ( # Import column names
    LISTING_COL, DATE_COL, WEEK_AS_INT_COL, SALES_CHANNEL_COL, SKU_COL, PRODUCT_COL, # <<< Corrected import name
    SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW, SALES_VALUE_TRANS_CURRENCY_COL,
    ORIGINAL_CURRENCY_COL,
    # Columns to potentially drop if they exist from old calculations
    YEAR_COL_RAW, CUSTOM_WEEK_COL, CUSTOM_YEAR_COL, CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL, QUARTER_COL
)

def display_tab(df):
    """Displays the Unrecognised Sales tab."""
    st.markdown("### Unrecognised Sales")

    if LISTING_COL not in df.columns:
        st.error(f"Column '{LISTING_COL}' not found. Cannot identify unrecognised sales.")
        return

    # Ensure Listing column is string type before searching
    df[LISTING_COL] = df[LISTING_COL].astype(str)
    # Filter rows where the listing contains 'unrecognised' (case-insensitive)
    unrecognised_sales = df[df[LISTING_COL].str.contains("unrecognised", case=False, na=False)].copy()

    # Columns to remove from the display (mostly derived/intermediate columns)
    columns_to_drop_orig = [
        # Original raw year if not needed, derived columns etc.
        YEAR_COL_RAW, CUSTOM_WEEK_COL, CUSTOM_YEAR_COL,
        CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL, QUARTER_COL,
        # Add any other intermediate columns you definitely don't want shown
        "Weekly Sales Value (£)", "YOY Growth (%)" # From original code, likely don't exist anymore
        ]
    # Find which of these columns actually exist in the filtered dataframe
    columns_to_drop_existing = [col for col in columns_to_drop_orig if col in unrecognised_sales.columns]

    if columns_to_drop_existing:
        unrecognised_sales = unrecognised_sales.drop(columns=columns_to_drop_existing, errors='ignore')

    if unrecognised_sales.empty:
        st.info("No unrecognised sales found based on 'Listing' column containing 'unrecognised'.")
    else:
        st.info(f"Found {len(unrecognised_sales)} rows potentially related to unrecognised sales.")

        # Define preferred order for displaying columns, using SALES_CHANNEL_COL
        display_cols_order = [
             DATE_COL, WEEK_AS_INT_COL, SALES_CHANNEL_COL, LISTING_COL, SKU_COL, PRODUCT_COL, # <<< Use correct variable
             SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW, SALES_VALUE_TRANS_CURRENCY_COL, ORIGINAL_CURRENCY_COL
        ]
        # Get columns that exist in the dataframe in the preferred order
        display_cols_existing = [col for col in display_cols_order if col in unrecognised_sales.columns]
        # Get any remaining columns not in the preferred list
        remaining_cols = sorted([col for col in unrecognised_sales.columns if col not in display_cols_existing])
        # Combine them
        final_display_cols = display_cols_existing + remaining_cols

        # Define formatting for specific columns
        style_format = {}
        if SALES_VALUE_GBP_COL in final_display_cols:
             style_format[SALES_VALUE_GBP_COL] = "£{:,.2f}"
        if SALES_VALUE_TRANS_CURRENCY_COL in final_display_cols:
             # Basic number formatting, currency symbol handled by ORIGINAL_CURRENCY_COL
             style_format[SALES_VALUE_TRANS_CURRENCY_COL] = "{:,.2f}"
        if ORDER_QTY_COL_RAW in final_display_cols:
             style_format[ORDER_QTY_COL_RAW] = "{:,.0f}" # Integer formatting for quantity


        # Display the dataframe without the index
        st.dataframe(
             unrecognised_sales[final_display_cols].style.format(style_format, na_rep='-'),
             use_container_width=True,
             hide_index=True
             )

