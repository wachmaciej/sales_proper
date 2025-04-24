# tabs/pivot_table.py
import streamlit as st
import pandas as pd
from plotting import create_pivot_table
from utils import get_custom_week_date_range # For header dates
from config import (
    CUSTOM_YEAR_COL, QUARTER_COL, SALES_CHANNEL_COL, LISTING_COL, PRODUCT_COL, # <<< Corrected import name
    WEEK_AS_INT_COL
)

def display_tab(df, available_years, default_years):
    """Displays the Pivot Table tab."""
    st.markdown("### Pivot Table: Revenue by Week")

    with st.expander("Pivot Table Filters", expanded=False):
        # Use default_years which defaults to current year if only one exists
        pivot_years = st.multiselect("Select Year(s) for Pivot Table", options=available_years, default=default_years, key="pivot_years_tab", help="Select year(s) to include in the table.")

        # Quarter filter remains active for this tab
        pivot_quarters = []
        if QUARTER_COL in df.columns:
            pivot_quarter_opts = sorted(df[QUARTER_COL].dropna().unique())
            pivot_quarters = st.multiselect("Select Quarter(s)", options=pivot_quarter_opts, default=[], key="pivot_quarters_tab", help="Select one or more quarters to filter by. Default shows all.")
        else:
            st.caption(f"{QUARTER_COL} filter unavailable")

        # Use SALES_CHANNEL_COL for filter logic
        pivot_channels = []
        if SALES_CHANNEL_COL in df.columns: # <<< Use correct variable
            pivot_channel_opts = sorted(df[SALES_CHANNEL_COL].dropna().unique())
            pivot_channels = st.multiselect("Select Sales Channel(s)", options=pivot_channel_opts, default=[], key="pivot_channels_tab", help="Select one or more channels to filter. If empty, all channels are shown.")
        else:
            st.caption(f"{SALES_CHANNEL_COL} filter unavailable") # <<< Use correct variable

        pivot_listings = []
        if LISTING_COL in df.columns:
            pivot_listing_opts = sorted(df[LISTING_COL].dropna().unique())
            pivot_listings = st.multiselect("Select Listing(s)", options=pivot_listing_opts, default=[], key="pivot_listings_tab", help="Select one or more listings to filter. If empty, all listings are shown.")
        else:
            st.caption(f"{LISTING_COL} filter unavailable")

        # Product filter depends on Listing selection
        pivot_products = []
        if PRODUCT_COL in df.columns:
            if pivot_listings: # Filter products based on selected listings
                 pivot_product_options = sorted(df[df[LISTING_COL].isin(pivot_listings)][PRODUCT_COL].dropna().unique())
            else: # Show all products if no listing is selected
                 pivot_product_options = sorted(df[PRODUCT_COL].dropna().unique())
            pivot_products = st.multiselect("Select Product(s)", options=pivot_product_options, default=[], key="pivot_products_tab", help="Select one or more products to filter. Options depend on selected listings.")
        else:
             st.caption(f"{PRODUCT_COL} filter unavailable")


    if not pivot_years:
        st.warning("Please select at least one year for the Pivot Table.")
    else:
        # Determine grouping key: Product if one Listing selected, else Listing
        grouping_key = None
        if PRODUCT_COL in df.columns and pivot_listings and len(pivot_listings) == 1:
             grouping_key = PRODUCT_COL
        elif LISTING_COL in df.columns:
             grouping_key = LISTING_COL

        if grouping_key:
            # If grouping by Listing, selected_products should effectively be empty for the pivot function
            # If grouping by Product, selected_products is used.
            effective_products = pivot_products if grouping_key == PRODUCT_COL else []

            # Handle empty quarter selection - pass all available quarters if none selected
            effective_quarters = pivot_quarters if pivot_quarters else (df[QUARTER_COL].dropna().unique().tolist() if QUARTER_COL in df else [])


            pivot = create_pivot_table(
                df,
                selected_years=pivot_years,
                selected_quarters=effective_quarters, # Pass selected or all
                selected_channels=pivot_channels,
                selected_listings=pivot_listings,
                selected_products=effective_products, # Pass selected products only if grouping by product
                grouping_key=grouping_key
            )

            # Check if the returned pivot table is valid data or an error message dataframe
            is_real_pivot = isinstance(pivot, pd.DataFrame) and \
                            not pivot.empty and \
                            pivot.index.name == grouping_key and \
                            pivot.index[0] not in ["No data", "Missing grouping column", "Missing 'Week' column", "No results", "No valid data", "Missing 'Week' column"] # Added check


            # Add multi-index header with dates if only ONE year is selected
            if len(pivot_years) == 1 and is_real_pivot:
                 try:
                     year_for_date = int(pivot_years[0])
                     new_columns_tuples = []
                     for col_name in pivot.columns:
                         if col_name == "Total Revenue":
                             new_columns_tuples.append(("Total Revenue", "")) # Top level, no date range
                         elif isinstance(col_name, str) and col_name.startswith("Week "):
                             try:
                                 week_number = int(col_name.split()[1])
                                 # Use helper to get date range
                                 start_dt, end_dt = get_custom_week_date_range(year_for_date, week_number)
                                 date_range_str = f"{start_dt.strftime('%d %b')} - {end_dt.strftime('%d %b')}" if start_dt and end_dt else ""
                                 new_columns_tuples.append((col_name, date_range_str)) # Top: Week X, Bottom: Date Range
                             except (IndexError, ValueError, TypeError):
                                 new_columns_tuples.append((col_name, "")) # Fallback
                         else:
                             new_columns_tuples.append((str(col_name), "")) # Handle other potential columns

                     if new_columns_tuples:
                          pivot.columns = pd.MultiIndex.from_tuples(new_columns_tuples, names=["Metric", "Date Range"])

                 except Exception as e:
                      st.warning(f"Could not create multi-index header for pivot table: {e}")


            # Display the pivot table (either styled or the error message)
            if is_real_pivot:
                 # Format numbers with commas, handle single and multi-index
                 format_dict = {}
                 if isinstance(pivot.columns, pd.MultiIndex):
                      # Format based on the top level 'Metric' name
                      format_dict = {col_tuple: "{:,.0f}" for col_tuple in pivot.columns}
                 elif isinstance(pivot.columns, pd.Index): # Handle single index case
                      format_dict = {col: "{:,.0f}" for col in pivot.columns}

                 # Apply formatting, replacing NaN with 0 for display
                 st.dataframe(pivot.style.format(format_dict, na_rep='0'), use_container_width=True)
            else:
                 # Display the dataframe containing the warning/error message
                 st.dataframe(pivot, use_container_width=True)

        else:
             st.error(f"Cannot create pivot table: Required grouping column ('{LISTING_COL}' or '{PRODUCT_COL}') not found or insufficient data.")
