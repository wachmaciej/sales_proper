# tabs/category_summary.py
import streamlit as st
import pandas as pd
import numpy as np
from config import (
    # Need LISTING for category assignment, PRODUCT for grouping within category
    LISTING_COL, PRODUCT_COL, ORDER_QTY_COL_RAW, SALES_VALUE_GBP_COL, SKU_COL,
    CUSTOM_YEAR_COL, WEEK_AS_INT_COL, SALES_CHANNEL_COL, SEASON_COL, QUARTER_COL
)
from utils import format_currency_int

# --- Define Custom Categories and Keywords ---
CATEGORY_KEYWORDS = {
    "Trousers": ["pants", "trousers", "trews"],
    "Polos": ["polo", "shirt"],
    "Shorts": ["shorts"],
    "Caps": ["cap", "solid hat", "patterned hat"],
    "Skirts": ["skirt", "skort"],
    "Jackets": ["jacket", "waterproof"],
    "Quarter Zips": ["quarter zip", "1/4 zip", "qtr zip", "midlayer"],
    "Knickers": ["knickers", "plus 2s"],
    "Baseball Caps": ["baseball cap"],
    "Accessories": ["belt", "socks", "accessory", "accessories", "towel", "marker", "glove", "headcover", "bag", "umbrella"],
}

# --- Define the DESIRED DISPLAY ORDER ---
ORDERED_CATEGORIES = [
    "Trousers",
    "Shorts",
    "Polos",
    "Caps",
    "Knickers",
    "Skirts",
    "Jackets",
    "Quarter Zips",
    "Baseball Caps",
    "Accessories",
]

# --- Helper function to assign category ---
def assign_category(listing_name):
    """Assigns a broader category based on keywords in the listing name."""
    if pd.isna(listing_name): return "Other"
    listing_lower = str(listing_name).lower()
    for category in ORDERED_CATEGORIES:
        keywords = CATEGORY_KEYWORDS.get(category, [])
        for keyword in keywords:
            if keyword in listing_lower:
                return category
    return "Other"

# --- Main Display Function ---
def display_tab(df, available_years, default_years):
    """Displays the Category Summary tab using custom category definitions."""
    st.markdown("### Sales Summary by Category")

    # --- Determine Filter Defaults ---
    default_summary_year = [2025] if 2025 in available_years else ([available_years[-1]] if available_years else [])
    # Calculate default week (latest week in default year)
    max_week_for_default = 1 # Default to week 1 if calculation fails
    if default_summary_year and WEEK_AS_INT_COL in df.columns and CUSTOM_YEAR_COL in df.columns:
        try:
            year_for_week_calc = int(default_summary_year[0])
            df[CUSTOM_YEAR_COL] = pd.to_numeric(df[CUSTOM_YEAR_COL], errors='coerce')
            df_default_year = df[df[CUSTOM_YEAR_COL] == year_for_week_calc].copy()
            if not df_default_year.empty:
                weeks_in_year = pd.to_numeric(df_default_year[WEEK_AS_INT_COL], errors='coerce')
                if weeks_in_year.notna().any():
                    max_week_for_default = int(weeks_in_year.max())
        except Exception as e:
            st.warning(f"Could not determine max week for default year {default_summary_year}: {e}")
            max_week_for_default = 1 # Reset to 1 on error

    # --- Filters ---
    with st.expander("Filters", expanded=True): # Expander open by default
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        with f_col1:
            summary_years = st.multiselect(
                "Year(s)", options=available_years, default=default_summary_year, key="category_summary_years"
            )
        with f_col2:
            selected_season = "ALL"
            if SEASON_COL in df.columns:
                season_options_data = sorted(df[SEASON_COL].dropna().astype(str).unique())
                filtered_season_data = [s for s in season_options_data if s and s.strip() and s.upper() != "ALL" and s.upper() != "AYR"]
                season_options = ["ALL"] + filtered_season_data
                selected_season = st.selectbox(
                    "Season", options=season_options, index=0, key="category_summary_season"
                )
            else: st.caption(f"{SEASON_COL} filter unavailable")
        with f_col3:
            selected_channels = []
            if SALES_CHANNEL_COL in df.columns:
                channel_options = sorted(df[SALES_CHANNEL_COL].dropna().astype(str).unique())
                selected_channels = st.multiselect(
                    "Channel(s)", options=channel_options, default=[], key="category_summary_channels"
                )
            else: st.caption(f"{SALES_CHANNEL_COL} filter unavailable")
        with f_col4:
            selected_quarters = []
            if QUARTER_COL in df.columns:
                 quarter_opts = sorted(df[QUARTER_COL].dropna().astype(str).unique())
                 selected_quarters = st.multiselect("Quarter(s)", options=quarter_opts, default=[], key="category_summary_quarters")
            else: st.caption(f"{QUARTER_COL} filter unavailable")

        # === Week Range Selection using Number Inputs ===
        with f_col5:
            st.markdown("###### Week Range") # Add a label for clarity
            w_col1, w_col2 = st.columns(2) # Use sub-columns for Start/End
            with w_col1:
                summary_start_week = st.number_input(
                    "Start", min_value=1, max_value=53,
                    value=max_week_for_default, # Default to latest week
                    step=1, key="cat_sum_start_week", label_visibility="collapsed" # Use collapsed label
                )
            with w_col2:
                summary_end_week = st.number_input(
                    "End", min_value=1, max_value=53,
                    value=max_week_for_default, # Default to latest week
                    step=1, key="cat_sum_end_week", label_visibility="collapsed" # Use collapsed label
                )

            # Add validation after the inputs
            if summary_end_week < summary_start_week:
                st.warning("End Week cannot be before Start Week. Results shown for Start Week only.", icon="⚠️")
                summary_end_week = summary_start_week # Adjust end week for filtering
        # === End Week Range Selection ===


    # --- Data Filtering ---
    if not summary_years:
        st.warning("Please select at least one Year in the filters.")
        return

    filtered_df = df.copy()

    # Filter by Year
    if CUSTOM_YEAR_COL not in filtered_df.columns:
         st.error(f"Critical Error: Year column '{CUSTOM_YEAR_COL}' not found.")
         st.stop()
    filtered_df[CUSTOM_YEAR_COL] = pd.to_numeric(filtered_df[CUSTOM_YEAR_COL], errors='coerce')
    filtered_df.dropna(subset=[CUSTOM_YEAR_COL], inplace=True)
    filtered_df[CUSTOM_YEAR_COL] = filtered_df[CUSTOM_YEAR_COL].astype(int)
    filtered_df = filtered_df[filtered_df[CUSTOM_YEAR_COL].isin(summary_years)]

    # Apply other filters (Season, Channel, Quarter)
    if SEASON_COL in filtered_df.columns and selected_season != "ALL":
        filtered_df[SEASON_COL] = filtered_df[SEASON_COL].astype(str)
        filtered_df = filtered_df[filtered_df[SEASON_COL] == selected_season]
    if selected_channels and SALES_CHANNEL_COL in filtered_df.columns:
        filtered_df[SALES_CHANNEL_COL] = filtered_df[SALES_CHANNEL_COL].astype(str)
        filtered_df = filtered_df[filtered_df[SALES_CHANNEL_COL].isin(selected_channels)]
    if selected_quarters and QUARTER_COL in filtered_df.columns:
        filtered_df[QUARTER_COL] = filtered_df[QUARTER_COL].astype(str)
        filtered_df = filtered_df[filtered_df[QUARTER_COL].isin(selected_quarters)]

    # Apply Week Range Filter using values from number inputs
    start_week, end_week = summary_start_week, summary_end_week # Use the validated values
    if WEEK_AS_INT_COL in filtered_df.columns:
        filtered_df[WEEK_AS_INT_COL] = pd.to_numeric(filtered_df[WEEK_AS_INT_COL], errors='coerce')
        filtered_df.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
        if not filtered_df.empty:
             filtered_df = filtered_df[
                 (filtered_df[WEEK_AS_INT_COL] >= start_week) &
                 (filtered_df[WEEK_AS_INT_COL] <= end_week)
             ]

    # --- Check for required columns for display/aggregation ---
    required_display_cols = {LISTING_COL, PRODUCT_COL, ORDER_QTY_COL_RAW, SALES_VALUE_GBP_COL}
    missing_display_cols = required_display_cols.difference(filtered_df.columns)
    if missing_display_cols:
        st.error(f"Data is missing required columns for this view after filtering: {missing_display_cols}")
        return

    # --- Assign Custom Categories (based on LISTING_COL) ---
    filtered_df[LISTING_COL] = filtered_df[LISTING_COL].astype(str)
    filtered_df['Broader Category'] = filtered_df[LISTING_COL].apply(assign_category)

    # Convert measure columns to numeric
    filtered_df[ORDER_QTY_COL_RAW] = pd.to_numeric(filtered_df[ORDER_QTY_COL_RAW], errors='coerce')
    filtered_df[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered_df[SALES_VALUE_GBP_COL], errors='coerce')

    # Drop rows missing essential info for aggregation after category assignment
    filtered_df[PRODUCT_COL] = filtered_df[PRODUCT_COL].astype(str)
    filtered_df.dropna(
        subset=[LISTING_COL, PRODUCT_COL, ORDER_QTY_COL_RAW, SALES_VALUE_GBP_COL, 'Broader Category'],
        inplace=True
    )

    if filtered_df.empty:
        st.info("No data available matching the selected filters after category assignment and cleaning.")
        return

    # --- Generate Tables per Custom Category ---
    categories_in_data = sorted(filtered_df['Broader Category'].unique())
    categories_to_show = [cat for cat in ORDERED_CATEGORIES if cat in categories_in_data]
    if "Other" in categories_in_data and "Other" not in categories_to_show:
        categories_to_show.append("Other")

    if not categories_to_show:
         st.info("No matching product categories found in the filtered data.")
         return

    st.markdown("---")

    for category in categories_to_show:
        st.subheader(category)
        df_category = filtered_df[filtered_df['Broader Category'] == category].copy()

        if df_category.empty:
            st.caption("No products found for this category.")
            continue

        # Aggregate by PRODUCT_COL within the broader category
        df_category[PRODUCT_COL] = df_category[PRODUCT_COL].astype(str)
        agg_category = df_category.groupby(PRODUCT_COL).agg(
            Items_Purchased=(ORDER_QTY_COL_RAW, 'sum'),
            Item_Revenue=(SALES_VALUE_GBP_COL, 'sum')
        ).reset_index()

        # Rename PRODUCT_COL to "Item name"
        agg_category.rename(columns={
            PRODUCT_COL: "Item name",
            "Items_Purchased": "Items purchased",
            "Item_Revenue": "Item revenue"
        }, inplace=True)

        # Sort by Items purchased descending (default sort)
        agg_category.sort_values(by="Items purchased", ascending=False, inplace=True)

        # Calculate Grand Total for the Broader Category
        grand_total_units = pd.to_numeric(agg_category["Items purchased"], errors='coerce').sum()
        grand_total_revenue = pd.to_numeric(agg_category["Item revenue"], errors='coerce').sum()

        # Prepare dataframe for display
        display_df = agg_category.copy()
        display_df["Items purchased"] = pd.to_numeric(display_df["Items purchased"], errors='coerce').fillna(0).astype(int)
        display_df["Item revenue"] = pd.to_numeric(display_df["Item revenue"], errors='coerce').apply(format_currency_int)

        # Display the table using st.dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Item name": st.column_config.TextColumn("Item name", help="Product Name", width="large"),
                "Items purchased": st.column_config.NumberColumn("Items purchased", help="Total units sold across all SKUs for this product", format="%d", width="small"),
                "Item revenue": st.column_config.TextColumn("Item revenue", help="Total revenue generated (£) across all SKUs for this product", width="medium")
            }
        )

        # Display Grand Total below the table
        st.markdown(
             f"**Grand total:** &nbsp;&nbsp;&nbsp; {int(grand_total_units):,} Items"
             f" &nbsp;&nbsp;&nbsp; {format_currency_int(grand_total_revenue)}"
        )
        st.markdown("---") # Separator between categories