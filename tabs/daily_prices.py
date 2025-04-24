# tabs/daily_prices.py
import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotting import create_daily_price_chart
from config import (
    CUSTOM_YEAR_COL, QUARTER_COL, SALES_CHANNEL_COL, LISTING_COL, # <<< Corrected import name
    WEEK_AS_INT_COL, MAIN_LISTINGS_FOR_DAILY_PRICE # Use constant from config
)

def display_tab(df, available_years, default_years):
    """Displays the Daily Prices tab."""
    st.markdown("### Daily Prices for Top Listings")

    # --- Filters for Top Listings ---
    with st.expander("Daily Price Filters (Top Listings)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Default to last 2 years for daily price view if available
            today_year = datetime.date.today().year
            default_daily_years = [year for year in available_years if year >= today_year - 1]
            if not default_daily_years: default_daily_years = default_years # Fallback to main default
            elif len(default_daily_years) > 2: default_daily_years = default_daily_years[-2:] # Limit to last 2

            selected_daily_years = st.multiselect(
                "Select Year(s)",
                options=available_years,
                default=default_daily_years,
                key="daily_years_top",
                help="Select year(s) to display daily prices for top listings."
                )
        with col2:
            # Quarter filter remains active for this tab
            quarter_options_daily = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            quarter_selection_daily = st.selectbox("Quarter(s)", options=quarter_options_daily, index=0, key="quarter_dropdown_daily_top")

            selected_daily_quarters = []
            if quarter_selection_daily == "Custom...":
                if QUARTER_COL in df.columns:
                    quarter_opts_daily = sorted(df[QUARTER_COL].dropna().unique())
                    selected_daily_quarters = st.multiselect("Select Quarter(s)", options=quarter_opts_daily, default=[], key="daily_quarters_custom_top", help="Select one or more quarters to filter.")
                else:
                    st.caption(f"{QUARTER_COL} filter unavailable")
            elif quarter_selection_daily == "All Quarters":
                selected_daily_quarters = df[QUARTER_COL].dropna().unique().tolist() if QUARTER_COL in df.columns else ["Q1", "Q2", "Q3", "Q4"]
            else:
                selected_daily_quarters = [quarter_selection_daily]
        # Use SALES_CHANNEL_COL for filter logic
        with col3:
             selected_daily_channels = []
             if SALES_CHANNEL_COL in df.columns: # <<< Use correct variable
                 channel_options_daily = sorted(df[SALES_CHANNEL_COL].dropna().unique())
                 selected_daily_channels = st.multiselect("Select Sales Channel(s)", options=channel_options_daily, default=[], key="daily_channels_top", help="Select one or more sales channels to filter the daily price data.")
             else:
                 st.caption(f"{SALES_CHANNEL_COL} filter unavailable")
        with col4:
             daily_week_range = st.slider(
                 "Select Week Range",
                 min_value=1,
                 max_value=53,
                 value=(1, 53),
                 step=1,
                 key="daily_week_range_top",
                 help="Select the range of weeks to display in the Daily Prices section."
                 )

    # --- Display Charts for Main Listings ---
    # Use the list from config.py
    main_listings = MAIN_LISTINGS_FOR_DAILY_PRICE

    if LISTING_COL not in df.columns:
        st.error(f"Column '{LISTING_COL}' not found. Cannot display Daily Price charts.")
    else:
        # Find which of the main listings are actually present in the data
        available_main_listings = [l for l in main_listings if l in df[LISTING_COL].unique()]
        if not available_main_listings:
             st.warning("None of the specified main listings found in the data.")
        else:
            if not selected_daily_years:
                 st.warning("Please select at least one year in the filters to view Daily Price charts.")
            else:
                for listing in available_main_listings:
                    st.subheader(listing)
                    # Pass selected filters to the chart function
                    fig_daily = create_daily_price_chart(
                        df, listing, selected_daily_years, selected_daily_quarters,
                        selected_daily_channels, week_range=daily_week_range
                        )
                    # Check if the figure object is valid before trying to plot
                    if fig_daily and isinstance(fig_daily, go.Figure):
                         st.plotly_chart(fig_daily, use_container_width=True)
                    # else: # Optional: message if chart generation failed for a specific listing
                    #    st.info(f"Could not generate daily price chart for {listing} with current filters.")


    # --- Daily Prices Comparison Section ---
    st.markdown("---") # Separator
    st.markdown("### Daily Prices Comparison")
    with st.expander("Comparison Chart Filters", expanded=False):
        comp_col1, comp_col2, comp_col3 = st.columns(3) # Year, Quarter, Channel
        comp_col4 = st.container() # Listing select below columns

        with comp_col1:
            # Use the same default logic as the top section for consistency
            today_year_comp = datetime.date.today().year
            default_comp_years = [year for year in available_years if year >= today_year_comp - 1]
            if not default_comp_years: default_comp_years = default_years
            elif len(default_comp_years) > 2: default_comp_years = default_comp_years[-2:]

            comp_years = st.multiselect("Select Year(s)", options=available_years, default=default_comp_years, key="comp_years", help="Select the year(s) for the comparison chart.")
        with comp_col2:
            # Quarter filter for comparison section
            comp_quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            comp_quarter_selection = st.selectbox("Quarter(s)", options=comp_quarter_options, index=0, key="quarter_dropdown_comp_prices")

            comp_quarters = []
            if comp_quarter_selection == "Custom...":
                if QUARTER_COL in df.columns:
                    comp_quarter_opts = sorted(df[QUARTER_COL].dropna().unique())
                    comp_quarters = st.multiselect("Select Quarter(s)", options=comp_quarter_opts, default=[], key="comp_quarters_custom", help="Select one or more quarters for comparison.")
                else: st.caption(f"{QUARTER_COL} filter unavailable")
            elif comp_quarter_selection == "All Quarters":
                comp_quarters = df[QUARTER_COL].dropna().unique().tolist() if QUARTER_COL in df.columns else ["Q1", "Q2", "Q3", "Q4"]
            else:
                comp_quarters = [comp_quarter_selection]
        # Use SALES_CHANNEL_COL for filter logic
        with comp_col3:
            comp_channels = []
            if SALES_CHANNEL_COL in df.columns: # <<< Use correct variable
                comp_channel_opts = sorted(df[SALES_CHANNEL_COL].dropna().unique())
                comp_channels = st.multiselect("Select Sales Channel(s)", options=comp_channel_opts, default=[], key="comp_channels", help="Select the sales channel(s) for the comparison chart.")
            else:
                st.caption(f"{SALES_CHANNEL_COL} filter unavailable")

        # Listing selection for comparison chart (outside the columns)
        comp_listing = None
        with comp_col4:
             if LISTING_COL in df.columns:
                 comp_listing_opts = sorted(df[LISTING_COL].dropna().unique())
                 # Find default index safely
                 default_comp_listing = comp_listing_opts[0] if comp_listing_opts else None
                 # Ensure index is valid if options exist
                 comp_listing_index = comp_listing_opts.index(default_comp_listing) if default_comp_listing and default_comp_listing in comp_listing_opts else 0
                 comp_listing = st.selectbox("Select Listing", options=comp_listing_opts, index=comp_listing_index if comp_listing_opts else 0, key="comp_listing", help="Select a listing for daily prices comparison.")
             else:
                 st.warning("Listing selection unavailable (column missing)")

    # Generate and display comparison chart
    if comp_listing and comp_years:
        # Pass comp_quarters to the function, No week range slider for comparison
        fig_comp = create_daily_price_chart(df, comp_listing, comp_years, comp_quarters, comp_channels, week_range=None)
        if fig_comp and isinstance(fig_comp, go.Figure):
             st.plotly_chart(fig_comp, use_container_width=True)
    elif not comp_listing and LISTING_COL in df.columns:
        st.info("Select a listing in the comparison filters above to view the comparison chart.")
    elif not comp_years:
        st.info("Select at least one year in the comparison filters above to view the comparison chart.")

