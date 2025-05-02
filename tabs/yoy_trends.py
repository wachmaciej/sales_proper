# tabs/yoy_trends.py
import streamlit as st
import pandas as pd
import datetime
from plotting import create_yoy_trends_chart
from utils import get_custom_week_date_range # For summary table
from config import (
    CUSTOM_YEAR_COL, WEEK_AS_INT_COL, QUARTER_COL, SALES_VALUE_GBP_COL,
    LISTING_COL, PRODUCT_COL, SALES_CHANNEL_COL, SEASON_COL,
    CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL
)

def display_tab(df, available_years, default_years):
    """Displays the YOY Trends tab."""
    st.markdown("### YOY Weekly Revenue Trends")

    # --- Filters ---
    with st.expander("Chart Filters", expanded=True):
        # ... (filter code remains the same) ...
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            yoy_years = st.multiselect("Year(s)", options=available_years, default=default_years, key="yoy_years_tab")
        with col2:
            selected_season = "ALL"
            if SEASON_COL in df.columns:
                season_options_data = sorted(df[SEASON_COL].dropna().unique())
                filtered_season_data = [season for season in season_options_data if season != "AYR"]
                season_options = ["ALL"] + filtered_season_data
                selected_season = st.selectbox("Season", options=season_options, index=0, key="yoy_season_tab", help="Filter data by season. Select ALL to include all seasons.")
            else:
                st.caption(f"{SEASON_COL} filter unavailable (column missing)")
        with col3:
            selected_channels = []
            if SALES_CHANNEL_COL in df.columns:
                channel_options = sorted(df[SALES_CHANNEL_COL].dropna().unique())
                selected_channels = st.multiselect("Channel(s)", options=channel_options, default=[], key="yoy_channels_tab")
            else:
                st.caption(f"{SALES_CHANNEL_COL} filter unavailable (column missing)")
        with col4:
            selected_listings = []
            if LISTING_COL in df.columns:
                listing_options = sorted(df[LISTING_COL].dropna().unique())
                selected_listings = st.multiselect("Listing(s)", options=listing_options, default=[], key="yoy_listings_tab")
            else:
                st.caption(f"{LISTING_COL} filter unavailable (column missing)")
        with col5:
            selected_products = []
            if PRODUCT_COL in df.columns:
                if selected_listings:
                    product_options = sorted(df[df[LISTING_COL].isin(selected_listings)][PRODUCT_COL].dropna().unique())
                else:
                    product_options = sorted(df[PRODUCT_COL].dropna().unique())
                selected_products = st.multiselect("Product(s)", options=product_options, default=[], key="yoy_products_tab")
            else:
                st.caption(f"{PRODUCT_COL} filter unavailable (column missing)")
        with col6:
            week_range_yoy = st.slider("Select Week Range", min_value=1, max_value=53, value=(1, 53), step=1, key="yoy_week_range_tab", help="Filter the YOY chart and summary table by week number.")


    # --- Create and Display Chart ---
    # ... (chart display code remains the same) ...
    time_grouping = "Week"
    if not yoy_years:
        st.warning("Please select at least one year in the filters to display the YOY chart.")
    else:
        fig_yoy = create_yoy_trends_chart(df, yoy_years, selected_channels, selected_listings, selected_products, time_grouping=time_grouping, week_range=week_range_yoy, selected_season=selected_season)
        st.plotly_chart(fig_yoy, use_container_width=True)


    # --- Revenue Summary Table ---
    st.markdown("### Revenue Summary")
    st.markdown("")

    # ... (filtering logic for summary table remains the same) ...
    filtered_df_summary = df.copy()
    if yoy_years:
        filtered_df_summary = filtered_df_summary[filtered_df_summary[CUSTOM_YEAR_COL].isin([int(y) for y in yoy_years])]
    if SEASON_COL in filtered_df_summary.columns and selected_season != "ALL":
        filtered_df_summary = filtered_df_summary[filtered_df_summary[SEASON_COL] == selected_season]
    if selected_channels:
        if SALES_CHANNEL_COL in filtered_df_summary.columns:
              filtered_df_summary = filtered_df_summary[filtered_df_summary[SALES_CHANNEL_COL].isin(selected_channels)]
    if selected_listings:
        if LISTING_COL in filtered_df_summary.columns:
              filtered_df_summary = filtered_df_summary[filtered_df_summary[LISTING_COL].isin(selected_listings)]
    if selected_products:
        if PRODUCT_COL in filtered_df_summary.columns:
              filtered_df_summary = filtered_df_summary[filtered_df_summary[PRODUCT_COL].isin(selected_products)]
    if week_range_yoy:
        start_week, end_week = week_range_yoy
        if WEEK_AS_INT_COL in filtered_df_summary.columns:
            filtered_df_summary[WEEK_AS_INT_COL] = pd.to_numeric(filtered_df_summary[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
            filtered_df_summary.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
            if not filtered_df_summary.empty:
                filtered_df_summary = filtered_df_summary[(filtered_df_summary[WEEK_AS_INT_COL] >= start_week) & (filtered_df_summary[WEEK_AS_INT_COL] <= end_week)]
        else:
            st.warning(f"Column '{WEEK_AS_INT_COL}' not found for week range filtering in Revenue Summary.")


    # --- Revenue summary table calculation code ---
    if filtered_df_summary.empty:
        st.info("No data available for the selected filters (including season and week range) to build the revenue summary table.")
    else:
        # ... (data cleaning and year/week determination logic remains the same) ...
        filtered_df_summary[CUSTOM_YEAR_COL] = pd.to_numeric(filtered_df_summary[CUSTOM_YEAR_COL], errors='coerce').astype('Int64')
        filtered_df_summary[WEEK_AS_INT_COL] = pd.to_numeric(filtered_df_summary[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
        filtered_df_summary[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered_df_summary[SALES_VALUE_GBP_COL], errors='coerce')
        required_summary_cols = [CUSTOM_YEAR_COL, WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL]
        grouping_key = None
        if PRODUCT_COL in filtered_df_summary.columns and selected_listings and len(selected_listings) == 1:
            grouping_key = PRODUCT_COL
        elif LISTING_COL in filtered_df_summary.columns:
            grouping_key = LISTING_COL
        if grouping_key:
              required_summary_cols.append(grouping_key)
        else:
              st.warning("Cannot determine grouping key (Listing/Product) for summary table. Ensure columns exist.")
              return # Exit the function if no grouping key
        filtered_df_summary.dropna(subset=required_summary_cols, inplace=True)

        if filtered_df_summary.empty:
            st.info("No valid data remaining after cleaning for summary table.")
        else:
            # ... (calculation of last week, last 4 weeks, previous year comparisons remains the same) ...
            filtered_years_present = sorted(filtered_df_summary[CUSTOM_YEAR_COL].unique())
            if not filtered_years_present:
                st.info("No valid years found in filtered data for summary.")
            else:
                filtered_current_year = filtered_years_present[-1]
                df_revenue_current = filtered_df_summary[filtered_df_summary[CUSTOM_YEAR_COL] == filtered_current_year].copy()
                if df_revenue_current.empty:
                    st.info(f"No data found for the latest filtered year ({filtered_current_year}) for summary.")
                else:
                    today = datetime.date.today()
                    # Ensure CUSTOM_WEEK_END_COL is date for comparison
                    df_revenue_current[CUSTOM_WEEK_END_COL] = pd.to_datetime(df_revenue_current[CUSTOM_WEEK_END_COL], errors='coerce').dt.date

                    # Separate current week data (for the "Current Week So Far" column)
                    df_current_week = df_revenue_current.copy()
                    # Ensure start date is date type for comparison
                    df_current_week[CUSTOM_WEEK_START_COL] = pd.to_datetime(df_current_week[CUSTOM_WEEK_START_COL], errors='coerce').dt.date

                    # Identify the current in-progress week (where today is between start and end dates)
                    current_week_data = df_current_week[
                        (df_current_week[CUSTOM_WEEK_START_COL] <= today) &
                        (df_current_week[CUSTOM_WEEK_END_COL] >= today)
                    ].copy()

                    current_week_number = None
                    if not current_week_data.empty:
                        # Make sure WEEK_AS_INT_COL exists before accessing
                        if WEEK_AS_INT_COL in current_week_data.columns:
                            current_week_number = current_week_data[WEEK_AS_INT_COL].iloc[0]
                        else:
                             st.warning(f"'{WEEK_AS_INT_COL}' not found in current_week_data.")


                    # Get data for completed weeks only (for the existing columns)
                    df_full_weeks_current = df_revenue_current.dropna(subset=[CUSTOM_WEEK_END_COL, WEEK_AS_INT_COL])
                    df_full_weeks_current = df_full_weeks_current[df_full_weeks_current[CUSTOM_WEEK_END_COL] < today].copy()
                    if df_full_weeks_current.empty:
                          st.info("No *completed* weeks found in the filtered current year data to build the summary.")
                          unique_weeks_current = pd.DataFrame(columns=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL, CUSTOM_WEEK_END_COL]) # Empty df with expected columns
                    else:
                          unique_weeks_current = (df_full_weeks_current.dropna(subset=[WEEK_AS_INT_COL, CUSTOM_YEAR_COL])
                                                  .groupby([CUSTOM_YEAR_COL, WEEK_AS_INT_COL])
                                                  .agg(Week_End=(CUSTOM_WEEK_END_COL, "first"))
                                                  .reset_index()
                                                  .sort_values("Week_End", ascending=True, na_position='first'))

                    if unique_weeks_current.empty or unique_weeks_current['Week_End'].isna().all():
                        st.info("Not enough complete week data in the filtered current year to build the revenue summary table.")
                    else:
                        # ... (calculations for rev_last_4_current, rev_last_1_current, etc. remain the same) ...
                        last_complete_week_row_current = unique_weeks_current.dropna(subset=['Week_End']).iloc[-1]
                        last_week_number = int(last_complete_week_row_current[WEEK_AS_INT_COL])
                        last_week_year = int(last_complete_week_row_current[CUSTOM_YEAR_COL])
                        last_4_weeks_current_df = unique_weeks_current.drop_duplicates(subset=[WEEK_AS_INT_COL]).dropna(subset=['Week_End']).tail(4)
                        last_4_week_numbers = last_4_weeks_current_df[WEEK_AS_INT_COL].astype(int).tolist()
                        rev_last_4_current = (df_full_weeks_current[df_full_weeks_current[WEEK_AS_INT_COL].isin(last_4_week_numbers)]
                                              .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                              .rename(f"Last 4 Weeks Revenue ({last_week_year})").round(0))
                        rev_last_1_current = (df_full_weeks_current[df_full_weeks_current[WEEK_AS_INT_COL] == last_week_number]
                                              .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                              .rename(f"Last Week Revenue ({last_week_year})").round(0))

                        # Calculate Current Week So Far Revenue
                        rev_current_week = pd.Series(dtype='float64', name=f"Current Week So Far ({filtered_current_year})")
                        if current_week_number is not None and not current_week_data.empty:
                            rev_current_week = (current_week_data
                                                .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                                .rename(f"Current Week So Far ({filtered_current_year})").round(0))

                        rev_last_4_last_year = pd.Series(dtype='float64', name="Last 4 Weeks Revenue (Prev Year)")
                        rev_last_1_last_year = pd.Series(dtype='float64', name="Last Week Revenue (Prev Year)")
                        prev_year_label = "Prev Year"
                        last_year = None
                        if last_week_year in filtered_years_present:
                            current_year_index = filtered_years_present.index(last_week_year)
                            if current_year_index > 0:
                                last_year = filtered_years_present[current_year_index - 1]
                                prev_year_label = str(last_year)
                        if last_year is not None:
                            df_revenue_last_year = filtered_df_summary[filtered_df_summary[CUSTOM_YEAR_COL] == last_year].copy()
                            if not df_revenue_last_year.empty:
                                df_revenue_last_year[WEEK_AS_INT_COL] = pd.to_numeric(df_revenue_last_year[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
                                df_revenue_last_year[SALES_VALUE_GBP_COL] = pd.to_numeric(df_revenue_last_year[SALES_VALUE_GBP_COL], errors='coerce')
                                df_revenue_last_year.dropna(subset=[WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, grouping_key], inplace=True)
                                if not df_revenue_last_year.empty:
                                    rev_last_1_last_year = (df_revenue_last_year[df_revenue_last_year[WEEK_AS_INT_COL] == last_week_number]
                                                            .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                                            .rename(f"Last Week Revenue ({last_year})").round(0))
                                    rev_last_4_last_year = (df_revenue_last_year[df_revenue_last_year[WEEK_AS_INT_COL].isin(last_4_week_numbers)]
                                                            .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                                            .rename(f"Last 4 Weeks Revenue ({last_year})").round(0))

                        all_keys = pd.Series(sorted(filtered_df_summary[grouping_key].dropna().unique()), name=grouping_key)
                        revenue_summary = pd.DataFrame({grouping_key: all_keys}).set_index(grouping_key)
                        revenue_summary = revenue_summary.join(rev_last_4_current)\
                                                         .join(rev_last_1_current)\
                                                         .join(rev_current_week)\
                                                         .join(rev_last_4_last_year.rename(f"Last 4 Weeks Revenue ({prev_year_label})"))\
                                                         .join(rev_last_1_last_year.rename(f"Last Week Revenue ({prev_year_label})"))
                        revenue_summary = revenue_summary.fillna(0)
                        current_4wk_col = f"Last 4 Weeks Revenue ({last_week_year})"
                        prev_4wk_col = f"Last 4 Weeks Revenue ({prev_year_label})"
                        current_1wk_col = f"Last Week Revenue ({last_week_year})"
                        prev_1wk_col = f"Last Week Revenue ({prev_year_label})"
                        current_week_col = f"Current Week So Far ({filtered_current_year})"

                        if current_4wk_col in revenue_summary.columns and prev_4wk_col in revenue_summary.columns:
                                revenue_summary["Last 4 Weeks Diff"] = revenue_summary[current_4wk_col] - revenue_summary[prev_4wk_col]
                        else: revenue_summary["Last 4 Weeks Diff"] = 0
                        if current_1wk_col in revenue_summary.columns and prev_1wk_col in revenue_summary.columns:
                            revenue_summary["Last Week Diff"] = revenue_summary[current_1wk_col] - revenue_summary[prev_1wk_col]
                        else: revenue_summary["Last Week Diff"] = 0
                        revenue_summary["Last 4 Weeks % Change"] = revenue_summary.apply(
                            lambda row: (row["Last 4 Weeks Diff"] / row[prev_4wk_col] * 100)
                            if prev_4wk_col in row and row[prev_4wk_col] != 0 else
                            (100.0 if "Last 4 Weeks Diff" in row and row["Last 4 Weeks Diff"] > 0 else 0.0), axis=1)
                        revenue_summary["Last Week % Change"] = revenue_summary.apply(
                            lambda row: (row["Last Week Diff"] / row[prev_1wk_col] * 100)
                            if prev_1wk_col in row and row[prev_1wk_col] != 0 else
                            (100.0 if "Last Week Diff" in row and row["Last Week Diff"] > 0 else 0.0), axis=1)
                        revenue_summary = revenue_summary.reset_index()

                        # Add calculation for Current Week % Change (compared to Last Week)
                        if current_week_col in revenue_summary.columns and current_1wk_col in revenue_summary.columns:
                            revenue_summary["Current Week % Change"] = revenue_summary.apply(
                                lambda row: ((row[current_week_col] / row[current_1wk_col] * 100) - 100) # Changed formula to ((curr/prev)*100) - 100
                                if row[current_1wk_col] != 0 else
                                (100.0 if row[current_week_col] > 0 else 0.0), axis=1)
                        else:
                            revenue_summary["Current Week % Change"] = 0

                        # Update the desired column order - move current_week_col to after "Last Week % Change"
                        desired_order = [grouping_key,
                                         current_4wk_col, prev_4wk_col, "Last 4 Weeks Diff", "Last 4 Weeks % Change",
                                         current_1wk_col, prev_1wk_col, "Last Week Diff", "Last Week % Change",
                                         current_week_col, "Current Week % Change"] # Added new % col
                        desired_order = [col for col in desired_order if col in revenue_summary.columns] # Ensure only existing columns are included
                        revenue_summary = revenue_summary[desired_order] # Reorder

                        # --- Calculate Total Summary Row ---
                        summary_row = {}
                        for col in desired_order:
                            if col != grouping_key and pd.api.types.is_numeric_dtype(revenue_summary[col]):
                                summary_row[col] = revenue_summary[col].sum()
                            else:
                                summary_row[col] = '' # Keep grouping key col blank for now

                        summary_row[grouping_key] = "Total" # Set grouping key value

                        # Get summed totals needed for percentage recalculation in the total row
                        total_last4_last_year = summary_row.get(prev_4wk_col, 0)
                        total_last_week_last_year = summary_row.get(prev_1wk_col, 0)
                        #total_last4_current = summary_row.get(current_4wk_col, 0) # Not directly needed if using diff
                        total_last_week_current = summary_row.get(current_1wk_col, 0) # Needed for Current Week %
                        total_current_week_so_far = summary_row.get(current_week_col, 0) # Needed for Current Week %
                        total_diff_4wk = summary_row.get("Last 4 Weeks Diff", 0)
                        total_diff_1wk = summary_row.get("Last Week Diff", 0)

                        # Recalculate % Change for the Total row using summed values
                        summary_row["Last 4 Weeks % Change"] = (total_diff_4wk / total_last4_last_year * 100) if total_last4_last_year != 0 else (100.0 if total_diff_4wk > 0 else 0.0)
                        summary_row["Last Week % Change"] = (total_diff_1wk / total_last_week_last_year * 100) if total_last_week_last_year != 0 else (100.0 if total_diff_1wk > 0 else 0.0)

                        # --- START: Added Correct Calculation for Total Current Week % Change ---
                        if current_week_col in summary_row and current_1wk_col in summary_row:
                            if total_last_week_current != 0:
                                 # Calculate using the formula: ((current / previous) - 1) * 100
                                summary_row["Current Week % Change"] = ((total_current_week_so_far / total_last_week_current) - 1) * 100
                            else:
                                # Handle division by zero: 100% if current > 0, else 0%
                                summary_row["Current Week % Change"] = 100.0 if total_current_week_so_far > 0 else 0.0
                        else:
                            # Ensure the column exists even if inputs are missing, default to 0
                            summary_row["Current Week % Change"] = 0.0
                        # --- END: Added Correct Calculation for Total Current Week % Change ---

                        # Create Total DataFrame using the corrected summary_row
                        total_df = pd.DataFrame([summary_row])[desired_order]


                        # --- Styling and Display ---
                        def color_diff(val):
                            # ... (color_diff function remains the same) ...
                            try:
                                val = float(val)
                                if val < -0.001: return 'color: red'
                                elif val > 0.001: return 'color: green'
                                else: return ''
                            except (ValueError, TypeError):
                                return ''

                        formats = {}
                        # ... (formats dictionary setup remains the same) ...
                        if current_4wk_col in revenue_summary.columns: formats[current_4wk_col] = "£{:,.0f}"
                        if prev_4wk_col in revenue_summary.columns: formats[prev_4wk_col] = "£{:,.0f}"
                        if current_1wk_col in revenue_summary.columns: formats[current_1wk_col] = "£{:,.0f}"
                        if prev_1wk_col in revenue_summary.columns: formats[prev_1wk_col] = "£{:,.0f}"
                        if current_week_col in revenue_summary.columns: formats[current_week_col] = "£{:,.0f}" # Format for new column
                        if "Last 4 Weeks Diff" in revenue_summary.columns: formats["Last 4 Weeks Diff"] = "{:,.0f}"
                        if "Last Week Diff" in revenue_summary.columns: formats["Last Week Diff"] = "{:,.0f}"
                        if "Last 4 Weeks % Change" in revenue_summary.columns: formats["Last 4 Weeks % Change"] = "{:.1f}%"
                        if "Last Week % Change" in revenue_summary.columns: formats["Last Week % Change"] = "{:.1f}%"
                        if "Current Week % Change" in revenue_summary.columns: formats["Current Week % Change"] = "{:.1f}%" # Format for new % column


                        color_cols = [col for col in ["Last 4 Weeks Diff", "Last Week Diff", "Last 4 Weeks % Change", "Last Week % Change", "Current Week % Change"] if col in revenue_summary.columns]

                        # --- CORRECTED STYLING with st.dataframe ---
                        # Style Total Row
                        styled_total = total_df.style.format(formats, na_rep='-') \
                                                     .apply(lambda x: x.map(color_diff), subset=color_cols) \
                                                     .set_properties(**{'font-weight': 'bold'}) \
                                                     .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                        {'selector': 'td', 'props': [('text-align', 'right')]}]) \
                                                     .hide(axis="index") # <<< Use hide(axis="index")

                        st.markdown("##### Total Summary")
                        # Use st.dataframe to render the Styler object
                        st.dataframe(styled_total, use_container_width=True)

                        # Style Main Table
                        # Convert relevant value columns to integer AFTER calculations, before styling
                        value_cols_to_int = [col for col in [current_4wk_col, prev_4wk_col, current_1wk_col, prev_1wk_col, current_week_col, "Last 4 Weeks Diff", "Last Week Diff"] if col in revenue_summary.columns]
                        if value_cols_to_int and not revenue_summary.empty:
                           # Ensure columns exist before trying conversion
                           existing_cols_to_int = [col for col in value_cols_to_int if col in revenue_summary.columns]
                           if existing_cols_to_int:
                               # Convert only existing columns, handling potential NaNs introduced by joins before fillna(0)
                                for col in existing_cols_to_int:
                                     # Use errors='ignore' if direct astype(int) fails due to non-finite numbers
                                     revenue_summary[col] = pd.to_numeric(revenue_summary[col], errors='coerce').fillna(0).astype(int)


                        styled_main = revenue_summary.style.format(formats, na_rep='-') \
                                                           .apply(lambda x: x.map(color_diff), subset=color_cols) \
                                                           .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                              {'selector': 'td', 'props': [('text-align', 'right')]}]) \
                                                           .hide(axis="index") # <<< Use hide(axis="index")

                        st.markdown("##### Detailed Summary")
                        # Use st.dataframe to render the Styler object
                        st.dataframe(styled_main, use_container_width=True)
                        # --- END OF CORRECTED STYLING ---

                        # If there's a current week, display the date range information
                        if current_week_number is not None and not current_week_data.empty:
                            # Safely access dates only if they exist
                            current_week_start = current_week_data[CUSTOM_WEEK_START_COL].iloc[0] if CUSTOM_WEEK_START_COL in current_week_data.columns else "N/A"
                            current_week_end = current_week_data[CUSTOM_WEEK_END_COL].iloc[0] if CUSTOM_WEEK_END_COL in current_week_data.columns else "N/A"
                           
