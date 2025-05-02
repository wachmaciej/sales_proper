# tabs/yoy_trends.py
import streamlit as st
import pandas as pd
import datetime
from plotting import create_yoy_trends_chart # Assumes this function is updated
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
        # --- Top Row Filters (Inclusion) ---
        # Reduced columns for the top row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            yoy_years = st.multiselect("Year(s)", options=available_years, default=default_years, key="yoy_years_tab")

        with col2:
            selected_season = "ALL"
            season_options = ["ALL"] # Default
            if SEASON_COL in df.columns:
                season_options_data = sorted(df[SEASON_COL].dropna().unique())
                filtered_season_data = [season for season in season_options_data if season != "AYR"]
                season_options = ["ALL"] + filtered_season_data
            else:
                st.caption(f"{SEASON_COL} missing")
            selected_season = st.selectbox("Season", options=season_options, index=0, key="yoy_season_tab", help="Filter data by season. Select ALL to include all seasons.")

        with col3:
            selected_channels = []
            channel_options = []
            if SALES_CHANNEL_COL in df.columns:
                channel_options = sorted(df[SALES_CHANNEL_COL].dropna().unique())
            else:
                st.caption(f"{SALES_CHANNEL_COL} missing")
            selected_channels = st.multiselect("Channel(s)", options=channel_options, default=[], key="yoy_channels_tab", help="Select channels to include.")

        with col4:
            selected_listings = []
            listing_options = [] # Define listing_options here to use in exclude filter too
            if LISTING_COL in df.columns:
                listing_options = sorted(df[LISTING_COL].dropna().unique())
            else:
                st.caption(f"{LISTING_COL} missing")
            selected_listings = st.multiselect("Include Listing(s)", options=listing_options, default=[], key="yoy_listings_tab", help="Select specific listings to include (leave blank for all).")

        # Determine available products based on selected listings (if any)
        product_options = []
        if PRODUCT_COL in df.columns:
            temp_df_for_products = df.copy()
            # Filter by selected listings *before* getting product options
            if selected_listings:
                 if LISTING_COL in temp_df_for_products.columns:
                     temp_df_for_products = temp_df_for_products[temp_df_for_products[LISTING_COL].isin(selected_listings)]
            product_options = sorted(temp_df_for_products[PRODUCT_COL].dropna().unique())
        else:
             st.caption(f"{PRODUCT_COL} missing")

        with col5:
            selected_products = []
            # Use the calculated product_options based on selected listings
            selected_products = st.multiselect("Include Product(s)", options=product_options, default=[], key="yoy_products_tab", help="Select specific products to include (leave blank for all relevant).")

        st.markdown("---") # Separator line

        # --- Bottom Row Filters (Exclusion & Time) ---
        # Added a third column for the week slider
        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            excluded_listings = []
            # Use the same listing_options from above
            if LISTING_COL in df.columns:
                excluded_listings = st.multiselect("Exclude Listing(s)", options=listing_options, default=[], key="yoy_exclude_listings_tab", help="Select specific listings to exclude from calculations.")
            # No caption needed if LISTING_COL missing, handled above

        with col_ex2:
            excluded_products = []
            # Use the same product_options determined above (based on selected listings)
            if PRODUCT_COL in df.columns:
                 excluded_products = st.multiselect("Exclude Product(s)", options=product_options, default=[], key="yoy_exclude_products_tab", help="Select specific products to exclude from calculations.")
            # No caption needed if PRODUCT_COL missing, handled above

        # --- MOVED: Week Slider ---
        with col_ex3:
            week_range_yoy = st.slider("Select Week Range", min_value=1, max_value=53, value=(1, 53), step=1, key="yoy_week_range_tab", help="Filter the YOY chart and summary table by week number.")
        # --- END MOVED ---


    # --- Create and Display Chart ---
    time_grouping = "Week"
    if not yoy_years:
        st.warning("Please select at least one year in the filters to display the YOY chart.")
    else:
        # Pass BOTH excluded_listings and excluded_products to the plotting function
        # Ensure your plotting.py's create_yoy_trends_chart accepts these!
        fig_yoy = create_yoy_trends_chart(
            df,
            yoy_years,
            selected_channels=selected_channels,
            selected_listings=selected_listings,
            selected_products=selected_products,
            time_grouping=time_grouping,
            week_range=week_range_yoy,
            selected_season=selected_season,
            excluded_listings=excluded_listings, # <-- Pass excluded listings
            excluded_products=excluded_products  # <-- Pass excluded products
        )
        st.plotly_chart(fig_yoy, use_container_width=True)


    # --- Revenue Summary Table ---
    st.markdown("### Revenue Summary")
    st.markdown("")

    # Start with a copy of the original dataframe
    filtered_df_summary = df.copy()

    # --- Apply Inclusion Filters ---
    if yoy_years:
        if CUSTOM_YEAR_COL in filtered_df_summary.columns:
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

    # --- Apply Exclusion Filters ---
    if excluded_listings:
        if LISTING_COL in filtered_df_summary.columns:
            # Use ~ to negate the isin condition, removing the excluded listings
            filtered_df_summary = filtered_df_summary[~filtered_df_summary[LISTING_COL].isin(excluded_listings)]
    if excluded_products:
        if PRODUCT_COL in filtered_df_summary.columns:
            # Use ~ to negate the isin condition, removing the excluded products
            filtered_df_summary = filtered_df_summary[~filtered_df_summary[PRODUCT_COL].isin(excluded_products)]
    # --- END Exclusion Filters ---

    # Apply week range filter (Last)
    if week_range_yoy:
        start_week, end_week = week_range_yoy
        if WEEK_AS_INT_COL in filtered_df_summary.columns:
            # Ensure week column is numeric, coercing errors and handling potential NaNs
            filtered_df_summary[WEEK_AS_INT_COL] = pd.to_numeric(filtered_df_summary[WEEK_AS_INT_COL], errors='coerce')
            filtered_df_summary.dropna(subset=[WEEK_AS_INT_COL], inplace=True) # Drop rows where week couldn't be converted
            # Convert to Int64 AFTER dropping NaNs
            if not filtered_df_summary.empty:
                 filtered_df_summary[WEEK_AS_INT_COL] = filtered_df_summary[WEEK_AS_INT_COL].astype('Int64')
                 # Apply the week range filter
                 filtered_df_summary = filtered_df_summary[
                     (filtered_df_summary[WEEK_AS_INT_COL] >= start_week) &
                     (filtered_df_summary[WEEK_AS_INT_COL] <= end_week)
                 ]
        else:
            st.warning(f"Column '{WEEK_AS_INT_COL}' not found for week range filtering in Revenue Summary.")


    # --- Revenue summary table calculation code ---
    if filtered_df_summary.empty:
        st.info("No data available for the selected filters (including exclusions, season, and week range) to build the revenue summary table.")
    else:
        # Data cleaning and type conversion
        if CUSTOM_YEAR_COL in filtered_df_summary.columns:
            filtered_df_summary[CUSTOM_YEAR_COL] = pd.to_numeric(filtered_df_summary[CUSTOM_YEAR_COL], errors='coerce').astype('Int64')
        # Week column already handled above during filtering
        if SALES_VALUE_GBP_COL in filtered_df_summary.columns:
            filtered_df_summary[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered_df_summary[SALES_VALUE_GBP_COL], errors='coerce')
        else: # Handle missing sales value column gracefully
             st.error(f"Critical Error: '{SALES_VALUE_GBP_COL}' column missing for Revenue Summary.")
             return

        # Determine grouping key
        grouping_key = None
        # Group by Product only if exactly one Listing is selected (and included, not excluded)
        if PRODUCT_COL in filtered_df_summary.columns and selected_listings and len(selected_listings) == 1:
             # Check if the single selected listing wasn't excluded and still exists after filtering
             single_listing = selected_listings[0]
             listing_exists_after_filter = False
             if LISTING_COL in filtered_df_summary.columns:
                 listing_exists_after_filter = not filtered_df_summary[filtered_df_summary[LISTING_COL] == single_listing].empty

             if listing_exists_after_filter:
                 grouping_key = PRODUCT_COL
             elif LISTING_COL in filtered_df_summary.columns: # Fallback to Listing if the single selected one was excluded or has no data
                 grouping_key = LISTING_COL
        elif LISTING_COL in filtered_df_summary.columns: # Default to Listing if multiple/no listings selected
             grouping_key = LISTING_COL

        if not grouping_key:
             st.warning("Cannot determine grouping key (Listing/Product) for summary table. Ensure relevant columns exist and data is available after filtering.")
             return # Exit the function if no grouping key

        # Define required columns including the determined grouping key and date columns
        required_summary_cols = [CUSTOM_YEAR_COL, WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, grouping_key]
        # Add date columns only if they exist
        if CUSTOM_WEEK_START_COL in filtered_df_summary.columns:
            required_summary_cols.append(CUSTOM_WEEK_START_COL)
        if CUSTOM_WEEK_END_COL in filtered_df_summary.columns:
            required_summary_cols.append(CUSTOM_WEEK_END_COL)

        # Drop rows with NaNs in essential columns needed for calculations
        # Check existence before dropping
        cols_to_drop_na = [col for col in required_summary_cols if col in filtered_df_summary.columns]
        filtered_df_summary.dropna(subset=cols_to_drop_na, inplace=True)


        if filtered_df_summary.empty:
            st.info("No valid data remaining after cleaning for summary table.")
        else:
            # Determine latest year and week based on the *filtered* data
            if CUSTOM_YEAR_COL not in filtered_df_summary.columns:
                 st.error(f"'{CUSTOM_YEAR_COL}' missing after cleaning.")
                 return
            filtered_years_present = sorted(filtered_df_summary[CUSTOM_YEAR_COL].unique())
            if not filtered_years_present:
                st.info("No valid years found in filtered data for summary.")
                return # Exit if no years

            filtered_current_year = filtered_years_present[-1]
            df_revenue_current = filtered_df_summary[filtered_df_summary[CUSTOM_YEAR_COL] == filtered_current_year].copy()

            # Initialize calculation results (in case current year has no data)
            rev_last_4_current = pd.Series(dtype='float64')
            rev_last_1_current = pd.Series(dtype='float64')
            rev_current_week = pd.Series(dtype='float64')
            last_week_year = filtered_current_year # Default year label
            last_year = None
            if len(filtered_years_present) > 1:
                last_year = filtered_years_present[-2]
            prev_year_label = str(last_year) if last_year else "Prev Year"
            rev_last_4_last_year = pd.Series(dtype='float64', name=f"Last 4 Weeks Revenue ({prev_year_label})")
            rev_last_1_last_year = pd.Series(dtype='float64', name=f"Last Week Revenue ({prev_year_label})")
            last_week_number = None # Initialize last week number
            last_4_week_numbers = [] # Initialize last 4 weeks
            current_week_number = None # Initialize current week number

            # Check if date columns exist before proceeding with date logic
            has_start_date = CUSTOM_WEEK_START_COL in df_revenue_current.columns
            has_end_date = CUSTOM_WEEK_END_COL in df_revenue_current.columns
            has_week_num = WEEK_AS_INT_COL in df_revenue_current.columns

            if df_revenue_current.empty:
                st.info(f"No data found for the latest filtered year ({filtered_current_year}) for summary.")
                # Use initialized empty series from above
                rev_last_4_current = pd.Series(dtype='float64', name=f"Last 4 Weeks Revenue ({filtered_current_year})")
                rev_last_1_current = pd.Series(dtype='float64', name=f"Last Week Revenue ({filtered_current_year})")
                rev_current_week = pd.Series(dtype='float64', name=f"Current Week So Far ({filtered_current_year})")
            elif not has_start_date or not has_end_date or not has_week_num:
                 st.warning(f"Missing date/week columns ({CUSTOM_WEEK_START_COL}, {CUSTOM_WEEK_END_COL}, {WEEK_AS_INT_COL}). Cannot calculate weekly summaries accurately.")
                 # Use initialized empty series from above
                 rev_last_4_current = pd.Series(dtype='float64', name=f"Last 4 Weeks Revenue ({filtered_current_year})")
                 rev_last_1_current = pd.Series(dtype='float64', name=f"Last Week Revenue ({filtered_current_year})")
                 rev_current_week = pd.Series(dtype='float64', name=f"Current Week So Far ({filtered_current_year})")
            else: # Proceed only if date columns exist
                today = datetime.date.today()
                # Ensure date columns are date objects for comparison
                df_revenue_current[CUSTOM_WEEK_START_COL] = pd.to_datetime(df_revenue_current[CUSTOM_WEEK_START_COL], errors='coerce').dt.date
                df_revenue_current[CUSTOM_WEEK_END_COL] = pd.to_datetime(df_revenue_current[CUSTOM_WEEK_END_COL], errors='coerce').dt.date

                # Identify the current in-progress week's data
                current_week_data = df_revenue_current[
                    (df_revenue_current[CUSTOM_WEEK_START_COL].notna()) & # Ensure dates are valid
                    (df_revenue_current[CUSTOM_WEEK_END_COL].notna()) &
                    (df_revenue_current[CUSTOM_WEEK_START_COL] <= today) &
                    (df_revenue_current[CUSTOM_WEEK_END_COL] >= today)
                ].copy()


                if not current_week_data.empty:
                    # Ensure week number is valid before assigning
                    week_num_series = pd.to_numeric(current_week_data[WEEK_AS_INT_COL], errors='coerce')
                    if not week_num_series.isna().all():
                        current_week_number = int(week_num_series.dropna().iloc[0])


                # Get data for completed weeks only (end date before today)
                df_full_weeks_current = df_revenue_current.dropna(subset=[CUSTOM_WEEK_END_COL, WEEK_AS_INT_COL])
                df_full_weeks_current = df_full_weeks_current[df_full_weeks_current[CUSTOM_WEEK_END_COL] < today].copy()

                if df_full_weeks_current.empty:
                    # st.info("No *completed* weeks found in the filtered current year data to build the summary.") # Optional info
                    # Initialize empty series if no completed weeks
                    rev_last_4_current = pd.Series(dtype='float64', name=f"Last 4 Weeks Revenue ({filtered_current_year})")
                    rev_last_1_current = pd.Series(dtype='float64', name=f"Last Week Revenue ({filtered_current_year})")
                    last_week_year = filtered_current_year # Keep year label consistent
                else:
                    # Find the latest completed week number and year
                    unique_weeks_current = (df_full_weeks_current.dropna(subset=[WEEK_AS_INT_COL, CUSTOM_YEAR_COL, CUSTOM_WEEK_END_COL])
                                           .groupby([CUSTOM_YEAR_COL, WEEK_AS_INT_COL])
                                           .agg(Week_End=(CUSTOM_WEEK_END_COL, "first"))
                                           .reset_index()
                                           .sort_values("Week_End", ascending=True, na_position='first'))

                    if unique_weeks_current.empty or unique_weeks_current['Week_End'].isna().all():
                         # st.info("Not enough complete week data in the filtered current year to build the revenue summary table.") # Optional info
                         rev_last_4_current = pd.Series(dtype='float64', name=f"Last 4 Weeks Revenue ({filtered_current_year})")
                         rev_last_1_current = pd.Series(dtype='float64', name=f"Last Week Revenue ({filtered_current_year})")
                         last_week_year = filtered_current_year
                    else:
                        last_complete_week_row_current = unique_weeks_current.iloc[-1]
                        last_week_number = int(last_complete_week_row_current[WEEK_AS_INT_COL])
                        last_week_year = int(last_complete_week_row_current[CUSTOM_YEAR_COL]) # Use the actual year of the last week

                        # Get last 4 completed week numbers
                        last_4_weeks_current_df = unique_weeks_current.drop_duplicates(subset=[WEEK_AS_INT_COL]).tail(4)
                        last_4_week_numbers = last_4_weeks_current_df[WEEK_AS_INT_COL].astype(int).tolist()

                        # Calculate revenue for last 4 and last 1 completed weeks
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

                # --- Previous Year Calculations ---
                # Find the previous year relative to the year of the last *completed* week
                if last_week_year in filtered_years_present:
                    current_year_index = filtered_years_present.index(last_week_year)
                    if current_year_index > 0:
                        last_year = filtered_years_present[current_year_index - 1]
                        prev_year_label = str(last_year)
                    else: # Handle case where last_week_year is the earliest year in the filtered list
                        last_year = None
                        prev_year_label = "Prev Year"
                else: # Handle case where last_week_year isn't in the filtered list (shouldn't happen ideally)
                     last_year = None
                     prev_year_label = "Prev Year"


                # Reset previous year series names based on determined label
                rev_last_4_last_year = pd.Series(dtype='float64', name=f"Last 4 Weeks Revenue ({prev_year_label})")
                rev_last_1_last_year = pd.Series(dtype='float64', name=f"Last Week Revenue ({prev_year_label})")


                if last_year is not None:
                    df_revenue_last_year = filtered_df_summary[filtered_df_summary[CUSTOM_YEAR_COL] == last_year].copy()
                    if not df_revenue_last_year.empty:
                        # Ensure types are correct for calculations
                        if WEEK_AS_INT_COL in df_revenue_last_year.columns:
                            df_revenue_last_year[WEEK_AS_INT_COL] = pd.to_numeric(df_revenue_last_year[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
                        if SALES_VALUE_GBP_COL in df_revenue_last_year.columns:
                             df_revenue_last_year[SALES_VALUE_GBP_COL] = pd.to_numeric(df_revenue_last_year[SALES_VALUE_GBP_COL], errors='coerce')

                        # Check required columns exist before dropna
                        cols_to_drop_na_prev = [WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, grouping_key]
                        cols_to_drop_na_prev = [col for col in cols_to_drop_na_prev if col in df_revenue_last_year.columns]
                        df_revenue_last_year.dropna(subset=cols_to_drop_na_prev, inplace=True)


                        # Check if last_week_number and last_4_week_numbers were defined (i.e., completed weeks existed in current year)
                        # And check if required columns still exist
                        if not df_revenue_last_year.empty and last_week_number is not None and last_4_week_numbers and \
                           WEEK_AS_INT_COL in df_revenue_last_year.columns and SALES_VALUE_GBP_COL in df_revenue_last_year.columns:
                             rev_last_1_last_year = (df_revenue_last_year[df_revenue_last_year[WEEK_AS_INT_COL] == last_week_number]
                                                     .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                                     .rename(f"Last Week Revenue ({last_year})").round(0)) # Use actual last_year

                             rev_last_4_last_year = (df_revenue_last_year[df_revenue_last_year[WEEK_AS_INT_COL].isin(last_4_week_numbers)]
                                                     .groupby(grouping_key)[SALES_VALUE_GBP_COL].sum()
                                                     .rename(f"Last 4 Weeks Revenue ({last_year})").round(0)) # Use actual last_year


            # --- Assemble Final Summary Table ---
            # Get all unique keys (listings/products) from the *filtered* data before grouping
            all_keys = pd.Series(sorted(filtered_df_summary[grouping_key].dropna().unique()), name=grouping_key)
            revenue_summary = pd.DataFrame({grouping_key: all_keys}).set_index(grouping_key)

            # Join calculated series - use consistent naming based on actual years/labels
            # Ensure series have names before joining
            current_4wk_col = rev_last_4_current.name if hasattr(rev_last_4_current, 'name') and rev_last_4_current.name else f"Last 4 Weeks Revenue ({last_week_year})"
            current_1wk_col = rev_last_1_current.name if hasattr(rev_last_1_current, 'name') and rev_last_1_current.name else f"Last Week Revenue ({last_week_year})"
            current_week_col = rev_current_week.name if hasattr(rev_current_week, 'name') and rev_current_week.name else f"Current Week So Far ({filtered_current_year})"
            prev_4wk_col = rev_last_4_last_year.name # Already named correctly above
            prev_1wk_col = rev_last_1_last_year.name # Already named correctly above


            revenue_summary = revenue_summary.join(rev_last_4_current.rename(current_4wk_col))\
                                             .join(rev_last_1_current.rename(current_1wk_col))\
                                             .join(rev_current_week.rename(current_week_col))\
                                             .join(rev_last_4_last_year)\
                                             .join(rev_last_1_last_year)

            revenue_summary = revenue_summary.fillna(0) # Fill NaNs resulting from joins

            # Calculate Differences and % Changes
            if current_4wk_col in revenue_summary.columns and prev_4wk_col in revenue_summary.columns:
                revenue_summary["Last 4 Weeks Diff"] = revenue_summary[current_4wk_col] - revenue_summary[prev_4wk_col]
            else: revenue_summary["Last 4 Weeks Diff"] = 0

            if current_1wk_col in revenue_summary.columns and prev_1wk_col in revenue_summary.columns:
                revenue_summary["Last Week Diff"] = revenue_summary[current_1wk_col] - revenue_summary[prev_1wk_col]
            else: revenue_summary["Last Week Diff"] = 0

            # Calculate % Change safely, handling division by zero
            revenue_summary["Last 4 Weeks % Change"] = revenue_summary.apply(
                lambda row: (row["Last 4 Weeks Diff"] / row[prev_4wk_col] * 100)
                if prev_4wk_col in row and row[prev_4wk_col] != 0 else
                (100.0 if "Last 4 Weeks Diff" in row and row["Last 4 Weeks Diff"] > 0 else 0.0), axis=1)

            revenue_summary["Last Week % Change"] = revenue_summary.apply(
                lambda row: (row["Last Week Diff"] / row[prev_1wk_col] * 100)
                if prev_1wk_col in row and row[prev_1wk_col] != 0 else
                (100.0 if "Last Week Diff" in row and row["Last Week Diff"] > 0 else 0.0), axis=1)

            # Calculate Current Week % Change (compared to Last Completed Week)
            if current_week_col in revenue_summary.columns and current_1wk_col in revenue_summary.columns:
                revenue_summary["Current Week % Change"] = revenue_summary.apply(
                    lambda row: ((row[current_week_col] / row[current_1wk_col]) - 1) * 100 # Formula: ((curr/prev)-1)*100
                    if row[current_1wk_col] != 0 else
                    (100.0 if row[current_week_col] > 0 else 0.0), axis=1) # Handle division by zero
            else:
                revenue_summary["Current Week % Change"] = 0.0 # Default if columns missing


            revenue_summary = revenue_summary.reset_index()

            # Define desired column order dynamically based on calculated columns
            desired_order = [grouping_key]
            # Add columns if they exist in the dataframe
            if current_4wk_col in revenue_summary.columns: desired_order.append(current_4wk_col)
            if prev_4wk_col in revenue_summary.columns: desired_order.append(prev_4wk_col)
            if "Last 4 Weeks Diff" in revenue_summary.columns: desired_order.append("Last 4 Weeks Diff")
            if "Last 4 Weeks % Change" in revenue_summary.columns: desired_order.append("Last 4 Weeks % Change")
            if current_1wk_col in revenue_summary.columns: desired_order.append(current_1wk_col)
            if prev_1wk_col in revenue_summary.columns: desired_order.append(prev_1wk_col)
            if "Last Week Diff" in revenue_summary.columns: desired_order.append("Last Week Diff")
            if "Last Week % Change" in revenue_summary.columns: desired_order.append("Last Week % Change")
            if current_week_col in revenue_summary.columns: desired_order.append(current_week_col)
            if "Current Week % Change" in revenue_summary.columns: desired_order.append("Current Week % Change")

            # Ensure only existing columns are selected and reorder
            revenue_summary = revenue_summary[[col for col in desired_order if col in revenue_summary.columns]]

            # --- Calculate Total Summary Row ---
            if not revenue_summary.empty:
                summary_row = {}
                for col in revenue_summary.columns:
                    if col != grouping_key and pd.api.types.is_numeric_dtype(revenue_summary[col]):
                        # Sum numeric columns, excluding percentage columns which need recalculation
                        if '%' not in col:
                            summary_row[col] = revenue_summary[col].sum()
                        else:
                            summary_row[col] = 0 # Placeholder for percentages
                    else:
                        summary_row[col] = '' # Blank for non-numeric like the grouping key

                summary_row[grouping_key] = "Total" # Set grouping key value for total row

                # Get summed totals needed for percentage recalculation
                total_last4_last_year = summary_row.get(prev_4wk_col, 0)
                total_last_week_last_year = summary_row.get(prev_1wk_col, 0)
                total_last_week_current = summary_row.get(current_1wk_col, 0)
                total_current_week_so_far = summary_row.get(current_week_col, 0)
                total_diff_4wk = summary_row.get("Last 4 Weeks Diff", 0)
                total_diff_1wk = summary_row.get("Last Week Diff", 0)

                # Recalculate % Change for the Total row using summed values
                if "Last 4 Weeks % Change" in summary_row:
                    summary_row["Last 4 Weeks % Change"] = (total_diff_4wk / total_last4_last_year * 100) if total_last4_last_year != 0 else (100.0 if total_diff_4wk > 0 else 0.0)
                if "Last Week % Change" in summary_row:
                    summary_row["Last Week % Change"] = (total_diff_1wk / total_last_week_last_year * 100) if total_last_week_last_year != 0 else (100.0 if total_diff_1wk > 0 else 0.0)
                if "Current Week % Change" in summary_row:
                     summary_row["Current Week % Change"] = ((total_current_week_so_far / total_last_week_current) - 1) * 100 if total_last_week_current != 0 else (100.0 if total_current_week_so_far > 0 else 0.0)


                # Create Total DataFrame using the corrected summary_row
                total_df = pd.DataFrame([summary_row])[revenue_summary.columns] # Use same column order as main table

                # --- Styling and Display ---
                def color_diff(val):
                    # Colors based on value (positive green, negative red)
                    try:
                        val = float(val)
                        if val < -0.001: return 'color: red'
                        elif val > 0.001: return 'color: green'
                        else: return '' # Neutral for zero or near-zero
                    except (ValueError, TypeError):
                        return '' # No style for non-numeric

                formats = {}
                # Define formats dynamically based on existing columns
                if current_4wk_col in revenue_summary.columns: formats[current_4wk_col] = "£{:,.0f}"
                if prev_4wk_col in revenue_summary.columns: formats[prev_4wk_col] = "£{:,.0f}"
                if current_1wk_col in revenue_summary.columns: formats[current_1wk_col] = "£{:,.0f}"
                if prev_1wk_col in revenue_summary.columns: formats[prev_1wk_col] = "£{:,.0f}"
                if current_week_col in revenue_summary.columns: formats[current_week_col] = "£{:,.0f}"
                if "Last 4 Weeks Diff" in revenue_summary.columns: formats["Last 4 Weeks Diff"] = "{:,.0f}"
                if "Last Week Diff" in revenue_summary.columns: formats["Last Week Diff"] = "{:,.0f}"
                if "Last 4 Weeks % Change" in revenue_summary.columns: formats["Last 4 Weeks % Change"] = "{:.1f}%"
                if "Last Week % Change" in revenue_summary.columns: formats["Last Week % Change"] = "{:.1f}%"
                if "Current Week % Change" in revenue_summary.columns: formats["Current Week % Change"] = "{:.1f}%"

                # Define columns to apply color formatting
                color_cols = [col for col in ["Last 4 Weeks Diff", "Last Week Diff", "Last 4 Weeks % Change", "Last Week % Change", "Current Week % Change"] if col in revenue_summary.columns]

                # --- Display Total Summary ---
                st.markdown("##### Total Summary")
                styled_total = total_df.style.format(formats, na_rep='-') \
                                             .apply(lambda x: x.map(color_diff), subset=color_cols) \
                                             .set_properties(**{'font-weight': 'bold'}) \
                                             .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                {'selector': 'td', 'props': [('text-align', 'right')]}]) \
                                             .hide(axis="index")
                st.dataframe(styled_total, use_container_width=True)


                # --- Display Detailed Summary ---
                st.markdown("##### Detailed Summary")
                # Convert relevant value columns to integer AFTER calculations, before styling main table
                value_cols_to_int = [col for col in revenue_summary.columns if ('Revenue' in col or 'Diff' in col) and '%' not in col]
                if value_cols_to_int:
                    for col in value_cols_to_int:
                         # Ensure column exists and handle potential non-finite numbers before converting
                         if col in revenue_summary.columns:
                            revenue_summary[col] = pd.to_numeric(revenue_summary[col], errors='coerce').fillna(0).astype(int)

                styled_main = revenue_summary.style.format(formats, na_rep='-') \
                                                  .apply(lambda x: x.map(color_diff), subset=color_cols) \
                                                  .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                     {'selector': 'td', 'props': [('text-align', 'right')]}]) \
                                                  .hide(axis="index")
                st.dataframe(styled_main, use_container_width=True)

                # --- REMOVED: Display Week Information Caption ---
                # The following lines have been removed as requested:
                # if has_start_date and has_end_date and 'current_week_data' in locals() and not current_week_data.empty and current_week_number is not None:
                #      current_week_start_obj = ...
                #      current_week_end_obj = ...
                #      if current_week_start_obj and current_week_end_obj:
                #         st.caption(f"Current week ({current_week_number})...")
                #      else:
                #         st.caption(f"Current week ({current_week_number})...")
                # elif last_week_number is not None:
                #      st.caption(f"Summary based on completed weeks up to Week {last_week_number}, {last_week_year}.")
                # else:
                #      st.caption("Weekly date information unavailable.")
                # --- END REMOVED ---


            else: # Handle case where revenue_summary is empty after all calculations
                 st.info("No data available to display in the summary table after applying all filters and calculations.")


