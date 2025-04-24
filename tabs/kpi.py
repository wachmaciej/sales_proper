# tabs/kpi.py
import streamlit as st
import pandas as pd
import datetime
from utils import format_currency, format_currency_int, get_custom_week_date_range
from config import (
    CUSTOM_YEAR_COL, WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW,
    CUSTOM_WEEK_START_COL, CUSTOM_WEEK_END_COL
)

def display_tab(df, available_years, current_year):
    """Displays the KPI tab."""
    st.markdown("### Key Performance Indicators")

    with st.expander("KPI Filters", expanded=True):
        today = datetime.date.today()
        selected_week = None
        available_weeks_in_current_year = []

        if CUSTOM_YEAR_COL not in df.columns or WEEK_AS_INT_COL not in df.columns:
            st.error(f"Missing '{CUSTOM_YEAR_COL}' or '{WEEK_AS_INT_COL}' for KPI calculations.")
        else:
            # Ensure Week column is numeric before filtering/sorting
            df[WEEK_AS_INT_COL] = pd.to_numeric(df[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
            current_year_weeks = df[df[CUSTOM_YEAR_COL] == current_year][WEEK_AS_INT_COL].dropna()

            if not current_year_weeks.empty:
                available_weeks_in_current_year = sorted(current_year_weeks.unique())
            else:
                available_weeks_in_current_year = []

            # Determine the default week (last *completed* week)
            full_weeks = []
            last_available_week = None
            if available_weeks_in_current_year:
                 last_available_week = available_weeks_in_current_year[-1]
                 for wk in available_weeks_in_current_year:
                    if pd.notna(wk):
                        try:
                            wk_int = int(wk)
                            week_start_dt, week_end_dt = get_custom_week_date_range(current_year, wk_int)
                            # Check if the week has completed
                            if week_end_dt and week_end_dt < today:
                                full_weeks.append(wk_int)
                        except (ValueError, TypeError):
                            continue # Skip if week number is invalid

            # Set default: last full week, or last available week if no full weeks yet, or 1 if no weeks
            default_week = full_weeks[-1] if full_weeks else (last_available_week if last_available_week is not None else 1)


            if available_weeks_in_current_year:
                selected_week = st.selectbox(
                    "Select Week for KPI Calculation",
                    options=available_weeks_in_current_year,
                    index=available_weeks_in_current_year.index(default_week) if default_week in available_weeks_in_current_year else 0,
                    key="kpi_week",
                    help="Select the week to calculate KPIs for. Defaults to the last completed week if available."
                )

                # Display selected week's date range
                if pd.notna(selected_week):
                     try:
                         selected_week_int_info = int(selected_week)
                         week_start_custom, week_end_custom = get_custom_week_date_range(current_year, selected_week_int_info)
                         if week_start_custom and week_end_custom:
                             st.info(f"Selected Week {selected_week_int_info}: {week_start_custom.strftime('%d %b')} - {week_end_custom.strftime('%d %b, %Y')}")
                         else:
                             st.warning(f"Could not determine date range for Week {selected_week_int_info}, Year {current_year}.")
                     except (ValueError, TypeError):
                         st.warning(f"Invalid week selected: {selected_week}")
                         selected_week = None # Reset selection if invalid
                else:
                    selected_week = None # Handles potential NaN/None selection if options are weird
            else:
                 st.warning(f"No weeks found for the current year ({current_year}) to calculate KPIs.")
                 selected_week = None

    # --- KPI Calculation and Display ---
    if selected_week is not None and pd.notna(selected_week):
        try:
            selected_week_int = int(selected_week)

            # Filter data for the specific week across *all* years available in the dataframe for comparison
            kpi_data_all_years = df[df[WEEK_AS_INT_COL] == selected_week_int].copy()

            if kpi_data_all_years.empty:
                st.info(f"No sales data found for Week {selected_week_int} across any year to calculate KPIs.")
            else:
                # Ensure required columns are numeric
                kpi_data_all_years[SALES_VALUE_GBP_COL] = pd.to_numeric(kpi_data_all_years[SALES_VALUE_GBP_COL], errors='coerce')

                # Calculate Revenue Summary per Year for the selected week
                revenue_summary = kpi_data_all_years.groupby(CUSTOM_YEAR_COL)[SALES_VALUE_GBP_COL].sum()

                # Calculate Units Summary per Year (optional)
                units_summary = None
                if ORDER_QTY_COL_RAW in kpi_data_all_years.columns:
                    kpi_data_all_years[ORDER_QTY_COL_RAW] = pd.to_numeric(kpi_data_all_years[ORDER_QTY_COL_RAW], errors='coerce')
                    units_summary = kpi_data_all_years.groupby(CUSTOM_YEAR_COL)[ORDER_QTY_COL_RAW].sum().fillna(0)
                else:
                    st.info(f"Column '{ORDER_QTY_COL_RAW}' not found, units and AOV KPIs will not be shown.")

                # Display metrics for all available years side-by-side
                all_custom_years_in_df = sorted(pd.to_numeric(df[CUSTOM_YEAR_COL], errors='coerce').dropna().unique().astype(int))
                kpi_cols = st.columns(len(all_custom_years_in_df))

                # Store previous year's values for delta calculation
                prev_rev = 0
                prev_units = 0

                for idx, year in enumerate(all_custom_years_in_df):
                    with kpi_cols[idx]:
                        # Get current year's values for the selected week
                        revenue = revenue_summary.get(year, 0)
                        total_units = units_summary.get(year, 0) if units_summary is not None else 0

                        # --- Revenue Metric ---
                        numeric_delta_rev = None
                        delta_rev_str = None
                        delta_rev_color = "off"

                        if idx > 0: # Can only calculate delta if not the first year
                            # prev_rev is from the *previous iteration's* year
                            if prev_rev != 0 or revenue != 0: # Avoid delta calculation if both are zero
                                numeric_delta_rev = revenue - prev_rev
                                delta_rev_str = f"{int(round(numeric_delta_rev)):,}" # Format as integer difference
                                delta_rev_color = "normal" # Enable color coding

                        st.metric(
                            label=f"Revenue {year} (Wk {selected_week_int})",
                            value=format_currency_int(revenue),
                            delta=delta_rev_str,
                            delta_color=delta_rev_color
                        )

                        # --- Units Metric ---
                        if units_summary is not None:
                            delta_units_str = None
                            delta_units_color = "off"

                            if idx > 0:
                                if prev_units != 0:
                                    delta_units_percent = ((total_units - prev_units) / prev_units) * 100
                                    delta_units_str = f"{delta_units_percent:.1f}%"
                                    delta_units_color = "normal"
                                elif total_units != 0: # Current units exist, previous was 0
                                    delta_units_str = "+Units" # Indicate increase from zero
                                    delta_units_color = "normal"
                                # If total_units is 0 and prev_units was 0, delta is None

                            st.metric(
                                label=f"Units Sold {year} (Wk {selected_week_int})",
                                value=f"{int(total_units):,}" if pd.notna(total_units) else "N/A",
                                delta=delta_units_str,
                                delta_color=delta_units_color
                            )

                            # --- AOV Metric ---
                            aov = revenue / total_units if total_units != 0 else 0
                            delta_aov_str = None
                            delta_aov_color = "off"

                            if idx > 0:
                                prev_aov = prev_rev / prev_units if prev_units != 0 else 0
                                if prev_aov != 0:
                                    delta_aov_percent = ((aov - prev_aov) / prev_aov) * 100
                                    delta_aov_str = f"{delta_aov_percent:.1f}%"
                                    delta_aov_color = "normal"
                                elif aov != 0: # Current AOV exists, previous was 0
                                    delta_aov_str = "+AOV"
                                    delta_aov_color = "normal"

                            st.metric(
                                label=f"AOV {year} (Wk {selected_week_int})",
                                value=format_currency(aov), # Use format_currency for potential decimals
                                delta=delta_aov_str,
                                delta_color=delta_aov_color
                            )

                        # Update previous values for the next iteration's delta calculation
                        prev_rev = revenue
                        prev_units = total_units

        except (ValueError, TypeError):
            st.error(f"Invalid week number encountered: {selected_week}. Cannot calculate KPIs.")
    elif selected_week is None and available_weeks_in_current_year: # Check if selection is None but options existed
        st.info("Select a valid week from the filters above to view KPIs.")
    # No message needed if available_weeks_in_current_year is empty, warning already shown