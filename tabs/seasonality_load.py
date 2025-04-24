# tabs/seasonality_load.py
import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import math
import numpy as np
from config import (
    SEASON_COL, DATE_COL, SKU_COL, ORDER_QTY_COL_RAW, CUSTOM_YEAR_COL,
    LISTING_COL, SALES_CHANNEL_COL, SALES_VALUE_TRANS_CURRENCY_COL,
    ORIGINAL_CURRENCY_COL
)
from utils import format_dynamic_currency, format_currency

# --- Helper Functions ---
def get_week_start_end(ref_date):
    """Returns the start (Monday) and end (Sunday) of the week for a given date."""
    if ref_date is None: return None, None
    start_of_week = ref_date - timedelta(days=ref_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

def get_last_n_weeks_range(ref_date, n):
    """Calculates the date range for the N weeks ending *before* the week of the ref_date."""
    if ref_date is None: return None, None
    start_ref_week, _ = get_week_start_end(ref_date);
    if start_ref_week is None: return None, None
    end_of_period = start_ref_week - timedelta(days=1)
    start_of_period_n_weeks_prior = end_of_period - timedelta(weeks=n-1)
    start_of_period = start_of_period_n_weeks_prior - timedelta(days=start_of_period_n_weeks_prior.weekday())
    return start_of_period, end_of_period

def calc_pct_diff(curr, prev):
    """Calculates percentage difference, handling zero division."""
    if not isinstance(curr, (int,float, np.number)) or not isinstance(prev, (int,float, np.number)): return np.nan
    if pd.isna(curr) or pd.isna(prev): return np.nan
    if prev == 0: return 100.0 if curr != 0 else 0.0
    try: return ((float(curr)-float(prev))/float(prev))*100.0
    except (ValueError, TypeError): return np.nan

def calculate_sales_rates(df_season_filtered, max_data_date):
    """Calculates LW/L4W sales rates for TY vs LY."""
    if df_season_filtered is None or df_season_filtered.empty or max_data_date is None: return pd.DataFrame()
    required_rate_cols = {DATE_COL, SKU_COL, LISTING_COL, ORDER_QTY_COL_RAW};
    if not required_rate_cols.issubset(df_season_filtered.columns):
        st.warning(f"Sales rate calc missing: {required_rate_cols.difference(df_season_filtered.columns)}")
        return pd.DataFrame()

    df_calc = df_season_filtered.copy()
    try:
        df_calc[DATE_COL] = pd.to_datetime(df_calc[DATE_COL]).dt.date
        df_calc[ORDER_QTY_COL_RAW] = pd.to_numeric(df_calc[ORDER_QTY_COL_RAW], errors='coerce').fillna(0);
        df_calc.dropna(subset=[DATE_COL, ORDER_QTY_COL_RAW, LISTING_COL, SKU_COL], inplace=True)
    except Exception as e:
        st.error(f"Error preparing data for sales rate calculation: {e}")
        return pd.DataFrame()
    if df_calc.empty: return pd.DataFrame()

    try:
        lw_ty_start, lw_ty_end = get_week_start_end(max_data_date)
        l4w_ty_start, l4w_ty_end = get_last_n_weeks_range(max_data_date, 4)
        if lw_ty_start is None or l4w_ty_start is None: raise ValueError("Could not calculate TY date ranges.")
        lw_ly_start_raw = lw_ty_start - timedelta(days=364); lw_ly_start = lw_ly_start_raw - timedelta(days=lw_ly_start_raw.weekday()); lw_ly_end = lw_ly_start + timedelta(days=6)
        l4w_ly_start_raw = l4w_ty_start - timedelta(days=364); l4w_ly_start = l4w_ly_start_raw - timedelta(days=l4w_ly_start_raw.weekday()); l4w_ly_end = l4w_ly_start + timedelta(weeks=4) - timedelta(days=1)
    except Exception as e:
        st.error(f"Error calculating date ranges for sales rates: {e}")
        return pd.DataFrame()

    def get_period_sales(start_date, end_date, num_weeks):
        if start_date is None or end_date is None or num_weeks <= 0: return pd.DataFrame({LISTING_COL: [], SKU_COL: [], f'Rate ({num_weeks}wk Avg)': []})
        mask = (df_calc[DATE_COL] >= start_date) & (df_calc[DATE_COL] <= end_date);
        period_data = df_calc.loc[mask]
        if period_data.empty: return pd.DataFrame({LISTING_COL: [], SKU_COL: [], f'Rate ({num_weeks}wk Avg)': []})
        agg = period_data.groupby([LISTING_COL, SKU_COL], observed=False)[ORDER_QTY_COL_RAW].sum().reset_index();
        agg[f'Rate ({num_weeks}wk Avg)'] = agg[ORDER_QTY_COL_RAW] / float(num_weeks)
        return agg[[LISTING_COL, SKU_COL, f'Rate ({num_weeks}wk Avg)']]

    lw_ty_rate_df = get_period_sales(lw_ty_start, lw_ty_end, 1)
    l4w_ty_rate_df = get_period_sales(l4w_ty_start, l4w_ty_end, 4)
    lw_ly_rate_df = get_period_sales(lw_ly_start, lw_ly_end, 1)
    l4w_ly_rate_df = get_period_sales(l4w_ly_start, l4w_ly_end, 4)

    all_skus = df_calc[[LISTING_COL, SKU_COL]].drop_duplicates().reset_index(drop=True);
    if all_skus.empty: return pd.DataFrame()

    merged = all_skus;
    rate_dfs = {
        'Rate (LW TY)': (lw_ty_rate_df, 'Rate (1wk Avg)'),
        'Rate (L4W TY)': (l4w_ty_rate_df, 'Rate (4wk Avg)'),
        'Rate (LW LY)': (lw_ly_rate_df, 'Rate (1wk Avg)'),
        'Rate (L4W LY)': (l4w_ly_rate_df, 'Rate (4wk Avg)')
    }
    for target_col, (df_rate, src_col) in rate_dfs.items():
        if not df_rate.empty and src_col in df_rate.columns:
            merged = pd.merge(merged, df_rate[[LISTING_COL, SKU_COL, src_col]].rename(columns={src_col:target_col}), on=[LISTING_COL, SKU_COL], how='left')
        else: merged[target_col] = 0.0
    merged[list(rate_dfs.keys())] = merged[list(rate_dfs.keys())].fillna(0)

    merged['% Diff (LW TY vs LY)'] = merged.apply(lambda r: calc_pct_diff(r['Rate (LW TY)'], r['Rate (LW LY)']), axis=1)
    merged['% Diff (L4W TY vs LY)'] = merged.apply(lambda r: calc_pct_diff(r['Rate (L4W TY)'], r['Rate (L4W LY)']), axis=1)
    merged['% Diff (LW vs L4W TY)'] = merged.apply(lambda r: calc_pct_diff(r['Rate (LW TY)'], r['Rate (L4W TY)']), axis=1)

    merged.rename(columns={SKU_COL:"Product SKU", LISTING_COL:"Listing"}, inplace=True);
    merged.sort_values(by=["Listing", "Product SKU"], inplace=True)
    final_cols = ["Listing", "Product SKU", 'Rate (LW TY)', 'Rate (LW LY)', '% Diff (LW TY vs LY)', 'Rate (L4W TY)', 'Rate (L4W LY)', '% Diff (L4W TY vs LY)', '% Diff (LW vs L4W TY)'];
    for col in final_cols:
        if col not in merged.columns: merged[col] = 0.0 if 'Rate' in col else np.nan
    merged = merged.reindex(columns=final_cols)
    return merged

def get_currency_info(df_filtered, selected_channels):
    """Determines currency symbol based on filtered data."""
    currency_symbol = ""; currency_label = "Avg Price"
    if df_filtered is None or df_filtered.empty: return currency_symbol, currency_label
    if ORIGINAL_CURRENCY_COL not in df_filtered.columns: return currency_symbol, currency_label # No warning needed here

    target_df = df_filtered.copy()
    if selected_channels and SALES_CHANNEL_COL in target_df.columns:
        target_df = target_df[target_df[SALES_CHANNEL_COL].isin(selected_channels)]
    if target_df.empty: return currency_symbol, currency_label

    unique_currencies = target_df[ORIGINAL_CURRENCY_COL].dropna().unique()
    if len(unique_currencies) == 1:
        currency_code = unique_currencies[0]
        if pd.isna(currency_code) or str(currency_code).strip() == '': currency_symbol = ""; currency_label = "Avg Price (No Currency)"
        elif currency_code == "GBP": currency_symbol = "£"
        elif currency_code == "USD": currency_symbol = "$"
        elif currency_code == "EUR": currency_symbol = "€"
        else: currency_symbol = f"{str(currency_code).strip()} "
        if currency_symbol.strip(): currency_label = f"Avg Price ({currency_symbol.strip()})"
        else: currency_label = "Avg Price"
    elif len(unique_currencies) > 1: st.caption("Note: Mixed currencies found."); currency_symbol = ""; currency_label = "Avg Price (Mixed)"
    else: currency_symbol = ""; currency_label = "Avg Price (No Currency)"
    return currency_symbol, currency_label

def color_diff(cell_value):
    """Applies color to cell based on positive/negative value."""
    color = 'black';
    if pd.notna(cell_value) and np.isfinite(cell_value) and isinstance(cell_value, (int, float, np.number)):
        if cell_value > 0.001: color = '#198754' # Use small threshold for float comparison
        elif cell_value < -0.001: color = '#dc3545'
    return f'color: {color}'

# --- Main Display Function ---
def display_tab(df, available_years):
    st.markdown("### Seasonality Load Planning")
    st.markdown("Analyze units sold and average selling price for a specific season, date range, and channel(s) across previous years, and generate a simple unit forecast.")

    # Check for required columns
    required_cols = {
        SEASON_COL, DATE_COL, SKU_COL, ORDER_QTY_COL_RAW, CUSTOM_YEAR_COL,
        LISTING_COL, SALES_CHANNEL_COL, SALES_VALUE_TRANS_CURRENCY_COL, ORIGINAL_CURRENCY_COL
    }
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        st.error(f"Required columns missing from the input data: {missing_cols}.")
        return

    # --- Initial Data Processing and Type Conversion ---
    df_processed = df.copy()
    try:
        df_processed[DATE_COL] = pd.to_datetime(df_processed[DATE_COL], errors='coerce');
        num_cols = [ORDER_QTY_COL_RAW, SALES_VALUE_TRANS_CURRENCY_COL, CUSTOM_YEAR_COL];
        for col in num_cols:
            if col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if CUSTOM_YEAR_COL in df_processed.columns: df_processed[CUSTOM_YEAR_COL] = df_processed[CUSTOM_YEAR_COL].astype('Int64')
        str_cols = [SEASON_COL, SKU_COL, LISTING_COL, SALES_CHANNEL_COL, ORIGINAL_CURRENCY_COL];
        for col in str_cols:
            if col in df_processed.columns: df_processed[col] = df_processed[col].fillna('').astype(str).str.strip()
        df_processed.dropna(subset=[DATE_COL, ORDER_QTY_COL_RAW, CUSTOM_YEAR_COL, SKU_COL, LISTING_COL, SEASON_COL], inplace=True)
        if df_processed.empty: st.warning("No data available after initial processing and NA removal."); return
    except Exception as e: st.error(f"Error preparing data types: {e}"); return

    # --- Filters ---
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            available_channels = sorted(df_processed[SALES_CHANNEL_COL].astype(str).dropna().unique())
            selected_channels = st.multiselect( "Sales Channel(s)", options=available_channels, default=[], key="seasonality_channels", help="Select channels to filter data. Leave empty for all." )
        with col2:
            available_seasons = sorted(df_processed[SEASON_COL].astype(str).dropna().unique())
            season_options = [s for s in available_seasons if s and s.strip() and s.upper() != "ALL"]
            selected_season = None
            if not season_options: st.warning("No specific seasons found in the data.")
            else: selected_season = st.selectbox( "Select Season", options=season_options, key="seasonality_season", index=0, help="Select the season to analyze." )
        with col3:
            st.markdown("###### Historical Month-Day Range"); c3a, c3b = st.columns(2); month_names = {m: datetime.date(2000, m, 1).strftime('%b') for m in range(1, 13)}
            with c3a: st.caption("Start"); start_month = st.selectbox("Month ", options=list(month_names.keys()), key="start_month", index=0, format_func=month_names.get); start_day = st.number_input("Day ", min_value=1, max_value=31, value=1, step=1, key="start_day", label_visibility="collapsed")
            with c3b: st.caption("End"); end_month = st.selectbox(" Month", options=list(month_names.keys()), key="end_month", index=11, format_func=month_names.get); end_day = st.number_input(" Day", min_value=1, max_value=31, value=31, step=1, key="end_day", label_visibility="collapsed")
            if not (1 <= start_day <= 31 and 1 <= end_day <= 31): st.warning("Please select a valid day (1-31).")
            start_month_day = (start_month, start_day); end_month_day = (end_month, end_day)

    # --- Data Filtering based on selections ---
    all_listing_agg_summaries = []; filtered_df_final = pd.DataFrame(); period_str = "the selected period"
    if selected_season:
        filtered_df = df_processed[df_processed[SEASON_COL] == selected_season].copy()
        if selected_channels: filtered_df = filtered_df[filtered_df[SALES_CHANNEL_COL].isin(selected_channels)]
        currency_symbol, avg_price_label = get_currency_info(filtered_df, selected_channels)

        if not filtered_df.empty:
            try:
                if pd.api.types.is_datetime64_any_dtype(filtered_df[DATE_COL]):
                    filtered_df['month_day'] = filtered_df[DATE_COL].apply(lambda d: (d.month, d.day) if pd.notna(d) else None); filtered_df.dropna(subset=['month_day'], inplace=True)
                    if start_month_day <= end_month_day: mask = (filtered_df['month_day'] >= start_month_day) & (filtered_df['month_day'] <= end_month_day)
                    else: mask = (filtered_df['month_day'] >= start_month_day) | (filtered_df['month_day'] <= end_month_day)
                    filtered_df_final = filtered_df[mask].copy()
                    if 'month_day' in filtered_df_final.columns: filtered_df_final.drop(columns=['month_day'], inplace=True)
                else: st.warning(f"Cannot filter by month-day: '{DATE_COL}' not date format."); filtered_df_final = filtered_df
            except Exception as e: st.error(f"Error during month-day filtering: {e}"); filtered_df_final = pd.DataFrame()
        start_m_name = month_names.get(start_month_day[0], '?'); end_m_name = month_names.get(end_month_day[0], '?'); period_str = f"{start_m_name} {start_month_day[1]} - {end_m_name} {end_month_day[1]}"

        # --- Display Historical Data Tables ---
        if filtered_df_final.empty :
            st.info(f"No sales data found for Season '{selected_season}'{', selected Channels' if selected_channels else ''}, within the period {period_str}.")
        else:
            available_listings = sorted(filtered_df_final[LISTING_COL].unique())
            if available_listings:
                st.markdown("---")
                st.markdown(f"###### Units Sold & {avg_price_label} per SKU / Year")
                st.caption(f"Data for Season '{selected_season}' between {period_str} (all years)")

                for listing in available_listings:
                    st.subheader(f"Listing: {listing}")
                    df_listing = filtered_df_final[filtered_df_final[LISTING_COL] == listing].copy()

                    if df_listing.empty or df_listing[CUSTOM_YEAR_COL].isna().all():
                        st.caption("No data available for this listing in the selected period/year range.")
                        if len(available_listings) > 1: st.markdown("---")
                        continue;

                    # Aggregate per SKU/Year for this listing
                    agg_summary_listing = df_listing.groupby([SKU_COL, CUSTOM_YEAR_COL]).agg(
                        Total_Units=(ORDER_QTY_COL_RAW, 'sum'),
                        Total_Sales_Trans=(SALES_VALUE_TRANS_CURRENCY_COL, 'sum')
                    ).reset_index()
                    agg_summary_listing['Avg_Price_Trans'] = (agg_summary_listing['Total_Sales_Trans'] / agg_summary_listing['Total_Units'].astype(float).replace(0, pd.NA)).fillna(0)

                    if agg_summary_listing.empty:
                        st.caption("Could not summarize data for this listing.")
                        if len(available_listings) > 1: st.markdown("---")
                        continue;

                    # Store summary for overall forecast
                    summary_for_forecast = agg_summary_listing.copy();
                    summary_for_forecast[LISTING_COL] = listing;
                    all_listing_agg_summaries.append(summary_for_forecast)

                    # --- Display Pivot Table for the Listing ---
                    try:
                        # Pivot the aggregated data for display
                        combined_pivot = pd.pivot_table(
                            agg_summary_listing,
                            index=SKU_COL,
                            columns=CUSTOM_YEAR_COL,
                            values=['Total_Units', 'Avg_Price_Trans'],
                            aggfunc='sum' # Already aggregated, sum just picks the value
                        ).fillna(0);

                        if combined_pivot.empty: raise ValueError("Pivot table generation resulted in empty DataFrame.")

                        # --- Corrected Header Flattening Logic ---
                        original_multi_columns = combined_pivot.columns.copy()
                        year_cols = sorted([yr for yr in original_multi_columns.get_level_values(1).unique() if pd.notna(yr)])
                        if not year_cols: raise ValueError("No valid year columns found in pivot table.")

                        new_column_names_list = []
                        final_formatters = {}
                        price_formatter = lambda x: format_dynamic_currency(x, currency_symbol)

                        for yr in year_cols:
                            yr_str = str(int(yr))
                            unit_col_orig = ('Total_Units', yr)
                            price_col_orig = ('Avg_Price_Trans', yr)
                            unit_col_new = f"Units Sold {yr_str}"
                            # Use the dynamic avg_price_label determined earlier
                            price_col_new = f"{avg_price_label} {yr_str}"

                            if unit_col_orig in original_multi_columns:
                                new_column_names_list.append(unit_col_new)
                                final_formatters[unit_col_new] = "{:,.0f}"
                            if price_col_orig in original_multi_columns:
                                new_column_names_list.append(price_col_new)
                                final_formatters[price_col_new] = price_formatter

                        if not new_column_names_list: raise ValueError("No display columns generated after flattening.")

                        # Directly assign the flattened list to the columns
                        combined_pivot.columns = new_column_names_list

                        # Ensure data types are appropriate (Units should be int)
                        for col_name in combined_pivot.columns:
                            if col_name.startswith("Units Sold"):
                                combined_pivot[col_name] = pd.to_numeric(combined_pivot[col_name], errors='coerce').fillna(0).astype(int)
                        # --- End Corrected Header Flattening Logic ---


                        # --- Calculate and Display Totals Row ---
                        listing_yearly_agg = df_listing.groupby(CUSTOM_YEAR_COL).agg(
                            Listing_Total_Units=(ORDER_QTY_COL_RAW, 'sum'),
                            Listing_Total_Sales_Trans=(SALES_VALUE_TRANS_CURRENCY_COL, 'sum'));
                        listing_yearly_agg['Listing_Avg_Price_Trans'] = (listing_yearly_agg['Listing_Total_Sales_Trans'] / listing_yearly_agg['Listing_Total_Units'].astype(float).replace(0, pd.NA)).fillna(0)

                        total_row_data = {};
                        for yr_float in year_cols:
                            yr_str = str(int(yr_float))
                            unit_col_new = f"Units Sold {yr_str}"
                            price_col_new = f"{avg_price_label} {yr_str}"

                            total_row_data[unit_col_new] = combined_pivot[unit_col_new].sum() if unit_col_new in combined_pivot.columns else 0
                            total_row_data[price_col_new] = listing_yearly_agg.loc[yr_float, 'Listing_Avg_Price_Trans'] if yr_float in listing_yearly_agg.index else 0

                        total_df = pd.DataFrame(total_row_data, index=[f"TOTAL ({listing})"]);
                        if not total_df.empty:
                            total_df = total_df.reindex(columns=new_column_names_list, fill_value=0)
                            st.dataframe(total_df.style.format(final_formatters, na_rep='-').set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                        else:
                            st.caption("Could not calculate listing totals.")

                        # --- Display SKU Breakdown Table ---
                        if not combined_pivot.empty:
                            combined_pivot.sort_index(inplace=True)
                            st.dataframe(combined_pivot.style.format(final_formatters, na_rep='-'), use_container_width=True)
                        else:
                            st.caption("Could not generate SKU breakdown.")

                    except Exception as e:
                        st.error(f"Could not display historical table for listing '{listing}': {e}")
                        st.dataframe(agg_summary_listing[[SKU_COL, CUSTOM_YEAR_COL, 'Total_Units', 'Avg_Price_Trans']]) # Fallback

                    if len(available_listings) > 1: st.markdown("---")

            else: # No available listings found
                st.markdown("---")
                st.caption(f"No specific listings found within filtered data for Season '{selected_season}' ({period_str}).")

    # --- Calculate and Display Sales Rates ---
    st.markdown("---")
    st.markdown(f"### Recent Sales Rate (Units / Week)")
    if selected_season:
        df_rate_filtered = df_processed[df_processed[SEASON_COL] == selected_season].copy()
        if selected_channels: df_rate_filtered = df_rate_filtered[df_rate_filtered[SALES_CHANNEL_COL].isin(selected_channels)]
        if not df_rate_filtered.empty:
            max_data_date_rate = df_rate_filtered[DATE_COL].max()
            if pd.notna(max_data_date_rate):
                max_data_date_rate_dt = max_data_date_rate.date()
                st.caption(f"Rates based on data up to {max_data_date_rate_dt.strftime('%Y-%m-%d')}. LW=Current Week, L4W=Previous 4 Weeks Avg. TY/LY.")
                sales_rate_df = calculate_sales_rates(df_rate_filtered, max_data_date_rate_dt)
                if not sales_rate_df.empty:
                    total_lw_ty = sales_rate_df['Rate (LW TY)'].sum(); total_lw_ly = sales_rate_df['Rate (LW LY)'].sum(); total_l4w_ty = sales_rate_df['Rate (L4W TY)'].sum(); total_l4w_ly = sales_rate_df['Rate (L4W LY)'].sum()
                    overall_pct_diff_lw = calc_pct_diff(total_lw_ty, total_lw_ly); overall_pct_diff_l4w = calc_pct_diff(total_l4w_ty, total_l4w_ly)
                    st.markdown("##### Overall % Change (TY vs LY)"); col_s1, col_s2 = st.columns(2)
                    with col_s1: st.metric(label="Total LW TY vs LY %", value=f"{overall_pct_diff_lw:+.1f}%" if pd.notna(overall_pct_diff_lw) else "N/A")
                    with col_s2: st.metric(label="Total L4W TY vs LY %", value=f"{overall_pct_diff_l4w:+.1f}%" if pd.notna(overall_pct_diff_l4w) else "N/A")
                    st.caption("Percentage change calculated based on the sum of rates across all displayed SKUs."); st.markdown("---")
                    rate_formatters = { 'Rate (LW TY)': "{:,.1f}",'Rate (L4W TY)': "{:,.1f}", 'Rate (LW LY)': "{:,.1f}",'Rate (L4W LY)': "{:,.1f}", }; percent_cols = ['% Diff (LW TY vs LY)', '% Diff (L4W TY vs LY)', '% Diff (LW vs L4W TY)']
                    for col in percent_cols: rate_formatters[col] = lambda x: f"{x:+.1f}%" if pd.notna(x) else '-'
                    styler = sales_rate_df.style.format(rate_formatters, na_rep="-", precision=1)
                    for col in percent_cols:
                        if col in styler.data.columns: styler = styler.applymap(color_diff, subset=[col])
                    st.dataframe(styler, use_container_width=True, hide_index=True)
                else: st.caption("Could not calculate sales rates.")
            else: st.caption("Could not determine latest date for rate calculation.")
        else: st.caption(f"No data found for Season '{selected_season}'{', selected Channels' if selected_channels else ''} to calculate sales rates.")
    else: st.info("Select a Season above to view recent sales rates.")


    # --- Global Forecasting Section ---
    st.markdown("---"); st.markdown("## Overall Unit Forecast")
    if all_listing_agg_summaries:
        st.markdown("Generate a simple forecast based on total units sold across all displayed listings for the selected season and period.")
        try:
            overall_agg_summary = pd.concat(all_listing_agg_summaries, ignore_index=True); overall_agg_summary.dropna(subset=[CUSTOM_YEAR_COL, SKU_COL, LISTING_COL, 'Total_Units'], inplace=True); overall_agg_summary[CUSTOM_YEAR_COL] = pd.to_numeric(overall_agg_summary[CUSTOM_YEAR_COL], errors='coerce').astype('Int64'); overall_agg_summary.dropna(subset=[CUSTOM_YEAR_COL], inplace=True)
            if overall_agg_summary.empty: raise ValueError("Aggregated summary for forecast empty.")
            overall_units_summary = overall_agg_summary.groupby([LISTING_COL, SKU_COL, CUSTOM_YEAR_COL])['Total_Units'].sum().reset_index();
            if overall_units_summary.empty: raise ValueError("Grouped summary for pivot empty.")
            overall_pivot_units = overall_units_summary.pivot_table(index=[LISTING_COL, SKU_COL], columns=CUSTOM_YEAR_COL, values='Total_Units', aggfunc='sum').fillna(0)
            if overall_pivot_units.empty: raise ValueError("Forecast pivot table empty.")
            overall_year_cols_raw = overall_pivot_units.columns; overall_year_cols = sorted([yr for yr in overall_year_cols_raw if pd.notna(yr) and isinstance(yr, (int, float, np.integer))])
            if not overall_year_cols: raise ValueError("No valid year columns in forecast pivot.")
            overall_pivot_units = overall_pivot_units.astype(int); fc1, fc2 = st.columns(2);
            with fc1: ref_year = st.selectbox("Select Reference Year (Overall)", options=overall_year_cols, index=len(overall_year_cols)-1 if overall_year_cols else 0, key="forecast_reference_year", format_func=lambda x: str(int(x)))
            with fc2: factor = st.number_input("Percentage Factor (%) (Overall)", min_value=0, max_value=500, value=100, step=5, key="forecast_factor")
            if ref_year is not None:
                if ref_year in overall_pivot_units.columns:
                    forecast_data = [];
                    for (listing, sku), row in overall_pivot_units.iterrows():
                        ref_u = row[ref_year]; fc_u = math.ceil((ref_u * factor)/100.0)
                        if fc_u > 0: forecast_data.append({"Listing": listing, "Product SKU": sku, f"Forecast Units ({factor}%)": fc_u})
                    if forecast_data: fc_df = pd.DataFrame(forecast_data); fc_df.sort_values(by=["Listing", "Product SKU"], inplace=True); fc_cols = ["Listing", "Product SKU", f"Forecast Units ({factor}%)"]; st.dataframe(fc_df[fc_cols].style.format({f"Forecast Units ({factor}%)": "{:,}"}), use_container_width=True, hide_index=True)
                    else: st.caption(f"No SKUs found with a positive forecast based on reference year {int(ref_year)} and factor {factor}%.")
                else: st.warning(f"Selected reference year ({int(ref_year)}) not found in aggregated data columns.")
            else: st.warning("No reference year selected or available.")
        except Exception as e: st.error(f"Could not prepare or display overall unit forecast: {e}")
    elif selected_season: st.info(f"No historical data available from Season '{selected_season}' ({period_str}) to generate forecast.")
    elif not selected_season: st.info("Please select a Season to view historical data and forecast.")

# --- End of display_tab function ---