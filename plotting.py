# plotting.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import get_custom_week_date_range # Import the helper function
from config import ( # Import column names from config
    CUSTOM_YEAR_COL, WEEK_AS_INT_COL, QUARTER_COL, SALES_VALUE_GBP_COL,
    LISTING_COL, PRODUCT_COL, SKU_COL, SALES_CHANNEL_COL, ORDER_QTY_COL_RAW,
    SALES_VALUE_TRANS_CURRENCY_COL, ORIGINAL_CURRENCY_COL, DATE_COL,
    SEASON_COL, YEAR_COL_RAW # Include YEAR_COL_RAW if used in daily price chart
)
import datetime # Import datetime for formatting

# =============================================================================
# Charting and Table Functions
# =============================================================================

def create_yoy_trends_chart(data, selected_years,
                            selected_channels=None, selected_listings=None,
                            selected_products=None, time_grouping="Week",
                            week_range=None, selected_season=None):
    """Creates the YOY Trends line chart, incorporating week range, season filter, and date range in tooltip."""
    filtered = data.copy()
    # Apply filters
    if selected_years:
        filtered = filtered[filtered[CUSTOM_YEAR_COL].isin([int(y) for y in selected_years])]
    if selected_season and selected_season != "ALL":
        if SEASON_COL in filtered.columns:
            filtered = filtered[filtered[SEASON_COL] == selected_season]
        else:
            st.warning(f"Column '{SEASON_COL}' not found for filtering.")
    if selected_channels and len(selected_channels) > 0:
        if SALES_CHANNEL_COL in filtered.columns:
            filtered = filtered[filtered[SALES_CHANNEL_COL].isin(selected_channels)]
        else: st.warning(f"Column '{SALES_CHANNEL_COL}' not found for filtering.")
    if selected_listings and len(selected_listings) > 0:
        if LISTING_COL in filtered.columns:
            filtered = filtered[filtered[LISTING_COL].isin(selected_listings)]
        else: st.warning(f"Column '{LISTING_COL}' not found for filtering.")
    if selected_products and len(selected_products) > 0:
        if PRODUCT_COL in filtered.columns:
            filtered = filtered[filtered[PRODUCT_COL].isin(selected_products)]
        else: st.warning(f"Column '{PRODUCT_COL}' not found for filtering.")

    if week_range:
        start_week, end_week = week_range
        if WEEK_AS_INT_COL in filtered.columns:
            filtered[WEEK_AS_INT_COL] = pd.to_numeric(filtered[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
            filtered.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
            if not filtered.empty:
                 filtered = filtered[(filtered[WEEK_AS_INT_COL] >= start_week) & (filtered[WEEK_AS_INT_COL] <= end_week)]
        else:
            st.warning(f"Column '{WEEK_AS_INT_COL}' not found for week range filtering in YOY chart.")

    if filtered.empty:
        st.warning("No data available for YOY Trends chart with selected filters.")
        return go.Figure()

    # Group data based on time_grouping
    if time_grouping == "Week":
        if WEEK_AS_INT_COL not in filtered.columns or CUSTOM_YEAR_COL not in filtered.columns:
             st.error(f"Critical Error: '{WEEK_AS_INT_COL}' or '{CUSTOM_YEAR_COL}' column lost for YOY chart grouping.")
             return go.Figure()
        grouped = filtered.groupby([CUSTOM_YEAR_COL, WEEK_AS_INT_COL])[SALES_VALUE_GBP_COL].sum().reset_index()
        x_col = WEEK_AS_INT_COL
        x_axis_label = "Week"
        grouped = grouped.sort_values(by=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL])
        title = "Weekly Revenue Trends by Custom Week Year"

        # --- Add Date Range Calculation ---
        def get_date_range_str(row):
            # Check if required columns exist and are valid before calling helper
            if CUSTOM_YEAR_COL in row and WEEK_AS_INT_COL in row and pd.notna(row[CUSTOM_YEAR_COL]) and pd.notna(row[WEEK_AS_INT_COL]):
                 start_dt, end_dt = get_custom_week_date_range(row[CUSTOM_YEAR_COL], row[WEEK_AS_INT_COL])
                 if start_dt and end_dt:
                     return f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d')}"
            return ""

        grouped[CUSTOM_YEAR_COL] = pd.to_numeric(grouped[CUSTOM_YEAR_COL], errors='coerce').astype('Int64')
        grouped[WEEK_AS_INT_COL] = pd.to_numeric(grouped[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
        grouped.dropna(subset=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL], inplace=True)

        if not grouped.empty:
            grouped['Week_Date_Range'] = grouped.apply(get_date_range_str, axis=1)
        else:
            grouped['Week_Date_Range'] = pd.Series(dtype='str')
        # --- End Date Range Calculation ---

    else: # Assume Quarter
        if QUARTER_COL not in filtered.columns or CUSTOM_YEAR_COL not in filtered.columns:
            st.error(f"Critical Error: '{QUARTER_COL}' or '{CUSTOM_YEAR_COL}' column lost for YOY chart grouping.")
            return go.Figure()
        grouped = filtered.groupby([CUSTOM_YEAR_COL, QUARTER_COL])[SALES_VALUE_GBP_COL].sum().reset_index()
        x_col = QUARTER_COL
        x_axis_label = "Quarter"
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        grouped[QUARTER_COL] = pd.Categorical(grouped[QUARTER_COL], categories=quarter_order, ordered=True)
        grouped = grouped.sort_values(by=[CUSTOM_YEAR_COL, QUARTER_COL])
        title = "Quarterly Revenue Trends by Custom Week Year"
        grouped['Week_Date_Range'] = "" # Add empty column for consistency

    if grouped.empty:
        st.warning("No data available after grouping for YOY Trends chart.")
        return go.Figure()

    grouped["RevenueK"] = grouped[SALES_VALUE_GBP_COL] / 1000

    # --- Update custom_data and hovertemplate ---
    custom_data_cols = ["RevenueK", "Week_Date_Range"]
    hover_template_str = f"<b>Week:</b> %{{x}}<br><b>Dates:</b> %{{customdata[1]}}<br><b>Revenue:</b> %{{customdata[0]:.1f}}K<extra></extra>"

    fig = px.line(grouped, x=x_col, y=SALES_VALUE_GBP_COL, color=CUSTOM_YEAR_COL, markers=True,
                  title=title,
                  labels={SALES_VALUE_GBP_COL: "Revenue (£)", x_col: x_axis_label, CUSTOM_YEAR_COL: "Year"},
                  custom_data=custom_data_cols)

    fig.update_traces(hovertemplate=hover_template_str)
    # --- End Update ---

    # Axis range logic remains the same
    if time_grouping == "Week":
        if not grouped.empty and WEEK_AS_INT_COL in grouped.columns:
            min_week_data = grouped[WEEK_AS_INT_COL].min()
            max_week_data = grouped[WEEK_AS_INT_COL].max()
            if pd.isna(min_week_data): min_week_data = 1
            if pd.isna(max_week_data): max_week_data = 52
            min_week_data = int(min_week_data)
            max_week_data = int(max_week_data)
        else:
            min_week_data, max_week_data = 1, 52
        min_week_plot = week_range[0] if week_range else min_week_data
        max_week_plot = week_range[1] if week_range else max_week_data
        fig.update_xaxes(range=[max(0.8, min_week_plot - 0.2), max_week_plot + 0.2], dtick=5)

    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50), legend_title_text='Year')
    fig.update_layout(hovermode='x unified')
    return fig


def create_pivot_table(data, selected_years, selected_quarters, selected_channels,
                       selected_listings, selected_products, grouping_key="Listing"):
    """Creates the pivot table."""
    filtered = data.copy()
    # Apply filters
    if selected_years:
        filtered = filtered[filtered[CUSTOM_YEAR_COL].isin([int(y) for y in selected_years])]
    if selected_quarters:
        if QUARTER_COL in filtered.columns:
             filtered = filtered[filtered[QUARTER_COL].isin(selected_quarters)]
        else: st.warning(f"Column '{QUARTER_COL}' not found for filtering pivot table.")
    if selected_channels and len(selected_channels) > 0:
        if SALES_CHANNEL_COL in filtered.columns:
            filtered = filtered[filtered[SALES_CHANNEL_COL].isin(selected_channels)]
        else: st.warning(f"Column '{SALES_CHANNEL_COL}' not found for filtering pivot table.")
    if selected_listings and len(selected_listings) > 0:
        if LISTING_COL in filtered.columns:
            filtered = filtered[filtered[LISTING_COL].isin(selected_listings)]
        else: st.warning(f"Column '{LISTING_COL}' not found for filtering pivot table.")
    if grouping_key == PRODUCT_COL and selected_products and len(selected_products) > 0:
        if PRODUCT_COL in filtered.columns:
            filtered = filtered[filtered[PRODUCT_COL].isin(selected_products)]
        else: st.warning(f"Column '{PRODUCT_COL}' not found for filtering pivot table.")

    if filtered.empty:
        st.warning("No data available for Pivot Table with selected filters.")
        return pd.DataFrame({grouping_key: ["No data"]})

    if grouping_key not in filtered.columns:
         st.error(f"Required grouping column ('{grouping_key}') not found for creating pivot table.")
         return pd.DataFrame({grouping_key: ["Missing grouping column"]})
    if WEEK_AS_INT_COL not in filtered.columns:
         st.error(f"Required column ('{WEEK_AS_INT_COL}') not found for creating pivot table.")
         return pd.DataFrame({grouping_key: [f"Missing '{WEEK_AS_INT_COL}' column"]})
    if SALES_VALUE_GBP_COL not in filtered.columns:
         st.error(f"Required column ('{SALES_VALUE_GBP_COL}') not found for creating pivot table.")
         return pd.DataFrame({grouping_key: [f"Missing '{SALES_VALUE_GBP_COL}' column"]})


    filtered[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered[SALES_VALUE_GBP_COL], errors='coerce')
    filtered.dropna(subset=[SALES_VALUE_GBP_COL, WEEK_AS_INT_COL, grouping_key], inplace=True)

    if filtered.empty:
         st.warning("No valid data left for Pivot Table after cleaning.")
         return pd.DataFrame({grouping_key: ["No valid data"]})

    pivot = pd.pivot_table(filtered, values=SALES_VALUE_GBP_COL, index=grouping_key,
                           columns=WEEK_AS_INT_COL, aggfunc="sum", fill_value=0)

    if pivot.empty:
        st.warning("Pivot table is empty after grouping.")
        return pd.DataFrame({grouping_key: ["No results"]})

    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0).astype(int) # Convert values to int after summing

    # Rename columns to 'Week X'
    new_columns = {}
    # Use Int64Dtype check for pandas >= 1.0
    Int64Dtype = pd.Int64Dtype() if hasattr(pd, 'Int64Dtype') else 'Int64'

    for col in pivot.columns:
        # Check if column name is numeric (int, float, or Int64) and not the Total Revenue column
        if isinstance(col, (int, float)) or (hasattr(col, 'dtype') and isinstance(col.dtype, type(Int64Dtype))):
             if col != "Total Revenue": # Check name explicitly as well
                 try:
                     week_num = int(col) if pd.notna(col) else 'NaN'
                     new_columns[col] = f"Week {week_num}"
                 except (ValueError, TypeError):
                     new_columns[col] = str(col) # Fallback if conversion fails
        elif col == "Total Revenue":
             new_columns[col] = "Total Revenue"
        else:
            # Handle potential non-numeric week column names if they arise
            new_columns[col] = str(col)

    pivot = pivot.rename(columns=new_columns)

    # Sort columns: Week 1, Week 2, ..., Total Revenue
    week_cols = sorted([col for col in pivot.columns if col.startswith("Week ") and col.split(' ')[1].isdigit()],
                       key=lambda x: int(x.split()[1]))
    if "Total Revenue" in pivot.columns:
        pivot = pivot[week_cols + ["Total Revenue"]]
    else:
        pivot = pivot[week_cols]

    return pivot


# <<< MODIFIED: Added selected_products parameter and filter logic >>>
def create_sku_line_chart(data, sku_text, selected_years,
                          selected_channels=None, week_range=None, selected_products=None): # Added selected_products
    """Creates the SKU Trends line chart with date range in tooltip and product filter."""
    required_cols = {SKU_COL, CUSTOM_YEAR_COL, WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW}
    # Add PRODUCT_COL if it's expected for filtering
    if PRODUCT_COL in data.columns:
        required_cols.add(PRODUCT_COL)

    if not required_cols.issubset(data.columns):
        missing = required_cols.difference(data.columns)
        if selected_products and PRODUCT_COL not in data.columns:
             st.error(f"Cannot filter by Product: Column '{PRODUCT_COL}' not found in data.")
        else:
             st.error(f"Dataset is missing required columns for SKU chart: {missing}")
        return go.Figure().update_layout(title_text=f"Missing data for SKU Chart: {missing}")

    filtered = data.copy()

    # --- ADDED: Product Filter Logic ---
    if selected_products and len(selected_products) > 0:
        if PRODUCT_COL in filtered.columns:
            filtered = filtered[filtered[PRODUCT_COL].isin(selected_products)]
        else:
            st.warning(f"Column '{PRODUCT_COL}' not found, cannot apply Product filter.")
    # --- END ADDED ---

    # Apply SKU text filter *after* product filter
    if SKU_COL in filtered.columns:
         filtered[SKU_COL] = filtered[SKU_COL].astype(str)
         if sku_text and sku_text.strip() != "":
              filtered = filtered[filtered[SKU_COL].str.contains(sku_text, case=False, na=False)]
    else:
         st.error(f"Column '{SKU_COL}' not found.")
         return go.Figure().update_layout(title_text=f"Column '{SKU_COL}' not found")

    # Apply other filters
    if selected_years:
        if CUSTOM_YEAR_COL in filtered.columns:
             filtered = filtered[filtered[CUSTOM_YEAR_COL].isin([int(y) for y in selected_years])]
        else:
             st.warning(f"Column '{CUSTOM_YEAR_COL}' not found for year filtering.")
             return go.Figure().update_layout(title_text=f"Missing '{CUSTOM_YEAR_COL}' column")

    if selected_channels and len(selected_channels) > 0:
        if SALES_CHANNEL_COL in filtered.columns:
            filtered = filtered[filtered[SALES_CHANNEL_COL].isin(selected_channels)]
        else: st.warning(f"Column '{SALES_CHANNEL_COL}' not found for filtering SKU chart.")

    if week_range:
        start_week, end_week = week_range
        if WEEK_AS_INT_COL in filtered.columns:
             filtered[WEEK_AS_INT_COL] = pd.to_numeric(filtered[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
             filtered.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
             if not filtered.empty:
                   filtered = filtered[(filtered[WEEK_AS_INT_COL] >= start_week) & (filtered[WEEK_AS_INT_COL] <= end_week)]
        else: st.warning(f"Column '{WEEK_AS_INT_COL}' not found for week range filtering in SKU chart.")

    # Check if data remains after all filters
    if filtered.empty:
        search_term_msg = f"matching '{sku_text}'" if sku_text and sku_text.strip() != "" else "within selected products"
        st.warning(f"No data available for SKUs {search_term_msg} with selected filters.")
        return go.Figure().update_layout(title_text=f"No data for selected SKUs")

    # Ensure numeric types before aggregation
    filtered[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered[SALES_VALUE_GBP_COL], errors='coerce')
    filtered[ORDER_QTY_COL_RAW] = pd.to_numeric(filtered[ORDER_QTY_COL_RAW], errors='coerce')
    agg_check_cols = [SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW]
    if CUSTOM_YEAR_COL in filtered.columns: agg_check_cols.append(CUSTOM_YEAR_COL)
    if WEEK_AS_INT_COL in filtered.columns: agg_check_cols.append(WEEK_AS_INT_COL)
    filtered.dropna(subset=agg_check_cols, inplace=True)


    if filtered.empty:
         st.warning(f"No valid numeric data for selected SKUs after cleaning.")
         return go.Figure().update_layout(title_text=f"No valid data for selected SKUs")

    # Check again if essential columns for grouping exist
    if CUSTOM_YEAR_COL not in filtered.columns or WEEK_AS_INT_COL not in filtered.columns:
         st.error("Essential columns for grouping (Year, Week) are missing after filtering/cleaning.")
         return go.Figure().update_layout(title_text="Grouping columns missing")


    # Group by week and year
    weekly_sku = filtered.groupby([CUSTOM_YEAR_COL, WEEK_AS_INT_COL]).agg({
        SALES_VALUE_GBP_COL: "sum",
        ORDER_QTY_COL_RAW: "sum"
    }).reset_index().sort_values(by=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL])

    if weekly_sku.empty:
        st.warning("No data after grouping for SKU chart.")
        return go.Figure().update_layout(title_text=f"No data for selected SKUs after grouping")

    # Add Date Range Calculation
    def get_date_range_str_sku(row):
        if CUSTOM_YEAR_COL in row and WEEK_AS_INT_COL in row and pd.notna(row[CUSTOM_YEAR_COL]) and pd.notna(row[WEEK_AS_INT_COL]):
             start_dt, end_dt = get_custom_week_date_range(row[CUSTOM_YEAR_COL], row[WEEK_AS_INT_COL])
             if start_dt and end_dt:
                 return f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d')}"
        return ""

    weekly_sku[CUSTOM_YEAR_COL] = pd.to_numeric(weekly_sku[CUSTOM_YEAR_COL], errors='coerce').astype('Int64')
    weekly_sku[WEEK_AS_INT_COL] = pd.to_numeric(weekly_sku[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
    weekly_sku.dropna(subset=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL], inplace=True)

    if not weekly_sku.empty:
        weekly_sku['Week_Date_Range'] = weekly_sku.apply(get_date_range_str_sku, axis=1)
    else:
        weekly_sku['Week_Date_Range'] = pd.Series(dtype='str')

    weekly_sku["RevenueK"] = weekly_sku[SALES_VALUE_GBP_COL] / 1000

    # Determine axis range based on data or slider
    min_week_data, max_week_data = 1, 52
    if not weekly_sku[WEEK_AS_INT_COL].empty:
         min_week_data_calc = weekly_sku[WEEK_AS_INT_COL].min()
         max_week_data_calc = weekly_sku[WEEK_AS_INT_COL].max()
         if pd.notna(min_week_data_calc): min_week_data = int(min_week_data_calc)
         if pd.notna(max_week_data_calc): max_week_data = int(max_week_data_calc)

    if week_range:
        min_week_plot, max_week_plot = week_range
    else:
        min_week_plot, max_week_plot = min_week_data, max_week_data

    # Update chart title based on filters
    chart_title = "Weekly Revenue Trends"
    if sku_text and sku_text.strip() != "":
        chart_title += f" for SKU matching: '{sku_text}'"
        if selected_products:
             chart_title += f" within selected Product(s)"
    elif selected_products:
        chart_title += f" for selected Product(s)"


    # Update custom_data and hovertemplate
    custom_data_cols_sku = ["RevenueK", ORDER_QTY_COL_RAW, "Week_Date_Range"]
    hover_template_str_sku = f"<b>Week:</b> %{{x}}<br><b>Dates:</b> %{{customdata[2]}}<br><b>Revenue:</b> %{{customdata[0]:.1f}}K<br><b>Units Sold:</b> %{{customdata[1]}}<extra></extra>"

    fig = px.line(weekly_sku, x=WEEK_AS_INT_COL, y=SALES_VALUE_GBP_COL, color=CUSTOM_YEAR_COL, markers=True,
                  title=chart_title,
                  labels={SALES_VALUE_GBP_COL: "Revenue (£)", CUSTOM_YEAR_COL: "Year", WEEK_AS_INT_COL: "Week"},
                  custom_data=custom_data_cols_sku)

    fig.update_traces(hovertemplate=hover_template_str_sku)

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            tickmode="linear",
            range=[max(0.8, min_week_plot - 0.2), max_week_plot + 0.2],
            dtick=5 if (max_week_plot - min_week_plot) > 10 else 1
            ),
        yaxis=dict(rangemode="tozero"),
        margin=dict(t=50, b=50),
        legend_title_text='Year'
        )
    return fig


def create_daily_price_chart(data, listing, selected_years, selected_quarters, selected_channels, week_range=None):
    """Creates the Daily Average Price line chart."""
    year_col_to_use = CUSTOM_YEAR_COL
    required_cols = {DATE_COL, LISTING_COL, year_col_to_use, SALES_VALUE_TRANS_CURRENCY_COL, ORDER_QTY_COL_RAW, WEEK_AS_INT_COL, QUARTER_COL, SALES_CHANNEL_COL}
    if ORIGINAL_CURRENCY_COL in data.columns:
        required_cols.add(ORIGINAL_CURRENCY_COL)
    if not required_cols.issubset(data.columns):
          missing = required_cols.difference(data.columns)
          st.error(f"Dataset is missing required columns for Daily Price chart: {missing}")
          return go.Figure().update_layout(title_text=f"Missing data for Daily Prices: {missing}")
    selected_years_int = [int(y) for y in selected_years]
    df_listing = data[(data[LISTING_COL] == listing) & (data[year_col_to_use].isin(selected_years_int))].copy()
    if selected_quarters:
        if QUARTER_COL not in df_listing.columns:
            st.warning(f"Column '{QUARTER_COL}' not found for Daily Price filtering.")
        else:
            df_listing = df_listing[df_listing[QUARTER_COL].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        if SALES_CHANNEL_COL not in df_listing.columns:
            st.warning(f"Column '{SALES_CHANNEL_COL}' not found for Daily Price filtering.")
        else:
            df_listing = df_listing[df_listing[SALES_CHANNEL_COL].isin(selected_channels)]
    if week_range:
        start_week, end_week = week_range
        if WEEK_AS_INT_COL not in df_listing.columns:
             st.warning(f"Column '{WEEK_AS_INT_COL}' not found for Daily Price week range filtering.")
        else:
             df_listing[WEEK_AS_INT_COL] = pd.to_numeric(df_listing[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
             df_listing.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
             if not df_listing.empty:
                   df_listing = df_listing[(df_listing[WEEK_AS_INT_COL] >= start_week) & (df_listing[WEEK_AS_INT_COL] <= end_week)]
    if df_listing.empty:
        st.warning(f"No data available for '{listing}' with the selected filters.")
        return go.Figure().update_layout(title_text=f"No data for '{listing}' with filters")
    display_currency = "Currency"
    currency_symbol = ""
    if ORIGINAL_CURRENCY_COL in df_listing.columns and not df_listing[ORIGINAL_CURRENCY_COL].dropna().empty:
         unique_currencies = df_listing[ORIGINAL_CURRENCY_COL].dropna().unique()
         if len(unique_currencies) > 0:
             display_currency = unique_currencies[0]
             if display_currency == "GBP": currency_symbol = "£"
             elif display_currency == "USD": currency_symbol = "$"
             elif display_currency == "EUR": currency_symbol = "€"
             if len(unique_currencies) > 1:
                  st.info(f"Note: Multiple transaction currencies found ({unique_currencies}) for '{listing}'. Displaying average price based on '{SALES_VALUE_TRANS_CURRENCY_COL}' in {display_currency}.")
    df_listing[DATE_COL] = pd.to_datetime(df_listing[DATE_COL], errors='coerce')
    df_listing[SALES_VALUE_TRANS_CURRENCY_COL] = pd.to_numeric(df_listing[SALES_VALUE_TRANS_CURRENCY_COL], errors='coerce')
    df_listing[ORDER_QTY_COL_RAW] = pd.to_numeric(df_listing[ORDER_QTY_COL_RAW], errors='coerce')
    df_listing.dropna(subset=[DATE_COL, SALES_VALUE_TRANS_CURRENCY_COL, ORDER_QTY_COL_RAW, year_col_to_use], inplace=True)
    df_listing = df_listing[df_listing[ORDER_QTY_COL_RAW] > 0]
    if df_listing.empty:
        st.warning(f"No valid sales/quantity data for '{listing}' to calculate daily price after cleaning.")
        return go.Figure().update_layout(title_text=f"No valid data for '{listing}' after cleaning")
    grouped = df_listing.groupby([df_listing[DATE_COL].dt.date, year_col_to_use]).agg(
        Total_Sales_Value=(SALES_VALUE_TRANS_CURRENCY_COL, "sum"),
        Total_Order_Quantity=(ORDER_QTY_COL_RAW, "sum")
    ).reset_index()
    if DATE_COL not in grouped.columns and 'level_0' in grouped.columns:
        grouped = grouped.rename(columns={'level_0': DATE_COL})
    if DATE_COL not in grouped.columns:
        st.error("Failed to identify Date column after grouping.")
        return go.Figure().update_layout(title_text="Error processing grouped data")
    grouped["Average Price"] = grouped["Total_Sales_Value"] / grouped["Total_Order_Quantity"]
    grouped[DATE_COL] = pd.to_datetime(grouped[DATE_COL])
    dfs_processed = []
    for yr in selected_years_int:
        df_year = grouped[grouped[year_col_to_use] == yr].copy()
        if df_year.empty:
            continue
        df_year["Day"] = df_year[DATE_COL].dt.dayofyear
        if df_year["Day"].empty or df_year["Day"].isna().all():
            continue
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())
        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))
        df_year["Average Price"] = df_year["Average Price"].ffill()
        df_year[year_col_to_use] = yr
        df_year["Average Price"] = pd.to_numeric(df_year["Average Price"], errors='coerce')
        df_year.dropna(subset=["Average Price"], inplace=True)
        if df_year.empty: continue
        prices = df_year["Average Price"].values.copy()
        last_valid_price = None
        for i in range(len(prices)):
            current_price = prices[i]
            if pd.notna(current_price):
                if last_valid_price is not None:
                    if current_price < 0.75 * last_valid_price:
                        prices[i] = last_valid_price
                    elif current_price > 1.25 * last_valid_price:
                         prices[i] = last_valid_price
                last_valid_price = prices[i]
        df_year["Smoothed Average Price"] = prices
        df_year = df_year.reset_index()
        df_year.dropna(subset=["Smoothed Average Price"], inplace=True)
        if not df_year.empty:
            dfs_processed.append(df_year)
    if not dfs_processed:
        st.warning("No data available after processing for the Daily Price chart.")
        return go.Figure().update_layout(title_text=f"No processed data for '{listing}'")
    combined = pd.concat(dfs_processed, ignore_index=True)
    if combined.empty:
        st.warning("Combined data is empty for the Daily Price chart.")
        return go.Figure().update_layout(title_text=f"Combined data empty for '{listing}'")
    fig = px.line(
        combined,
        x="Day",
        y="Smoothed Average Price",
        color=year_col_to_use,
        title=f"Daily Average Price for {listing}",
        labels={"Day": "Day of Year", "Smoothed Average Price": f"Avg Price ({currency_symbol}{display_currency})", year_col_to_use: "Year"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50), legend_title_text='Year')
    fig.update_layout(hovermode='x unified')
    return fig

