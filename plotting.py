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

# <<< MODIFIED: Added excluded_products parameter and filter logic >>>
def create_yoy_trends_chart(data, selected_years,
                            selected_channels=None, selected_listings=None,
                            selected_products=None, time_grouping="Week",
                            week_range=None, selected_season=None,
                            excluded_listings=None, # <-- Existing exclude parameter
                            excluded_products=None): # <-- Added new exclude parameter
    """Creates the YOY Trends line chart, incorporating week range, season filter,
       listing/product exclusion, and date range in tooltip."""

    # Start with a copy of the data
    filtered = data.copy()

    # --- Apply Inclusion Filters FIRST ---
    if selected_years:
        if CUSTOM_YEAR_COL in filtered.columns:
            filtered = filtered[filtered[CUSTOM_YEAR_COL].isin([int(y) for y in selected_years])]
        else:
            st.warning(f"Column '{CUSTOM_YEAR_COL}' not found for year filtering.")

    if selected_season and selected_season != "ALL":
        if SEASON_COL in filtered.columns:
            filtered = filtered[filtered[SEASON_COL] == selected_season]
        else:
            st.warning(f"Column '{SEASON_COL}' not found for season filtering.")

    if selected_channels and len(selected_channels) > 0:
        if SALES_CHANNEL_COL in filtered.columns:
            filtered = filtered[filtered[SALES_CHANNEL_COL].isin(selected_channels)]
        else:
            st.warning(f"Column '{SALES_CHANNEL_COL}' not found for channel filtering.")

    if selected_listings and len(selected_listings) > 0:
        if LISTING_COL in filtered.columns:
            filtered = filtered[filtered[LISTING_COL].isin(selected_listings)]
        else:
            st.warning(f"Column '{LISTING_COL}' not found for listing inclusion filtering.")

    if selected_products and len(selected_products) > 0:
        if PRODUCT_COL in filtered.columns:
            filtered = filtered[filtered[PRODUCT_COL].isin(selected_products)]
        else:
            st.warning(f"Column '{PRODUCT_COL}' not found for product filtering.")

    # --- Apply Exclusion Filters ---
    if excluded_listings and len(excluded_listings) > 0:
        if LISTING_COL in filtered.columns:
            # Use ~ to negate the isin condition, removing the excluded listings
            filtered = filtered[~filtered[LISTING_COL].isin(excluded_listings)]
        else:
            # Warning potentially shown above, pass silently
            pass
    # --- NEW: Apply Product Exclusion Filter ---
    if excluded_products and len(excluded_products) > 0:
        if PRODUCT_COL in filtered.columns:
            # Use ~ to negate the isin condition, removing the excluded products
            filtered = filtered[~filtered[PRODUCT_COL].isin(excluded_products)]
        else:
             # Warning potentially shown above, pass silently
            pass
    # --- END NEW ---

    # Apply week range filter (after other filters)
    if week_range:
        start_week, end_week = week_range
        if WEEK_AS_INT_COL in filtered.columns:
            # Ensure week column is numeric, coercing errors and handling potential NaNs
            filtered[WEEK_AS_INT_COL] = pd.to_numeric(filtered[WEEK_AS_INT_COL], errors='coerce')
            filtered.dropna(subset=[WEEK_AS_INT_COL], inplace=True) # Drop rows where week couldn't be converted
            # Convert to Int64 AFTER dropping NaNs
            if not filtered.empty:
                 filtered[WEEK_AS_INT_COL] = filtered[WEEK_AS_INT_COL].astype('Int64')
                 # Apply the week range filter
                 filtered = filtered[
                     (filtered[WEEK_AS_INT_COL] >= start_week) &
                     (filtered[WEEK_AS_INT_COL] <= end_week)
                 ]
        else:
            st.warning(f"Column '{WEEK_AS_INT_COL}' not found for week range filtering in YOY chart.")

    # Check if data remains after all filters
    if filtered.empty:
        st.warning("No data available for YOY Trends chart with selected filters (including exclusions).")
        return go.Figure() # Return an empty figure

    # --- Grouping and Plotting ---
    grouped = pd.DataFrame() # Initialize empty DataFrame
    x_col = ""
    x_axis_label = ""
    title = ""

    # Group data based on time_grouping
    if time_grouping == "Week":
        # Check for essential grouping columns *after* filtering
        if WEEK_AS_INT_COL not in filtered.columns or CUSTOM_YEAR_COL not in filtered.columns:
            st.error(f"Critical Error: '{WEEK_AS_INT_COL}' or '{CUSTOM_YEAR_COL}' column lost or missing after filtering for YOY chart grouping.")
            return go.Figure() # Return empty figure if essential columns missing

        # Ensure Sales Value column is numeric before grouping
        if SALES_VALUE_GBP_COL in filtered.columns:
            filtered[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered[SALES_VALUE_GBP_COL], errors='coerce')
            filtered.dropna(subset=[SALES_VALUE_GBP_COL], inplace=True) # Drop rows where conversion failed
        else:
             st.error(f"Critical Error: '{SALES_VALUE_GBP_COL}' column missing for YOY chart grouping.")
             return go.Figure()

        if filtered.empty: # Check again after potential dropna
             st.warning("No valid numeric sales data available after filtering for YOY Trends chart.")
             return go.Figure()

        # Perform grouping
        grouped = filtered.groupby([CUSTOM_YEAR_COL, WEEK_AS_INT_COL])[SALES_VALUE_GBP_COL].sum().reset_index()
        x_col = WEEK_AS_INT_COL
        x_axis_label = "Week"
        grouped = grouped.sort_values(by=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL])
        title = "Weekly Revenue Trends by Custom Week Year"

        # --- Add Date Range Calculation ---
        def get_date_range_str(row):
            # Check if required columns exist and are valid before calling helper
            if CUSTOM_YEAR_COL in row and WEEK_AS_INT_COL in row and pd.notna(row[CUSTOM_YEAR_COL]) and pd.notna(row[WEEK_AS_INT_COL]):
                try:
                    # Ensure types are correct for the helper function
                    year_val = int(row[CUSTOM_YEAR_COL])
                    week_val = int(row[WEEK_AS_INT_COL])
                    start_dt, end_dt = get_custom_week_date_range(year_val, week_val)
                    if start_dt and end_dt:
                        return f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d')}"
                except (ValueError, TypeError):
                    # Handle cases where conversion to int fails or helper function errors
                    return "Invalid Date"
            return "" # Return empty if columns missing or NaN

        # Ensure types before applying the date range function
        grouped[CUSTOM_YEAR_COL] = pd.to_numeric(grouped[CUSTOM_YEAR_COL], errors='coerce').astype('Int64')
        grouped[WEEK_AS_INT_COL] = pd.to_numeric(grouped[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
        grouped.dropna(subset=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL], inplace=True) # Drop if conversion failed

        if not grouped.empty:
            grouped['Week_Date_Range'] = grouped.apply(get_date_range_str, axis=1)
        else:
            # If grouped is empty after potential dropna, create the column anyway
            grouped['Week_Date_Range'] = pd.Series(dtype='str')
        # --- End Date Range Calculation ---

    else: # Assume Quarter (Add similar checks as above if needed)
        if QUARTER_COL not in filtered.columns or CUSTOM_YEAR_COL not in filtered.columns:
            st.error(f"Critical Error: '{QUARTER_COL}' or '{CUSTOM_YEAR_COL}' column lost for YOY chart grouping.")
            return go.Figure()
        # Add check for SALES_VALUE_GBP_COL and numeric conversion if not done already
        if SALES_VALUE_GBP_COL not in filtered.columns:
             st.error(f"Critical Error: '{SALES_VALUE_GBP_COL}' column missing for YOY chart grouping.")
             return go.Figure()
        filtered[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered[SALES_VALUE_GBP_COL], errors='coerce')
        filtered.dropna(subset=[SALES_VALUE_GBP_COL], inplace=True)
        if filtered.empty:
             st.warning("No valid numeric sales data available after filtering for YOY Trends chart (Quarterly).")
             return go.Figure()

        grouped = filtered.groupby([CUSTOM_YEAR_COL, QUARTER_COL])[SALES_VALUE_GBP_COL].sum().reset_index()
        x_col = QUARTER_COL
        x_axis_label = "Quarter"
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        grouped[QUARTER_COL] = pd.Categorical(grouped[QUARTER_COL], categories=quarter_order, ordered=True)
        grouped = grouped.sort_values(by=[CUSTOM_YEAR_COL, QUARTER_COL])
        title = "Quarterly Revenue Trends by Custom Week Year"
        grouped['Week_Date_Range'] = "" # Add empty column for consistency

    # Check if grouping resulted in data
    if grouped.empty:
        st.warning("No data available after grouping for YOY Trends chart.")
        return go.Figure()

    # Calculate Revenue in Thousands for tooltip
    grouped["RevenueK"] = grouped[SALES_VALUE_GBP_COL] / 1000

    # --- Update custom_data and hovertemplate ---
    custom_data_cols = ["RevenueK", "Week_Date_Range"]
    hover_template_str = f"<b>{x_axis_label}:</b> %{{x}}<br>"
    if time_grouping == "Week":
        hover_template_str += f"<b>Dates:</b> %{{customdata[1]}}<br>"
    hover_template_str += f"<b>Revenue:</b> %{{customdata[0]:.1f}}K<extra></extra>" # Use RevenueK

    # Create the line chart
    fig = px.line(grouped, x=x_col, y=SALES_VALUE_GBP_COL, color=CUSTOM_YEAR_COL, markers=True,
                  title=title,
                  labels={SALES_VALUE_GBP_COL: "Revenue (£)", x_col: x_axis_label, CUSTOM_YEAR_COL: "Year"},
                  custom_data=custom_data_cols) # Pass the correct custom data

    # Apply the hover template
    fig.update_traces(hovertemplate=hover_template_str)
    # --- End Update ---

    # --- Axis range logic ---
    if time_grouping == "Week":
        min_week_data, max_week_data = 1, 52 # Defaults
        if not grouped.empty and WEEK_AS_INT_COL in grouped.columns and grouped[WEEK_AS_INT_COL].notna().any():
            min_week_data_calc = grouped[WEEK_AS_INT_COL].min()
            max_week_data_calc = grouped[WEEK_AS_INT_COL].max()
            # Check if min/max are valid numbers before casting to int
            if pd.notna(min_week_data_calc): min_week_data = int(min_week_data_calc)
            if pd.notna(max_week_data_calc): max_week_data = int(max_week_data_calc)

        # Use week_range from slider if provided, otherwise use data range
        min_week_plot = week_range[0] if week_range else min_week_data
        max_week_plot = week_range[1] if week_range else max_week_data

        # Ensure plot range is sensible
        min_plot_final = max(0.8, min_week_plot - 0.2)
        max_plot_final = max_week_plot + 0.2

        # Set x-axis range and ticks
        fig.update_xaxes(range=[min_plot_final, max_plot_final], dtick=5)

    # --- General Layout Updates ---
    fig.update_yaxes(rangemode="tozero") # Ensure y-axis starts at 0
    fig.update_layout(
        margin=dict(t=50, b=50), # Adjust margins
        legend_title_text='Year', # Set legend title
        hovermode='x unified' # Improve hover behavior
    )

    return fig # Return the configured figure


def create_pivot_table(data, selected_years, selected_quarters, selected_channels,
                       selected_listings, selected_products, grouping_key="Listing"):
    """Creates the pivot table."""
    # This function remains unchanged as the request was specific to the YOY chart
    filtered = data.copy()
    # Apply filters (similar logic as YOY chart, ensure columns exist)
    if selected_years:
        if CUSTOM_YEAR_COL in filtered.columns:
            filtered = filtered[filtered[CUSTOM_YEAR_COL].isin([int(y) for y in selected_years])]
        else: st.warning(f"Column '{CUSTOM_YEAR_COL}' not found for pivot table year filter.")
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

    # Apply product filter only if grouping by product
    if grouping_key == PRODUCT_COL and selected_products and len(selected_products) > 0:
        if PRODUCT_COL in filtered.columns:
             filtered = filtered[filtered[PRODUCT_COL].isin(selected_products)]
        else: st.warning(f"Column '{PRODUCT_COL}' not found for filtering pivot table.")

    # Check for required columns for pivot operation itself
    required_pivot_cols = {grouping_key, WEEK_AS_INT_COL, SALES_VALUE_GBP_COL}
    if not required_pivot_cols.issubset(filtered.columns):
        missing = required_pivot_cols.difference(filtered.columns)
        st.error(f"Dataset is missing required columns for Pivot Table operation: {missing}")
        return pd.DataFrame({grouping_key: [f"Missing columns: {missing}"]})

    # Check if data remains after initial filters
    if filtered.empty:
        st.warning("No data available for Pivot Table with selected filters.")
        # Return an empty DataFrame with the grouping key column for consistency
        return pd.DataFrame(columns=[grouping_key])

    # Ensure data types are correct and handle NaNs before pivoting
    filtered[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered[SALES_VALUE_GBP_COL], errors='coerce')
    filtered[WEEK_AS_INT_COL] = pd.to_numeric(filtered[WEEK_AS_INT_COL], errors='coerce') # Ensure week is numeric
    filtered.dropna(subset=[SALES_VALUE_GBP_COL, WEEK_AS_INT_COL, grouping_key], inplace=True)

    if filtered.empty:
         st.warning("No valid data left for Pivot Table after cleaning (NaN removal).")
         return pd.DataFrame(columns=[grouping_key])

    # Ensure week column is integer type for column headers
    filtered[WEEK_AS_INT_COL] = filtered[WEEK_AS_INT_COL].astype(int)

    # Create the pivot table
    try:
        pivot = pd.pivot_table(filtered, values=SALES_VALUE_GBP_COL, index=grouping_key,
                               columns=WEEK_AS_INT_COL, aggfunc="sum", fill_value=0)
    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        return pd.DataFrame({grouping_key: ["Pivot error"]})


    if pivot.empty:
        st.warning("Pivot table is empty after grouping (no combinations found).")
        return pd.DataFrame(columns=[grouping_key]) # Return empty with index name

    # Calculate Total Revenue
    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0).astype(int) # Convert values to int after summing

    # Rename columns to 'Week X'
    # Ensure column names are integers before formatting
    pivot.columns = [f"Week {col}" if isinstance(col, (int, float)) and col != "Total Revenue" else col for col in pivot.columns]


    # Sort columns: Week 1, Week 2, ..., Total Revenue
    week_cols = sorted([col for col in pivot.columns if isinstance(col, str) and col.startswith("Week ") and col.split(' ')[1].isdigit()],
                       key=lambda x: int(x.split()[1]))

    # Construct final column order
    final_cols = week_cols
    if "Total Revenue" in pivot.columns:
        final_cols.append("Total Revenue")

    # Reindex pivot table with sorted columns
    pivot = pivot[final_cols]

    return pivot.reset_index() # Return with index as a column


def create_sku_line_chart(data, sku_text, selected_years,
                          selected_channels=None, week_range=None, selected_products=None):
    """Creates the SKU Trends line chart with date range in tooltip and product filter."""
    # This function remains unchanged
    # Define essential columns for the chart logic
    required_cols = {SKU_COL, CUSTOM_YEAR_COL, WEEK_AS_INT_COL, SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW}
    # Add product column if filtering by it
    if selected_products:
        required_cols.add(PRODUCT_COL)
    # Add channel column if filtering by it
    if selected_channels:
         required_cols.add(SALES_CHANNEL_COL)

    # Check if all required columns are present in the input data
    if not required_cols.issubset(data.columns):
        missing = required_cols.difference(data.columns)
        st.error(f"Dataset is missing required columns for SKU chart: {missing}")
        return go.Figure().update_layout(title_text=f"Missing data for SKU Chart: {missing}")

    # Start filtering
    filtered = data.copy()

    # --- Apply Product Filter FIRST (if provided) ---
    if selected_products and len(selected_products) > 0:
        # PRODUCT_COL existence checked above
        filtered = filtered[filtered[PRODUCT_COL].isin(selected_products)]

    # --- Apply SKU text filter (if provided) ---
    # Ensure SKU column is string type for searching
    filtered[SKU_COL] = filtered[SKU_COL].astype(str)
    if sku_text and sku_text.strip() != "":
        filtered = filtered[filtered[SKU_COL].str.contains(sku_text, case=False, na=False)]

    # Apply other filters (Year, Channel, Week Range)
    if selected_years:
        # CUSTOM_YEAR_COL existence checked above
        filtered = filtered[filtered[CUSTOM_YEAR_COL].isin([int(y) for y in selected_years])]

    if selected_channels and len(selected_channels) > 0:
        # SALES_CHANNEL_COL existence checked above (if selected_channels provided)
        filtered = filtered[filtered[SALES_CHANNEL_COL].isin(selected_channels)]

    if week_range:
        start_week, end_week = week_range
        # WEEK_AS_INT_COL existence checked above
        filtered[WEEK_AS_INT_COL] = pd.to_numeric(filtered[WEEK_AS_INT_COL], errors='coerce')
        filtered.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
        if not filtered.empty:
             filtered[WEEK_AS_INT_COL] = filtered[WEEK_AS_INT_COL].astype('Int64')
             filtered = filtered[
                 (filtered[WEEK_AS_INT_COL] >= start_week) &
                 (filtered[WEEK_AS_INT_COL] <= end_week)
             ]

    # Check if data remains after all filters
    if filtered.empty:
        search_term_msg = f"matching '{sku_text}'" if sku_text and sku_text.strip() != "" else ""
        product_msg = f"within selected Product(s)" if selected_products else ""
        filter_msg = f"{search_term_msg} {product_msg}".strip()
        st.warning(f"No data available for SKUs {filter_msg} with selected filters.")
        return go.Figure().update_layout(title_text=f"No data for selected SKUs")

    # Ensure numeric types before aggregation and handle NaNs
    filtered[SALES_VALUE_GBP_COL] = pd.to_numeric(filtered[SALES_VALUE_GBP_COL], errors='coerce')
    filtered[ORDER_QTY_COL_RAW] = pd.to_numeric(filtered[ORDER_QTY_COL_RAW], errors='coerce')
    # Also ensure grouping columns are valid
    filtered[CUSTOM_YEAR_COL] = pd.to_numeric(filtered[CUSTOM_YEAR_COL], errors='coerce')
    filtered[WEEK_AS_INT_COL] = pd.to_numeric(filtered[WEEK_AS_INT_COL], errors='coerce')

    agg_check_cols = [SALES_VALUE_GBP_COL, ORDER_QTY_COL_RAW, CUSTOM_YEAR_COL, WEEK_AS_INT_COL]
    filtered.dropna(subset=agg_check_cols, inplace=True) # Drop rows with NaN in any essential column

    if filtered.empty:
         st.warning(f"No valid numeric data for selected SKUs after cleaning.")
         return go.Figure().update_layout(title_text=f"No valid data for selected SKUs")

    # Convert grouping columns to integer types AFTER dropping NaNs
    filtered[CUSTOM_YEAR_COL] = filtered[CUSTOM_YEAR_COL].astype(int)
    filtered[WEEK_AS_INT_COL] = filtered[WEEK_AS_INT_COL].astype(int)


    # Group by week and year
    try:
        weekly_sku = filtered.groupby([CUSTOM_YEAR_COL, WEEK_AS_INT_COL]).agg(
            Total_Revenue=(SALES_VALUE_GBP_COL, "sum"),
            Total_Units=(ORDER_QTY_COL_RAW, "sum")
        ).reset_index()
    except Exception as e:
         st.error(f"Error during SKU data aggregation: {e}")
         return go.Figure().update_layout(title_text="Aggregation Error")

    if weekly_sku.empty:
        st.warning("No data after grouping for SKU chart.")
        return go.Figure().update_layout(title_text=f"No data for selected SKUs after grouping")

    # Sort for consistent line plotting
    weekly_sku = weekly_sku.sort_values(by=[CUSTOM_YEAR_COL, WEEK_AS_INT_COL])

    # Add Date Range Calculation
    def get_date_range_str_sku(row):
        # Assumes CUSTOM_YEAR_COL and WEEK_AS_INT_COL are present and valid integers
        try:
            start_dt, end_dt = get_custom_week_date_range(row[CUSTOM_YEAR_COL], row[WEEK_AS_INT_COL])
            if start_dt and end_dt:
                return f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d')}"
        except (ValueError, TypeError, KeyError): # Catch potential errors
             return "Invalid Date"
        return ""

    # Apply date range function safely
    weekly_sku['Week_Date_Range'] = weekly_sku.apply(get_date_range_str_sku, axis=1)

    # Calculate Revenue in Thousands for tooltip
    weekly_sku["RevenueK"] = weekly_sku["Total_Revenue"] / 1000

    # Determine axis range based on data or slider
    min_week_data, max_week_data = 1, 52
    if not weekly_sku[WEEK_AS_INT_COL].empty:
        min_week_data_calc = weekly_sku[WEEK_AS_INT_COL].min()
        max_week_data_calc = weekly_sku[WEEK_AS_INT_COL].max()
        if pd.notna(min_week_data_calc): min_week_data = int(min_week_data_calc)
        if pd.notna(max_week_data_calc): max_week_data = int(max_week_data_calc)

    min_week_plot = week_range[0] if week_range else min_week_data
    max_week_plot = week_range[1] if week_range else max_week_data

    # Update chart title based on filters applied
    chart_title = "Weekly Revenue Trends"
    title_suffix = []
    if sku_text and sku_text.strip() != "":
        title_suffix.append(f"SKU matching: '{sku_text}'")
    if selected_products:
         title_suffix.append(f"selected Product(s)")
    if title_suffix:
         chart_title += " for " + " and ".join(title_suffix)


    # Update custom_data and hovertemplate
    custom_data_cols_sku = ["RevenueK", "Total_Units", "Week_Date_Range"]
    hover_template_str_sku = f"<b>Week:</b> %{{x}}<br><b>Dates:</b> %{{customdata[2]}}<br><b>Revenue:</b> %{{customdata[0]:.1f}}K<br><b>Units Sold:</b> %{{customdata[1]:,.0f}}<extra></extra>" # Format units

    # Create the plot
    fig = px.line(weekly_sku, x=WEEK_AS_INT_COL, y="Total_Revenue", color=CUSTOM_YEAR_COL, markers=True,
                  title=chart_title,
                  labels={"Total_Revenue": "Revenue (£)", CUSTOM_YEAR_COL: "Year", WEEK_AS_INT_COL: "Week"},
                  custom_data=custom_data_cols_sku)

    fig.update_traces(hovertemplate=hover_template_str_sku)

    # Layout updates
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
    # This function remains unchanged
    # Determine which year column to use (assuming CUSTOM_YEAR_COL is preferred)
    year_col_to_use = CUSTOM_YEAR_COL

    # Define required columns, conditionally adding optional ones
    required_cols = {DATE_COL, LISTING_COL, year_col_to_use, SALES_VALUE_TRANS_CURRENCY_COL, ORDER_QTY_COL_RAW, WEEK_AS_INT_COL, QUARTER_COL, SALES_CHANNEL_COL}
    if ORIGINAL_CURRENCY_COL in data.columns:
        required_cols.add(ORIGINAL_CURRENCY_COL)

    # Check for missing columns
    if not required_cols.issubset(data.columns):
         missing = required_cols.difference(data.columns)
         st.error(f"Dataset is missing required columns for Daily Price chart: {missing}")
         return go.Figure().update_layout(title_text=f"Missing data for Daily Prices: {missing}")

    # Convert selected years to integers
    selected_years_int = [int(y) for y in selected_years]

    # Initial filtering by listing and year
    df_listing = data[(data[LISTING_COL] == listing) & (data[year_col_to_use].isin(selected_years_int))].copy()

    # Apply optional filters (Quarter, Channel, Week Range)
    if selected_quarters:
        # QUARTER_COL checked in required_cols
        df_listing = df_listing[df_listing[QUARTER_COL].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        # SALES_CHANNEL_COL checked in required_cols
        df_listing = df_listing[df_listing[SALES_CHANNEL_COL].isin(selected_channels)]
    if week_range:
        start_week, end_week = week_range
        # WEEK_AS_INT_COL checked in required_cols
        df_listing[WEEK_AS_INT_COL] = pd.to_numeric(df_listing[WEEK_AS_INT_COL], errors='coerce')
        df_listing.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
        if not df_listing.empty:
             df_listing[WEEK_AS_INT_COL] = df_listing[WEEK_AS_INT_COL].astype('Int64')
             df_listing = df_listing[
                 (df_listing[WEEK_AS_INT_COL] >= start_week) &
                 (df_listing[WEEK_AS_INT_COL] <= end_week)
             ]

    # Check if data remains after filtering
    if df_listing.empty:
        st.warning(f"No data available for '{listing}' with the selected filters.")
        return go.Figure().update_layout(title_text=f"No data for '{listing}' with filters")

    # Determine display currency and symbol
    display_currency = "Currency"
    currency_symbol = ""
    if ORIGINAL_CURRENCY_COL in df_listing.columns and not df_listing[ORIGINAL_CURRENCY_COL].dropna().empty:
         unique_currencies = df_listing[ORIGINAL_CURRENCY_COL].dropna().unique()
         if len(unique_currencies) > 0:
             display_currency = unique_currencies[0] # Use the first found currency
             currency_map = {"GBP": "£", "USD": "$", "EUR": "€"}
             currency_symbol = currency_map.get(display_currency, "") # Get symbol or default to empty
             if len(unique_currencies) > 1:
                  st.info(f"Note: Multiple transaction currencies found ({unique_currencies}) for '{listing}'. Displaying average price based on '{SALES_VALUE_TRANS_CURRENCY_COL}' in {display_currency}.")

    # Data cleaning and preparation for aggregation
    df_listing[DATE_COL] = pd.to_datetime(df_listing[DATE_COL], errors='coerce')
    df_listing[SALES_VALUE_TRANS_CURRENCY_COL] = pd.to_numeric(df_listing[SALES_VALUE_TRANS_CURRENCY_COL], errors='coerce')
    df_listing[ORDER_QTY_COL_RAW] = pd.to_numeric(df_listing[ORDER_QTY_COL_RAW], errors='coerce')

    # Drop rows with NaNs in essential columns for calculation, and where quantity is zero or less
    df_listing.dropna(subset=[DATE_COL, SALES_VALUE_TRANS_CURRENCY_COL, ORDER_QTY_COL_RAW, year_col_to_use], inplace=True)
    df_listing = df_listing[df_listing[ORDER_QTY_COL_RAW] > 0]

    if df_listing.empty:
        st.warning(f"No valid sales/quantity data for '{listing}' to calculate daily price after cleaning.")
        return go.Figure().update_layout(title_text=f"No valid data for '{listing}' after cleaning")

    # Group by date and year to calculate daily totals
    try:
        grouped = df_listing.groupby([df_listing[DATE_COL].dt.date, year_col_to_use]).agg(
            Total_Sales_Value=(SALES_VALUE_TRANS_CURRENCY_COL, "sum"),
            Total_Order_Quantity=(ORDER_QTY_COL_RAW, "sum")
        ).reset_index()
    except Exception as e:
        st.error(f"Error during daily price aggregation: {e}")
        return go.Figure().update_layout(title_text="Aggregation Error")


    # Rename the date column if necessary (groupby might put it in index or level_0)
    if DATE_COL not in grouped.columns and 'level_0' in grouped.columns:
         grouped = grouped.rename(columns={'level_0': DATE_COL})

    # Ensure the date column is present after potential renaming
    if DATE_COL not in grouped.columns:
         st.error("Failed to identify Date column after grouping.")
         return go.Figure().update_layout(title_text="Error processing grouped data")

    # Calculate Average Price
    grouped["Average Price"] = grouped["Total_Sales_Value"] / grouped["Total_Order_Quantity"]
    grouped[DATE_COL] = pd.to_datetime(grouped[DATE_COL]) # Convert date back to datetime

    # Process each year separately for smoothing and filling gaps
    dfs_processed = []
    for yr in selected_years_int:
        df_year = grouped[grouped[year_col_to_use] == yr].copy()
        if df_year.empty:
            continue # Skip year if no data

        # Calculate Day of Year
        df_year["Day"] = df_year[DATE_COL].dt.dayofyear

        # Check if Day column is valid before proceeding
        if df_year["Day"].empty or df_year["Day"].isna().all():
            st.warning(f"Could not determine day of year for {yr}. Skipping this year.")
            continue

        # Reindex to fill missing days within the year's range
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())
        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))

        # Forward fill the average price to handle missing days
        df_year["Average Price"] = df_year["Average Price"].ffill()

        # Restore year column and ensure Average Price is numeric
        df_year[year_col_to_use] = yr
        df_year["Average Price"] = pd.to_numeric(df_year["Average Price"], errors='coerce')
        df_year.dropna(subset=["Average Price"], inplace=True) # Drop days where price is still NaN

        if df_year.empty: continue # Skip if no valid prices after ffill

        # --- Simple Smoothing Logic ---
        # (Limit price jumps to +/- 25% of the previous day's valid price)
        prices = df_year["Average Price"].values.copy()
        last_valid_price = None
        for i in range(len(prices)):
            current_price = prices[i]
            if pd.notna(current_price):
                if last_valid_price is not None:
                    # Apply smoothing constraints
                    if current_price < 0.75 * last_valid_price:
                        prices[i] = last_valid_price # Replace low jump with previous price
                    elif current_price > 1.25 * last_valid_price:
                         prices[i] = last_valid_price # Replace high jump with previous price
                last_valid_price = prices[i] # Update last valid price

        df_year["Smoothed Average Price"] = prices
        # --- End Smoothing ---

        # Reset index and drop any remaining NaNs in smoothed price
        df_year = df_year.reset_index()
        df_year.dropna(subset=["Smoothed Average Price"], inplace=True)

        if not df_year.empty:
            dfs_processed.append(df_year)

    # Check if any data was processed
    if not dfs_processed:
        st.warning("No data available after processing for the Daily Price chart.")
        return go.Figure().update_layout(title_text=f"No processed data for '{listing}'")

    # Combine processed data for all years
    combined = pd.concat(dfs_processed, ignore_index=True)

    if combined.empty:
        st.warning("Combined data is empty for the Daily Price chart.")
        return go.Figure().update_layout(title_text=f"Combined data empty for '{listing}'")

    # Create the final line chart
    fig = px.line(
        combined,
        x="Day", # Use Day of Year for x-axis
        y="Smoothed Average Price", # Plot the smoothed price
        color=year_col_to_use, # Color lines by year
        title=f"Daily Average Price Trend for {listing}",
        labels={"Day": "Day of Year", "Smoothed Average Price": f"Avg Price ({currency_symbol}{display_currency})", year_col_to_use: "Year"},
        color_discrete_sequence=px.colors.qualitative.Set1 # Use a distinct color sequence
    )

    # Layout adjustments
    fig.update_yaxes(rangemode="tozero") # Y-axis starts at 0
    fig.update_layout(
        margin=dict(t=50, b=50),
        legend_title_text='Year',
        hovermode='x unified' # Unified hover info across years for the same day
        )

    return fig
