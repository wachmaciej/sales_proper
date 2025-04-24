# tabs/sku_trends.py
import streamlit as st
import pandas as pd
from plotting import create_sku_line_chart
from config import (
    SKU_COL, CUSTOM_YEAR_COL, SALES_CHANNEL_COL, WEEK_AS_INT_COL, ORDER_QTY_COL_RAW,
    PRODUCT_COL # <<< Import PRODUCT_COL
)

def display_tab(df, available_years, default_years):
    """Displays the SKU Trends tab with Product filter."""
    st.markdown("### SKU Trends")

    if SKU_COL not in df.columns:
        st.error(f"The dataset does not contain a '{SKU_COL}' column. SKU Trends cannot be displayed.")
        return # Stop rendering this tab

    with st.expander("Chart Filters", expanded=True):
        # Increased columns to 5
        col1, col2, col3, col4, col5 = st.columns(5)

        # --- ADDED: Product Filter ---
        with col1:
            selected_products_sku = []
            # Check if PRODUCT_COL exists before creating the filter
            if PRODUCT_COL in df.columns:
                product_options_sku = sorted(df[PRODUCT_COL].astype(str).dropna().unique()) # Ensure string type for sorting
                selected_products_sku = st.multiselect(
                    "Product(s)",
                    options=product_options_sku,
                    default=[], # Default to none selected
                    key="sku_products_tab",
                    help="Filter by Product(s) first (optional)."
                )
            else:
                st.caption(f"{PRODUCT_COL} filter unavailable")
        # --- END ADDED ---

        with col2: # Was col1
            # Filter available SKUs based on selected products if any
            sku_options_df = df.copy()
            # Apply product filter only if PRODUCT_COL exists and products are selected
            if selected_products_sku and PRODUCT_COL in sku_options_df.columns:
                 sku_options_df = sku_options_df[sku_options_df[PRODUCT_COL].isin(selected_products_sku)]

            # SKU Text Input
            sku_text = st.text_input(
                "Filter by SKU (Optional)",
                value="",
                key="sku_input",
                help="Enter a specific SKU (or part of it) to filter further within selected Products."
             )


        with col3: # Was col2
            sku_years = st.multiselect("Select Year(s)", options=available_years, default=default_years, key="sku_years", help="Default includes current and previous custom week year.")
        with col4: # Was col3
            sku_channels = []
            if SALES_CHANNEL_COL in df.columns:
                sku_channel_opts = sorted(df[SALES_CHANNEL_COL].astype(str).dropna().unique()) # Ensure string type
                sku_channels = st.multiselect("Select Sales Channel(s)", options=sku_channel_opts, default=[], key="sku_channels", help="Select one or more sales channels to filter SKU trends. If empty, all channels are shown.")
            else:
                st.caption(f"{SALES_CHANNEL_COL} filter unavailable")
        with col5: # Was col4
            week_range_sku = st.slider("Select Week Range", min_value=1, max_value=53, value=(1, 53), step=1, key="sku_week_range", help="Select the range of weeks to display.")

    # --- Display Chart ---
    # Require either a Product selection OR an SKU text input
    if not selected_products_sku and (not sku_text or sku_text.strip() == ""):
        st.info("Please select at least one Product OR enter a Product SKU in the filters above to view trends.")
    elif not sku_years:
        st.warning("Please select at least one year in the filters to view SKU trends.")
    else:
        # --- MODIFIED: Pass selected_products_sku to the plotting function ---
        fig_sku = create_sku_line_chart(
            df,
            sku_text, # Pass the text input value
            sku_years,
            selected_channels=sku_channels,
            week_range=week_range_sku,
            selected_products=selected_products_sku # Pass selected products
            )
        # --- END MODIFIED ---
        if fig_sku is not None:
            st.plotly_chart(fig_sku, use_container_width=True)

        # --- Display Summary Tables ---
        # Filter data again for summary, including the new product filter
        filtered_sku_data = df.copy()

        # Apply Product Filter first (if applicable)
        if selected_products_sku and PRODUCT_COL in filtered_sku_data.columns:
            filtered_sku_data = filtered_sku_data[filtered_sku_data[PRODUCT_COL].isin(selected_products_sku)]

        # Apply SKU text filter (if provided) on the already product-filtered data
        if sku_text and sku_text.strip() != "" and SKU_COL in filtered_sku_data.columns:
             filtered_sku_data[SKU_COL] = filtered_sku_data[SKU_COL].astype(str)
             filtered_sku_data = filtered_sku_data[filtered_sku_data[SKU_COL].str.contains(sku_text, case=False, na=False)]
        # If no SKU text, use all SKUs from the selected products (or all products if none selected)

        # Apply other filters
        if sku_years:
            # Ensure CUSTOM_YEAR_COL exists before filtering
            if CUSTOM_YEAR_COL in filtered_sku_data.columns:
                filtered_sku_data = filtered_sku_data[filtered_sku_data[CUSTOM_YEAR_COL].isin(sku_years)]
            else:
                 st.warning(f"Column '{CUSTOM_YEAR_COL}' not found for year filtering in summary.")
                 filtered_sku_data = pd.DataFrame() # Clear data if year column missing

        if sku_channels and not filtered_sku_data.empty: # Check if df not already empty
            if SALES_CHANNEL_COL in filtered_sku_data.columns:
                 filtered_sku_data = filtered_sku_data[filtered_sku_data[SALES_CHANNEL_COL].isin(sku_channels)]
            # else: Do nothing if column missing, warning was shown in filter section

        if week_range_sku and not filtered_sku_data.empty: # Check if df not already empty
            start_w, end_w = week_range_sku
            if WEEK_AS_INT_COL in filtered_sku_data.columns:
                filtered_sku_data[WEEK_AS_INT_COL] = pd.to_numeric(filtered_sku_data[WEEK_AS_INT_COL], errors='coerce').astype('Int64')
                filtered_sku_data.dropna(subset=[WEEK_AS_INT_COL], inplace=True)
                if not filtered_sku_data.empty:
                     filtered_sku_data = filtered_sku_data[(filtered_sku_data[WEEK_AS_INT_COL] >= start_w) & (filtered_sku_data[WEEK_AS_INT_COL] <= end_w)]
            else: st.warning(f"Week column '{WEEK_AS_INT_COL}' missing for week range filter in SKU summary.")

        # Check for Order Quantity column for units summary
        if ORDER_QTY_COL_RAW in filtered_sku_data.columns and not filtered_sku_data.empty:
            filtered_sku_data[ORDER_QTY_COL_RAW] = pd.to_numeric(filtered_sku_data[ORDER_QTY_COL_RAW], errors='coerce')
            # Drop rows missing essential info for unit summary
            # Ensure CUSTOM_YEAR_COL and SKU_COL exist before dropping NA based on them
            required_agg_cols = [ORDER_QTY_COL_RAW]
            if CUSTOM_YEAR_COL in filtered_sku_data.columns: required_agg_cols.append(CUSTOM_YEAR_COL)
            if SKU_COL in filtered_sku_data.columns: required_agg_cols.append(SKU_COL)
            # Only drop if all required agg cols are present
            if all(col in filtered_sku_data.columns for col in required_agg_cols):
                filtered_sku_data.dropna(subset=required_agg_cols, inplace=True)


            if not filtered_sku_data.empty and CUSTOM_YEAR_COL in filtered_sku_data.columns:
                # Total Units Summary (summing across all matching SKUs/Products)
                total_units = filtered_sku_data.groupby(CUSTOM_YEAR_COL)[ORDER_QTY_COL_RAW].sum().reset_index()
                if not total_units.empty:
                    total_units_indexed = total_units.set_index(CUSTOM_YEAR_COL)
                    total_units_summary = total_units_indexed[[ORDER_QTY_COL_RAW]].T
                    summary_label = "Total Units (Selected SKUs/Products)" # Default label
                    if sku_text and sku_text.strip() != "":
                        summary_label = f"Total Units ({sku_text})"
                        if selected_products_sku: # Add context if both filters used
                             summary_label += " within Selected Products"
                    elif selected_products_sku:
                         summary_label = f"Total Units (Selected Products)"

                    total_units_summary.index = [summary_label]
                    st.markdown("##### Total Units Sold Summary")
                    st.dataframe(total_units_summary.fillna(0).astype(int).style.format("{:,}"), use_container_width=True)

                # SKU Breakdown Table
                # Group by Product first (if available), then SKU, then Year
                grouping_cols_sku = []
                if PRODUCT_COL in filtered_sku_data.columns: grouping_cols_sku.append(PRODUCT_COL)
                if SKU_COL in filtered_sku_data.columns: grouping_cols_sku.append(SKU_COL)
                if CUSTOM_YEAR_COL in filtered_sku_data.columns: grouping_cols_sku.append(CUSTOM_YEAR_COL)

                # Proceed only if essential grouping columns exist
                if SKU_COL in grouping_cols_sku and CUSTOM_YEAR_COL in grouping_cols_sku:
                    sku_units = filtered_sku_data.groupby(grouping_cols_sku)[ORDER_QTY_COL_RAW].sum().reset_index()

                    if not sku_units.empty:
                        pivot_index_cols = [PRODUCT_COL, SKU_COL] if PRODUCT_COL in grouping_cols_sku else [SKU_COL]
                        try:
                             sku_pivot = sku_units.pivot(
                                 index=pivot_index_cols,
                                 columns=CUSTOM_YEAR_COL,
                                 values=ORDER_QTY_COL_RAW
                             )
                             sku_pivot = sku_pivot.fillna(0).astype(int)
                             st.markdown("##### SKU Breakdown (Units Sold by Custom Week Year)")
                             st.dataframe(sku_pivot.style.format("{:,}"), use_container_width=True)
                        except Exception as e:
                             st.warning(f"Could not create SKU breakdown pivot table: {e}")
                             # Show flat summary if pivot fails
                             st.dataframe(sku_units) # Display the non-pivoted summary
                else:
                     st.caption("Cannot group data for SKU breakdown (missing SKU or Year column).")


            # else: # Optional message if no data after cleaning
            #     st.info("No valid 'Order Quantity' data found for the selected filters.")
        elif filtered_sku_data.empty:
             # Message if filters resulted in no data before checking quantity col
             st.info("No data matches the selected filters for the summary tables.")
        else: # Quantity column missing
            st.info(f"Column '{ORDER_QTY_COL_RAW}' not found, cannot show units sold summary.")

