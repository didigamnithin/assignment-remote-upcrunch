import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Enhanced UI imports for colourful dashboard
try:
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.badges import badge
    from streamlit_option_menu import option_menu
    from streamlit_card import card
    from streamlit_lottie import st_lottie
except Exception:  # pragma: no cover
    style_metric_cards = None
    colored_header = None
    badge = None
    option_menu = None
    card = None
    st_lottie = None

# Guard optional plotting libs to avoid hard failures
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

try:
    import streamlit as st
except Exception as e:  # pragma: no cover
    raise SystemExit("Streamlit must be installed to run this app: pip install streamlit plotly pandas numpy") from e

# Optional analytics dependencies
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except Exception:  # pragma: no cover
    seasonal_decompose = None

try:
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover
    Prophet = None

try:
    from sklearn.linear_model import Ridge  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
except Exception:  # pragma: no cover
    Ridge = None
    train_test_split = None


# -------------------------------
# Data loading and utilities
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_FILES = {
    "marketing": BASE_DIR / "upchurch doordash marketing.csv",
    "operations": BASE_DIR / "upchurch doordash operations.csv",
    "payouts": BASE_DIR / "upchurch doordash payouts.csv",
    "sales": BASE_DIR / "upchurch doordash sales.csv",
    "ubereats": BASE_DIR / "upchurch ubereats sales and payouts.csv",
}


def _safe_lower(text: str) -> str:
    return text.lower().strip()


def find_column_by_keywords(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Return the first dataframe column whose lowercase name contains any of the keywords (lowercased).
    Useful when exact column names may vary slightly across exports.
    """
    lowered = {col: _safe_lower(col) for col in df.columns}
    for col, low in lowered.items():
        for kw in keywords:
            if _safe_lower(kw) in low:
                return col
    return None


def safe_divide(numerator, denominator):
    """Robust element-wise divide that tolerates zeros/NaNs and scalar denominators.

    Rules:
    - If denominator is 0 or NaN → return numerator
    - Otherwise → return numerator / denominator
    Works for scalars, Series, and DataFrames without shape-mismatch errors.
    """
    # Normalize inputs to pandas objects when possible for consistent behavior
    num = numerator
    den = denominator
    if not isinstance(num, (pd.Series, pd.DataFrame)):
        num = pd.to_numeric(num, errors="coerce")
    if not isinstance(den, (pd.Series, pd.DataFrame)):
        den = pd.to_numeric(den, errors="coerce")

    # Fast-path: scalar denominator (common in min-max normalization)
    if np.isscalar(den) or (isinstance(den, (pd.Series, pd.DataFrame)) and den.size == 1):
        den_scalar = den if np.isscalar(den) else (den.values.item() if hasattr(den, "values") else float(den))
        if pd.isna(den_scalar) or den_scalar == 0:
            return num
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / den_scalar

    # General case: align shapes where possible
    aligned_den = den
    if isinstance(num, pd.Series) and isinstance(den, pd.Series):
        aligned_den = den.reindex_like(num)
    elif isinstance(num, pd.DataFrame) and isinstance(den, pd.DataFrame):
        aligned_den = den.reindex_like(num)

    mask = (aligned_den == 0) | pd.isna(aligned_den)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / aligned_den

    # Where denominator invalid, fall back to numerator
    if isinstance(result, (pd.Series, pd.DataFrame)):
        result = result.mask(mask, num)
        result = result.replace([np.inf, -np.inf], np.nan).mask(mask, num)
    else:
        if bool(np.any(mask)):
            result = num
        if np.isinf(result):
            result = num
    return result


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback for potential encoding issues
        return pd.read_csv(path, encoding="latin-1")


def parse_datetime_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name in df.columns:
        df = df.copy()
        # Try to parse with specific format first, then fallback to auto-detection
        try:
            # Try ISO format first (YYYY-MM-DD)
            df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d', errors="coerce")
        except:
            try:
                # Try MM/DD/YY format
                df[col_name] = pd.to_datetime(df[col_name], format='%m/%d/%y', errors="coerce")
            except:
                # Fallback to auto-detection
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
    return df


def add_week_start(df: pd.DataFrame, date_col: str, new_col: str = "Week") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df = df.copy()
    series = pd.to_datetime(df[date_col], errors="coerce")
    df[new_col] = series.dt.to_period("W-MON").apply(lambda r: r.start_time if pd.notna(r) else pd.NaT)
    return df


def filter_df_by_date(df: pd.DataFrame, date_col: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    mask = pd.Series(True, index=df.index)
    # Use the already parsed datetime column if available, otherwise parse it
    if df[date_col].dtype == 'datetime64[ns]':
        dates = df[date_col]
    else:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return df.loc[mask]


def make_metric_card(label: str, value, delta: Optional[str] = None, help_text: Optional[str] = None):
    col = st.container()
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_text)


# -------------------------------
# Section: Marketing
# -------------------------------

def section_marketing(marketing_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="📢 Marketing Analytics",
            description="Store-level campaign performance, customer acquisition, and ROI analysis",
            color_name="green-70"
        )
    else:
        st.markdown("## 📢 Marketing Analytics")
        st.markdown("*Store-level campaign performance, customer acquisition, and ROI analysis*")
    
    if marketing_df.empty:
        st.info("📭 Marketing CSV not found or empty.")
        return

    # Parse and prepare
    marketing_df = parse_datetime_column(marketing_df, "Date")
    
    # Debug information for zero values issue
    st.info(f"📊 Marketing Data Summary: {len(marketing_df)} rows loaded")
    if "Date" in marketing_df.columns:
        date_range = f"Date range: {marketing_df['Date'].min()} to {marketing_df['Date'].max()}"
        st.info(date_range)
        st.info("📅 Note: Data filtered to end on August 10, 2025 as requested")
    
    # Check for zero values in key metrics
    key_metrics = ["Orders", "Sales", "Marketing Fees | (Including any applicable taxes)", "ROAS"]
    zero_counts = {}
    for metric in key_metrics:
        if metric in marketing_df.columns:
            zero_count = (marketing_df[metric] == 0).sum()
            total_count = len(marketing_df)
            zero_counts[metric] = f"{zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)"
    
    if zero_counts:
        st.info("🔍 Zero Values Analysis:")
        for metric, count in zero_counts.items():
            st.info(f"  {metric}: {count} zero values")

    # Store-Level Analysis Section
    st.markdown("## 🏪 Store-Level Marketing Analysis")
    
    # Get unique stores from marketing data
    if "Store Name" in marketing_df.columns:
        stores = sorted(marketing_df["Store Name"].dropna().unique().tolist())
        
        # Store selection interface
        st.markdown("**Select Store for Analysis:**")
        
        # Create store selection buttons in a grid
        store_cols = st.columns(4)
        
        # Initialize session state for store selection
        if "marketing_selected_store" not in st.session_state:
            st.session_state.marketing_selected_store = stores[0] if stores else None
        
        for i, store in enumerate(stores):
            col_idx = i % 4
            with store_cols[col_idx]:
                if st.button(store, key=f"marketing_store_{i}", use_container_width=True):
                    st.session_state.marketing_selected_store = store
        
        # Store-specific analysis
        if st.session_state.marketing_selected_store:
            st.markdown(f"## 📊 {st.session_state.marketing_selected_store} - Marketing Performance Analysis")
            
            # Filter data for selected store
            store_data = marketing_df[marketing_df["Store Name"] == st.session_state.marketing_selected_store].copy()
            
            if not store_data.empty:
                # Store-specific KPIs
                st.markdown("### 🎯 Store Performance Metrics")
                store_kpi_cols = st.columns(4)
                
                with store_kpi_cols[0]:
                    store_orders = store_data.get("Orders", pd.Series(dtype=float)).sum()
                    st.metric("Total Orders", f"{int(store_orders):,}")
                
                with store_kpi_cols[1]:
                    store_sales = store_data.get("Sales", pd.Series(dtype=float)).sum()
                    st.metric("Total Sales", f"${store_sales:,.0f}")
                
                with store_kpi_cols[2]:
                    store_aov = store_data.get("Average Order Value", pd.Series(dtype=float)).mean()
                    if pd.notnull(store_aov):
                        st.metric("Average Order Value", f"${store_aov:,.2f}")
                    else:
                        st.metric("Average Order Value", "—")
                
                with store_kpi_cols[3]:
                    store_roas = store_data.get("ROAS", pd.Series(dtype=float)).mean()
                    if pd.notnull(store_roas):
                        st.metric("Average ROAS", f"{store_roas:.2f}x")
                    else:
                        st.metric("Average ROAS", "—")
                
                # Customer acquisition metrics
                st.markdown("### 👥 Customer Acquisition Analysis")
                customer_cols = st.columns(3)
                
                with customer_cols[0]:
                    new_customers = store_data.get("New Customers Acquired", pd.Series(dtype=float)).sum()
                    st.metric("New Customers", f"{int(new_customers):,}")
                
                with customer_cols[1]:
                    total_customers = store_data.get("Total Customers Acquired", pd.Series(dtype=float)).sum()
                    st.metric("Total Customers", f"{int(total_customers):,}")
                
                with customer_cols[2]:
                    if total_customers > 0:
                        repeat_customers = total_customers - new_customers
                        st.metric("Repeat Customers", f"{int(repeat_customers):,}")
                    else:
                        st.metric("Repeat Customers", "—")
                
                # Campaign-level analysis for the store
                st.markdown("### 📢 Campaign Performance by Store")
                
                # Get campaign columns
                campaign_cols = [col for col in store_data.columns if 'campaign' in col.lower() or 'promotion' in col.lower()]
                
                if campaign_cols:
                    # Campaign summary table
                    campaign_summary = []
                    for campaign_col in campaign_cols:
                        campaign_data = store_data.groupby(campaign_col).agg({
                            "Orders": "sum",
                            "Sales": "sum",
                            "New Customers Acquired": "sum",
                            "Total Customers Acquired": "sum"
                        }).reset_index()
                        
                        for _, row in campaign_data.iterrows():
                            campaign_summary.append({
                                "Campaign": row[campaign_col],
                                "Orders": row["Orders"],
                                "Sales": row["Sales"],
                                "New Customers": row["New Customers Acquired"],
                                "Total Customers": row["Total Customers Acquired"]
                            })
                    
                    if campaign_summary:
                        campaign_df = pd.DataFrame(campaign_summary)
                        st.dataframe(campaign_df, use_container_width=True)
                        
                        # Campaign performance charts
                        if px is not None:
                            # Orders by campaign
                            fig_orders = px.bar(campaign_df, x="Campaign", y="Orders", 
                                              title=f"{st.session_state.marketing_selected_store} - Orders by Campaign",
                                              color="Orders", color_continuous_scale="viridis")
                            st.plotly_chart(fig_orders, use_container_width=True)
                            
                            # Sales by campaign
                            fig_sales = px.bar(campaign_df, x="Campaign", y="Sales", 
                                             title=f"{st.session_state.marketing_selected_store} - Sales by Campaign",
                                             color="Sales", color_continuous_scale="plasma")
                            st.plotly_chart(fig_sales, use_container_width=True)
                else:
                    st.info("No campaign-specific columns found in the data.")
                
                # Time series analysis for the store
                st.markdown("### 📈 Store Performance Over Time")
                if "Date" in store_data.columns:
                    store_data = parse_datetime_column(store_data, "Date")
                    store_data = store_data.sort_values("Date")
                    
                    # Daily trends for the store
                    time_series_cols = [c for c in ["Orders", "Sales", "Average Order Value", "ROAS", "New Customers Acquired"] 
                                      if c in store_data.columns]
                    
                    if time_series_cols:
                        for col in time_series_cols:
                            fig = px.line(store_data, x="Date", y=col, 
                                        title=f"{st.session_state.marketing_selected_store} - {col} Over Time",
                                        markers=True)
                            fig.update_layout(xaxis_title="Date", yaxis_title=col)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Store comparison with other stores
                st.markdown("### 🔍 Store Performance Comparison")
                
                # Calculate store rankings
                all_stores_summary = marketing_df.groupby("Store Name").agg({
                    "Orders": "sum",
                    "Sales": "sum",
                    "ROAS": "mean",
                    "New Customers Acquired": "sum"
                }).reset_index()
                
                # Only show ranking if a specific store is selected (not "All Stores")
                if st.session_state.marketing_selected_store != "All Stores":
                    # Find current store's rank
                    store_rank = all_stores_summary[all_stores_summary["Store Name"] == st.session_state.marketing_selected_store].index[0] + 1
                    total_stores = len(all_stores_summary)
                    
                    # Display ranking metrics
                    rank_cols = st.columns(4)
                    with rank_cols[0]:
                        st.metric("Store Rank", f"#{store_rank} of {total_stores}")
                    
                    with rank_cols[1]:
                        store_percentile = (store_rank / total_stores) * 100
                        st.metric("Percentile", f"{store_percentile:.1f}%")
                    
                    with rank_cols[2]:
                        if store_rank > 1:
                            next_store = all_stores_summary.iloc[store_rank - 2]
                            gap_orders = store_orders - next_store["Orders"]
                            st.metric("Orders Gap to Next", f"{int(gap_orders):,}")
                        else:
                            st.metric("Orders Gap to Next", "🏆 Top Store")
                    
                    with rank_cols[3]:
                        if store_rank > 1:
                            gap_sales = store_sales - next_store["Sales"]
                            st.metric("Sales Gap to Next", f"${gap_sales:,.0f}")
                        else:
                            st.metric("Sales Gap to Next", "🏆 Top Store")
                    
                    # Top 5 stores comparison
                    st.markdown("**🏆 Top 5 Stores by Orders**")
                    top_stores = all_stores_summary.nlargest(5, "Orders")
                    fig_top = px.bar(top_stores, x="Store Name", y="Orders", 
                                   title="Top 5 Stores - Orders Comparison",
                                   color="Orders", color_continuous_scale="viridis")
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Top 5 stores by sales
                    st.markdown("**💰 Top 5 Stores by Sales**")
                    top_sales = all_stores_summary.nlargest(5, "Sales")
                    fig_sales_top = px.bar(top_sales, x="Store Name", y="Sales", 
                                         title="Top 5 Stores - Sales Comparison",
                                         color="Sales", color_continuous_scale="plasma")
                    st.plotly_chart(fig_sales_top, use_container_width=True)
        
    else:
        st.info("Store Name column not found in marketing data. Store-level analysis unavailable.")


# -------------------------------
# Section: Operations
# -------------------------------

def section_operations(ops_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="⚙️ Operations Analytics",
            description="Store-level performance, ratings, and operational efficiency metrics",
            color_name="blue-70"
        )
    else:
        st.markdown("## ⚙️ Operations Analytics")
        st.markdown("*Store-level performance, ratings, and operational efficiency metrics*")
    
    if ops_df.empty:
        st.info("📭 Operations CSV not found or empty.")
        return

    ops_df = parse_datetime_column(ops_df, "Start Date")
    ops_df = parse_datetime_column(ops_df, "End Date")
    ops_df = add_week_start(ops_df, "Start Date", new_col="Week")

    # KPIs and derived metrics
    kpis = [
        "Total Orders Including Cancelled Orders",
        "Total Delivered or Picked Up Orders",
        "Total Missing or Incorrect Orders",
        "Total Error Charges",
        "Total Cancelled Orders",
        "Total Downtime in Minutes",
        "Average Rating",
    ]

    # Derived
    ops_df = ops_df.copy()
    if (
        "Total Cancelled Orders" in ops_df.columns and
        "Total Orders Including Cancelled Orders" in ops_df.columns
    ):
        ops_df["Cancellation Rate"] = safe_divide(
            ops_df["Total Cancelled Orders"], ops_df["Total Orders Including Cancelled Orders"]
        )
    if (
        "Total Downtime in Minutes" in ops_df.columns and
        "Total Orders Including Cancelled Orders" in ops_df.columns
    ):
        ops_df["Downtime per Order (min)"] = safe_divide(
            ops_df["Total Downtime in Minutes"], ops_df["Total Orders Including Cancelled Orders"]
        )

    # Show data summary
    st.info(f"📊 Operations Data Summary: {len(ops_df)} rows loaded")
    if "Start Date" in ops_df.columns:
        date_range = f"Date range: {ops_df['Start Date'].min()} to {ops_df['Start Date'].max()}"
        st.info(date_range)
        st.info("📅 Note: Data filtered to end on August 10, 2025 as requested")

    # Store-Level Analysis Section
    st.markdown("## 🏪 Store-Level Operations Analysis")
    
    # Get unique stores from operations data
    if "Store Name" in ops_df.columns:
        stores = sorted(ops_df["Store Name"].dropna().unique().tolist())
        
        # Store selection interface
        st.markdown("**Select Store for Analysis:**")
        
        # Create store selection buttons in a grid
        store_cols = st.columns(4)
        
        # Initialize session state for store selection
        if "ops_selected_store" not in st.session_state:
            st.session_state.ops_selected_store = stores[0] if stores else None
        
        for i, store in enumerate(stores):
            col_idx = i % 4
            with store_cols[col_idx]:
                if st.button(store, key=f"ops_store_{i}", use_container_width=True):
                    st.session_state.ops_selected_store = store
        
        # Store-specific analysis
        if st.session_state.ops_selected_store:
            st.markdown(f"## 📊 {st.session_state.ops_selected_store} - Operations Performance Analysis")
            
            # Filter data for selected store
            store_data = ops_df[ops_df["Store Name"] == st.session_state.ops_selected_store].copy()
            
            if not store_data.empty:
                # Store-specific KPIs
                st.markdown("### 🎯 Store Operations Metrics")
                store_kpi_cols = st.columns(4)
                
                with store_kpi_cols[0]:
                    store_delivered = store_data.get("Total Delivered or Picked Up Orders", pd.Series(dtype=float)).sum()
                    st.metric("Total Delivered Orders", f"{int(store_delivered):,}")
                
                with store_kpi_cols[1]:
                    store_cancelled = store_data.get("Total Cancelled Orders", pd.Series(dtype=float)).sum()
                    st.metric("Total Cancelled Orders", f"{int(store_cancelled):,}")
                
                with store_kpi_cols[2]:
                    store_rating = store_data.get("Average Rating", pd.Series(dtype=float)).mean()
                    if pd.notnull(store_rating):
                        st.metric("Average Rating", f"{store_rating:.2f}")
                    else:
                        st.metric("Average Rating", "—")
                
                with store_kpi_cols[3]:
                    store_downtime = store_data.get("Total Downtime in Minutes", pd.Series(dtype=float)).sum()
                    st.metric("Total Downtime", f"{int(store_downtime)} min")
                
                # Operational efficiency metrics
                st.markdown("### ⚡ Operational Efficiency")
                efficiency_cols = st.columns(3)
                
                with efficiency_cols[0]:
                    if "Total Orders Including Cancelled Orders" in store_data.columns:
                        total_orders = store_data["Total Orders Including Cancelled Orders"].sum()
                        fulfillment_rate = (store_delivered / total_orders) * 100 if total_orders > 0 else 0
                        st.metric("Fulfillment Rate", f"{fulfillment_rate:.1f}%")
                    else:
                        st.metric("Fulfillment Rate", "—")
                
                with efficiency_cols[1]:
                    if "Total Orders Including Cancelled Orders" in store_data.columns:
                        cancellation_rate = (store_cancelled / total_orders) * 100 if total_orders > 0 else 0
                        st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
                    else:
                        st.metric("Cancellation Rate", "—")
                
                with efficiency_cols[2]:
                    if total_orders > 0:
                        downtime_per_order = store_downtime / total_orders
                        st.metric("Downtime per Order", f"{downtime_per_order:.1f} min")
                    else:
                        st.metric("Downtime per Order", "—")
                
                # Time series analysis for the store
                st.markdown("### 📈 Store Performance Over Time")
                if "Week" in store_data.columns:
                    # Weekly trends for the store
                    time_series_cols = [c for c in ["Total Delivered or Picked Up Orders", "Total Cancelled Orders", 
                                                   "Average Rating", "Total Downtime in Minutes"] 
                                      if c in store_data.columns]
                    
                    if time_series_cols:
                        for col in time_series_cols:
                            weekly_data = store_data.groupby("Week")[col].mean().reset_index()
                            fig = px.line(weekly_data, x="Week", y=col, 
                                        title=f"{st.session_state.ops_selected_store} - {col} Over Time",
                                        markers=True)
                            fig.update_layout(xaxis_title="Week", yaxis_title=col)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Store comparison with other stores
                st.markdown("### 🔍 Store Performance Comparison")
                
                # Calculate store rankings
                all_stores_summary = ops_df.groupby("Store Name").agg({
                    "Total Delivered or Picked Up Orders": "sum",
                    "Total Cancelled Orders": "sum",
                    "Average Rating": "mean",
                    "Total Downtime in Minutes": "sum"
                }).reset_index()
                
                # Only show ranking if a specific store is selected (not "All Stores")
                if st.session_state.ops_selected_store != "All Stores":
                    # Find current store's rank
                    store_rank = all_stores_summary[all_stores_summary["Store Name"] == st.session_state.ops_selected_store].index[0] + 1
                    total_stores = len(all_stores_summary)
                    
                    # Display ranking metrics
                    rank_cols = st.columns(4)
                    with rank_cols[0]:
                        st.metric("Store Rank", f"#{store_rank} of {total_stores}")
                    
                    with rank_cols[1]:
                        store_percentile = (store_rank / total_stores) * 100
                        st.metric("Percentile", f"{store_percentile:.1f}%")
                    
                    with rank_cols[2]:
                        if store_rank > 1:
                            next_store = all_stores_summary.iloc[store_rank - 2]
                            gap_orders = store_delivered - next_store["Total Delivered or Picked Up Orders"]
                            st.metric("Orders Gap to Next", f"{int(gap_orders):,}")
                        else:
                            st.metric("Orders Gap to Next", "🏆 Top Store")
                    
                    with rank_cols[3]:
                        if store_rank > 1:
                            gap_rating = store_rating - next_store["Average Rating"]
                            st.metric("Rating Gap to Next", f"{gap_rating:+.2f}")
                        else:
                            st.metric("Rating Gap to Next", "🏆 Top Store")
                    
                    # Top 5 stores comparison
                    st.markdown("**🏆 Top 5 Stores by Delivered Orders**")
                    top_stores = all_stores_summary.nlargest(5, "Total Delivered or Picked Up Orders")
                    fig_top = px.bar(top_stores, x="Store Name", y="Total Delivered or Picked Up Orders", 
                                   title="Top 5 Stores - Delivered Orders Comparison",
                                   color="Total Delivered or Picked Up Orders", color_continuous_scale="viridis")
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Top 5 stores by rating
                    st.markdown("**⭐ Top 5 Stores by Average Rating**")
                    top_rating = all_stores_summary.nlargest(5, "Average Rating")
                    fig_rating = px.bar(top_rating, x="Store Name", y="Average Rating", 
                                      title="Top 5 Stores - Average Rating Comparison",
                                      color="Average Rating", color_continuous_scale="plasma")
                    st.plotly_chart(fig_rating, use_container_width=True)
                
            else:
                st.warning(f"No data found for store: {st.session_state.ops_selected_store}")
        else:
            # All stores overview
            st.markdown("### 📊 All Stores Overview")
            
            if "Store Name" in ops_df.columns:
                # Store summary table
                store_summary = ops_df.groupby("Store Name").agg({
                    "Total Delivered or Picked Up Orders": "sum",
                    "Total Cancelled Orders": "sum",
                    "Average Rating": "mean",
                    "Total Downtime in Minutes": "sum"
                }).reset_index()
                
                st.dataframe(store_summary, use_container_width=True)
                
                # Store performance heatmap
                if px is not None:
                    store_metrics = store_summary.set_index("Store Name")[["Total Delivered or Picked Up Orders", 
                                                                        "Total Cancelled Orders", "Average Rating", 
                                                                        "Total Downtime in Minutes"]]
                    fig_heatmap = px.imshow(store_metrics.T, 
                                          title="Store Operations Performance Heatmap",
                                          color_continuous_scale="viridis",
                                          aspect="auto")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Store Name column not found in operations data. Store-level analysis unavailable.")


# -------------------------------
# Section: Sales
# -------------------------------

def section_sales(sales_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="💰 Sales Analytics",
            description="Store-level revenue analysis, order trends, and financial performance",
            color_name="orange-70"
        )
    else:
        st.markdown("## 💰 Sales Analytics")
        st.markdown("*Store-level revenue analysis, order trends, and financial performance*")
    
    if sales_df.empty:
        st.info("📭 Sales CSV not found or empty.")
        return

    sales_df = parse_datetime_column(sales_df, "Start Date")
    sales_df = parse_datetime_column(sales_df, "End Date")
    sales_df = add_week_start(sales_df, "Start Date", new_col="Week")

    # Derived metrics mirroring the notebook
    def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
        return safe_divide(numer, denom)

    if {
        "Total Orders Including Cancelled Orders",
        "Total Delivered or Picked Up Orders",
        "Gross Sales",
    }.issubset(sales_df.columns):
        sales_df = sales_df.assign(
            Cancellation_Rate=safe_ratio(
                sales_df["Total Orders Including Cancelled Orders"] - sales_df["Total Delivered or Picked Up Orders"],
                sales_df["Total Orders Including Cancelled Orders"],
            ),
            Fulfillment_Rate=safe_ratio(
                sales_df["Total Delivered or Picked Up Orders"],
                sales_df["Total Orders Including Cancelled Orders"],
            ),
            Commission_Rate=safe_ratio(sales_df.get("Total Commission"), sales_df.get("Gross Sales")),
            Promo_ROI=safe_ratio(
                sales_df.get("Total Promotion Sales | (for historical reference only)"),
                sales_df.get("Total Promotion Fees | (for historical reference only)"),
            ),
            Ad_ROI=safe_ratio(
                sales_df.get("Total Ad Sales | (for historical reference only)"),
                sales_df.get("Total Ad Fees | (for historical reference only)"),
            ),
            Revenue_per_Delivered=safe_ratio(
                sales_df.get("Gross Sales"), sales_df.get("Total Delivered or Picked Up Orders")
            ),
        )

    # Show data summary
    st.info(f"📊 Sales Data Summary: {len(sales_df)} rows loaded")
    if "Start Date" in sales_df.columns:
        date_range = f"Date range: {sales_df['Start Date'].min()} to {sales_df['Start Date'].max()}"
        st.info(date_range)
        st.info("📅 Note: Data filtered to end on August 10, 2025 as requested")

    # Store-Level Analysis Section
    st.markdown("## 🏪 Store-Level Sales Analysis")
    
    # Get unique stores from sales data
    if "Store Name" in sales_df.columns:
        stores = sorted(sales_df["Store Name"].dropna().unique().tolist())
        
        # Store selection interface
        st.markdown("**Select Store for Analysis:**")
        
        # Create store selection buttons in a grid
        store_cols = st.columns(4)
        
        # Initialize session state for store selection
        if "sales_selected_store" not in st.session_state:
            st.session_state.sales_selected_store = stores[0] if stores else None
        
        for i, store in enumerate(stores):
            col_idx = i % 4
            with store_cols[col_idx]:
                if st.button(store, key=f"sales_store_{i}", use_container_width=True):
                    st.session_state.sales_selected_store = store
        
        # Store-specific analysis
        if st.session_state.sales_selected_store:
            st.markdown(f"## 📊 {st.session_state.sales_selected_store} - Sales Performance Analysis")
            
            # Filter data for selected store
            store_data = sales_df[sales_df["Store Name"] == st.session_state.sales_selected_store].copy()
            
            if not store_data.empty:
                # Store-specific KPIs
                st.markdown("### 🎯 Store Sales Metrics")
                store_kpi_cols = st.columns(4)
                
                with store_kpi_cols[0]:
                    store_gross_sales = store_data.get("Gross Sales", pd.Series(dtype=float)).sum()
                    st.metric("Total Gross Sales", f"${store_gross_sales:,.0f}")
                
                with store_kpi_cols[1]:
                    store_orders = store_data.get("Total Delivered or Picked Up Orders", pd.Series(dtype=float)).sum()
                    st.metric("Total Delivered Orders", f"{int(store_orders):,}")
                
                with store_kpi_cols[2]:
                    store_aov = store_data.get("AOV", pd.Series(dtype=float)).mean()
                    if pd.notnull(store_aov):
                        st.metric("Average Order Value", f"${store_aov:,.2f}")
                    else:
                        st.metric("Average Order Value", "—")
                
                with store_kpi_cols[3]:
                    store_commission = store_data.get("Total Commission", pd.Series(dtype=float)).sum()
                    st.metric("Total Commission", f"${store_commission:,.0f}")
                
                # Financial performance metrics
                st.markdown("### 💰 Financial Performance")
                financial_cols = st.columns(3)
                
                with financial_cols[0]:
                    if "Total Promotion Sales | (for historical reference only)" in store_data.columns:
                        promo_sales = store_data["Total Promotion Sales | (for historical reference only)"].sum()
                        st.metric("Promotion Sales", f"${promo_sales:,.0f}")
                    else:
                        st.metric("Promotion Sales", "—")
                
                with financial_cols[1]:
                    if "Total Ad Sales | (for historical reference only)" in store_data.columns:
                        ad_sales = store_data["Total Ad Sales | (for historical reference only)"].sum()
                        st.metric("Ad Sales", f"${ad_sales:,.0f}")
                    else:
                        st.metric("Ad Sales", "—")
                
                with financial_cols[2]:
                    if "Total Promotion Fees | (for historical reference only)" in store_data.columns:
                        promo_fees = store_data["Total Promotion Fees | (for historical reference only)"].sum()
                        st.metric("Promotion Fees", f"${promo_fees:,.0f}")
                    else:
                        st.metric("Promotion Fees", "—")
                
                # Operational efficiency metrics
                st.markdown("### ⚡ Operational Efficiency")
                efficiency_cols = st.columns(3)
                
                with efficiency_cols[0]:
                    if "Fulfillment_Rate" in store_data.columns:
                        store_fulfillment = store_data["Fulfillment_Rate"].mean()
                        st.metric("Fulfillment Rate", f"{store_fulfillment*100:.1f}%")
                    else:
                        st.metric("Fulfillment Rate", "—")
                
                with efficiency_cols[1]:
                    if "Cancellation_Rate" in store_data.columns:
                        store_cancellation = store_data["Cancellation_Rate"].mean()
                        st.metric("Cancellation Rate", f"{store_cancellation*100:.1f}%")
                    else:
                        st.metric("Cancellation Rate", "—")
                
                with efficiency_cols[2]:
                    if "Commission_Rate" in store_data.columns:
                        store_commission_rate = store_data["Commission_Rate"].mean()
                        st.metric("Commission Rate", f"{store_commission_rate*100:.1f}%")
                    else:
                        st.metric("Commission Rate", "—")
                
                # Time series analysis for the store
                st.markdown("### 📈 Store Performance Over Time")
                if "Week" in store_data.columns:
                    # Weekly trends for the store
                    time_series_cols = [c for c in ["Gross Sales", "Total Delivered or Picked Up Orders", 
                                                   "AOV", "Fulfillment_Rate"] 
                                      if c in store_data.columns]
                    
                    if time_series_cols:
                        for col in time_series_cols:
                            weekly_data = store_data.groupby("Week")[col].sum().reset_index()
                            if "Rate" in col:
                                weekly_data[col] = store_data.groupby("Week")[col].mean().values
                            
                            fig = px.line(weekly_data, x="Week", y=col, 
                                        title=f"{st.session_state.sales_selected_store} - {col} Over Time",
                                        markers=True)
                            fig.update_layout(xaxis_title="Week", yaxis_title=col)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Store comparison with other stores
                st.markdown("### 🔍 Store Performance Comparison")
                
                # Calculate store rankings
                all_stores_summary = sales_df.groupby("Store Name").agg({
                    "Gross Sales": "sum",
                    "Total Delivered or Picked Up Orders": "sum",
                    "AOV": "mean",
                    "Total Commission": "sum"
                }).reset_index()
                
                # Only show ranking if a specific store is selected (not "All Stores")
                if st.session_state.sales_selected_store != "All Stores":
                    # Find current store's rank
                    store_rank = all_stores_summary[all_stores_summary["Store Name"] == st.session_state.sales_selected_store].index[0] + 1
                    total_stores = len(all_stores_summary)
                    
                    # Display ranking metrics
                    rank_cols = st.columns(4)
                    with rank_cols[0]:
                        st.metric("Store Rank", f"#{store_rank} of {total_stores}")
                    
                    with rank_cols[1]:
                        store_percentile = (store_rank / total_stores) * 100
                        st.metric("Percentile", f"{store_percentile:.1f}%")
                    
                    with rank_cols[2]:
                        if store_rank > 1:
                            next_store = all_stores_summary.iloc[store_rank - 2]
                            gap_sales = store_gross_sales - next_store["Gross Sales"]
                            st.metric("Sales Gap to Next", f"${gap_sales:,.0f}")
                        else:
                            st.metric("Sales Gap to Next", "🏆 Top Store")
                    
                    with rank_cols[3]:
                        if store_rank > 1:
                            gap_orders = store_orders - next_store["Total Delivered or Picked Up Orders"]
                            st.metric("Orders Gap to Next", f"{int(gap_orders):,}")
                        else:
                            st.metric("Orders Gap to Next", "🏆 Top Store")
                    
                    # Top 5 stores comparison
                    st.markdown("**🏆 Top 5 Stores by Gross Sales**")
                    top_stores = all_stores_summary.nlargest(5, "Gross Sales")
                    fig_top = px.bar(top_stores, x="Store Name", y="Gross Sales", 
                                   title="Top 5 Stores - Gross Sales Comparison",
                                   color="Gross Sales", color_continuous_scale="viridis")
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Top 5 stores by AOV
                    st.markdown("**💰 Top 5 Stores by Average Order Value**")
                    top_aov = all_stores_summary.nlargest(5, "AOV")
                    fig_aov = px.bar(top_aov, x="Store Name", y="AOV", 
                                   title="Top 5 Stores - AOV Comparison",
                                   color="AOV", color_continuous_scale="plasma")
                    st.plotly_chart(fig_aov, use_container_width=True)
                
            else:
                st.warning(f"No data found for store: {st.session_state.sales_selected_store}")
        else:
            # All stores overview
            st.markdown("### 📊 All Stores Overview")
            
            if "Store Name" in sales_df.columns:
                # Store summary table
                store_summary = sales_df.groupby("Store Name").agg({
                    "Gross Sales": "sum",
                    "Total Delivered or Picked Up Orders": "sum",
                    "AOV": "mean",
                    "Total Commission": "sum"
                }).reset_index()
                
                st.dataframe(store_summary, use_container_width=True)
                
                # Store performance heatmap
                if px is not None:
                    store_metrics = store_summary.set_index("Store Name")[["Gross Sales", 
                                                                        "Total Delivered or Picked Up Orders", 
                                                                        "AOV", "Total Commission"]]
                    fig_heatmap = px.imshow(store_metrics.T, 
                                          title="Store Sales Performance Heatmap",
                                          color_continuous_scale="viridis",
                                          aspect="auto")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Store Name column not found in sales data. Store-level analysis unavailable.")


# -------------------------------
# Section: Payouts
# -------------------------------

def section_payouts(payout_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="💸 Payouts Analytics",
            description="Store-level payment analysis, commission tracking, and financial settlements",
            color_name="red-70"
        )
    else:
        st.markdown("## 💸 Payouts Analytics")
        st.markdown("*Store-level payment analysis, commission tracking, and financial settlements*")
    
    if payout_df.empty:
        st.info("📭 Payouts CSV not found or empty.")
        return

    payout_df = parse_datetime_column(payout_df, "Payout Date")
    
    # Show filtering information
    st.info(f"📊 Payouts Data Summary: {len(payout_df)} rows loaded")
    if "Payout Date" in payout_df.columns:
        date_range = f"Date range: {payout_df['Payout Date'].min()} to {payout_df['Payout Date'].max()}"
        st.info(date_range)
        st.info("📅 Note: July 29, 2025 filtered out and data ends on August 10, 2025 as requested")

    # Store-Level Analysis Section
    st.markdown("## 🏪 Store-Level Payouts Analysis")
    
    # Group by Store and Date
    if {"Store Name", "Payout Date"}.issubset(payout_df.columns):
        metrics = [
            c for c in [
                "Net Payout",
                "Subtotal",
                "Commission",
                "Drive Charge",
                "Marketing Fees | (Including any applicable taxes)",
                "Customer Discounts from Marketing | (Funded by You)",
            ] if c in payout_df.columns
        ]
        grouped = payout_df.groupby(["Store Name", "Payout Date"])[metrics].sum().reset_index()

        stores = sorted(grouped["Store Name"].unique().tolist())
        
        # Create store selection with buttons
        st.markdown("**Select Store for Analysis:**")
        
        # Create a grid of buttons for store selection
        cols = st.columns(3)  # 3 columns for better layout
        
        # Initialize session state for store selection
        if "payouts_selected_store" not in st.session_state:
            st.session_state.payouts_selected_store = stores[0] if stores else None
        
        for i, store in enumerate(stores):
            col_idx = i % 3
            with cols[col_idx]:
                if st.button(store, key=f"store_{i}", use_container_width=True):
                    st.session_state.payouts_selected_store = store
        
        # Apply the selection
        if st.session_state.payouts_selected_store:
            data = grouped[grouped["Store Name"] == st.session_state.payouts_selected_store].sort_values("Payout Date")
            
            # Store-level detailed analysis
            st.markdown(f"## 📊 {st.session_state.payouts_selected_store} - Detailed Payout Analysis")
            
            # Store-specific KPIs
            store_kpi_cols = st.columns(4)
            with store_kpi_cols[0]:
                store_net_payout = data[net_payout_col].sum()
                st.metric("Store Net Payout", f"${store_net_payout:,.0f}")
            with store_kpi_cols[1]:
                store_avg_payout = data[net_payout_col].mean()
                st.metric("Avg Payout/Period", f"${store_avg_payout:,.0f}")
            with store_kpi_cols[2]:
                store_commission = data[commission_col].sum()
                st.metric("Total Commission", f"${store_commission:,.0f}")
            with store_kpi_cols[3]:
                store_mk_fees = data[mk_fee_col].sum()
                st.metric("Total Marketing Fees", f"${store_mk_fees:,.0f}")
            
            # Store payout trends
            if len(data) > 1:
                st.markdown("**📈 Payout Trends Over Time**")
                fig_trend = px.line(data, x="Payout Date", y=net_payout_col, 
                                  title=f"{st.session_state.payouts_selected_store} - Net Payout Trend",
                                  markers=True)
                fig_trend.update_layout(xaxis_title="Payout Date", yaxis_title="Net Payout ($)")
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Payout components breakdown
                st.markdown("**💰 Payout Components Breakdown**")
                components_data = data.melt(id_vars=["Payout Date"], 
                                          value_vars=[c for c in metrics if c != "Payout Date"],
                                          var_name="Component", value_name="Amount")
                fig_components = px.bar(components_data, x="Payout Date", y="Amount", 
                                      color="Component", 
                                      title=f"{st.session_state.payouts_selected_store} - Payout Components by Period")
                st.plotly_chart(fig_components, use_container_width=True)
                
                # Store performance summary
                st.markdown("**📋 Store Performance Summary**")
                summary_data = data[metrics].describe()
                st.dataframe(summary_data, use_container_width=True)
                
                # Comparison with other stores
                st.markdown("**🔍 Store Comparison**")
                all_stores_summary = grouped.groupby("Store Name")[net_payout_col].sum().sort_values(ascending=False)
                
                # Only show ranking if a specific store is selected (not "All Stores")
                if st.session_state.payouts_selected_store != "All Stores":
                    # Find current store's rank
                    store_rank = all_stores_summary.index.get_loc(st.session_state.payouts_selected_store) + 1
                    total_stores = len(all_stores_summary)
                    
                    # Display ranking metrics
                    rank_cols = st.columns(4)
                    with rank_cols[0]:
                        st.metric("Store Rank", f"#{store_rank} of {total_stores}")
                    
                    with rank_cols[1]:
                        store_percentile = (store_rank / total_stores) * 100
                        st.metric("Percentile", f"{store_percentile:.1f}%")
                    
                    with rank_cols[2]:
                        if store_rank > 1:
                            next_store = all_stores_summary.iloc[store_rank - 2]
                            gap = store_net_payout - next_store
                            st.metric("Gap to Next", f"${gap:,.0f}")
                        else:
                            st.metric("Gap to Next", "🏆 Top Store")
                    
                    with rank_cols[3]:
                        if store_rank > 1:
                            gap_payout = store_net_payout - next_store
                            st.metric("Payout Gap to Next", f"${gap_payout:,.0f}")
                        else:
                            st.metric("Payout Gap to Next", "🏆 Top Store")
                    
                    # Top 5 stores comparison
                    st.markdown("**🏆 Top 5 Stores by Net Payout**")
                    top_stores = all_stores_summary.head(5)
                    fig_top = px.bar(x=top_stores.values, y=top_stores.index, 
                                   orientation='h', 
                                   title="Top 5 Stores - Net Payout Comparison")
                    fig_top.update_layout(xaxis_title="Net Payout ($)", yaxis_title="Store Name")
                    st.plotly_chart(fig_top, use_container_width=True)
        
        else:
            data = grouped.groupby("Payout Date")[metrics].sum().reset_index().sort_values("Payout Date")
            
            # Overall summary for all stores
            st.markdown("**📊 All Stores - Overall Payout Summary**")
            
            # Overall trends
            if len(data) > 1:
                st.markdown("**📈 Overall Payout Trends**")
                fig_overall = px.line(data, x="Payout Date", y=net_payout_col, 
                                    title="All Stores - Net Payout Trend Over Time",
                                    markers=True)
                fig_overall.update_layout(xaxis_title="Payout Date", yaxis_title="Net Payout ($)")
                st.plotly_chart(fig_overall, use_container_width=True)

        if px is not None and not data.empty:
            st.markdown("**Payout components over time**")
            fig = px.line(data, x="Payout Date", y=metrics, title=(st.session_state.payouts_selected_store if st.session_state.payouts_selected_store != "All Stores" else "All Stores"))
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Section: UberEats
# -------------------------------

def section_ubereats(ubereats_df: pd.DataFrame):
    # Debug information as requested by user
    st.info("🔍 DEBUG: UberEats Section - Starting Analysis")
    st.info(f"🔍 DEBUG: DataFrame received - Shape: {ubereats_df.shape}")
    st.info(f"🔍 DEBUG: DataFrame columns: {list(ubereats_df.columns)}")
    st.info(f"🔍 DEBUG: DataFrame info - {ubereats_df.info() if not ubereats_df.empty else 'Empty DataFrame'}")
    
    # Colourful section header
    if colored_header:
        colored_header(
            label="🚗 UberEats Analytics",
            description="Store-level sales and payout analysis for UberEats operations",
            color_name="blue-70"
        )
    else:
        st.markdown("## 🚗 UberEats Analytics")
        st.markdown("*Store-level sales and payout analysis for UberEats operations*")
    
    if ubereats_df.empty:
        st.info("📭 UberEats CSV not found or empty.")
        st.info("🔍 DEBUG: DataFrame is empty - this is why analysis cannot proceed")
        return

    # Parse and prepare data
    ubereats_df = parse_datetime_column(ubereats_df, "Payout Date")
    ubereats_df = add_week_start(ubereats_df, "Payout Date", new_col="Week")

    # Calculate derived metrics
    ubereats_df = ubereats_df.copy()
    if "Sales (incl. tax)" in ubereats_df.columns and "Order Count" in ubereats_df.columns:
        ubereats_df["Average Order Value"] = safe_divide(
            ubereats_df["Sales (incl. tax)"], ubereats_df["Order Count"]
        )
    
    if "Total Sales after Adjustments (incl tax)" in ubereats_df.columns and "Sales (incl. tax)" in ubereats_df.columns:
        ubereats_df["Adjustment Rate"] = safe_divide(
            ubereats_df["Total Sales after Adjustments (incl tax)"] - ubereats_df["Sales (incl. tax)"],
            ubereats_df["Sales (incl. tax)"]
        )

    # Show data summary
    st.info(f"📊 UberEats Data Summary: {len(ubereats_df)} rows loaded")
    if "Payout Date" in ubereats_df.columns:
        date_range = f"Date range: {ubereats_df['Payout Date'].min()} to {ubereats_df['Payout Date'].max()}"
        st.info(date_range)
        st.info("📅 Note: Data filtered to end on August 10, 2025 as requested")

    # Store-Level Analysis Section
    st.markdown("## 🏪 Store-Level UberEats Analysis")
    
    # Get unique stores from UberEats data
    if "Store Name" in ubereats_df.columns:
        stores = sorted(ubereats_df["Store Name"].dropna().unique().tolist())
        
        # Store selection interface
        st.markdown("**Select Store for Analysis:**")
        
        # Create store selection buttons in a grid
        store_cols = st.columns(4)
        
        # Initialize session state for store selection
        if "ubereats_selected_store" not in st.session_state:
            st.session_state.ubereats_selected_store = stores[0] if stores else None
        
        for i, store in enumerate(stores):
            col_idx = i % 4
            with store_cols[col_idx]:
                if st.button(store, key=f"ubereats_store_{i}", use_container_width=True):
                    st.session_state.ubereats_selected_store = store
        
        # Store-specific analysis
        if st.session_state.ubereats_selected_store:
            st.markdown(f"## 📊 {st.session_state.ubereats_selected_store} - UberEats Performance Analysis")
            
            # Filter data for selected store
            store_data = ubereats_df[ubereats_df["Store Name"] == st.session_state.ubereats_selected_store].copy()
            
            if not store_data.empty:
                # Store-specific KPIs
                st.markdown("### 🎯 Store Performance Metrics")
                store_kpi_cols = st.columns(4)
                
                with store_kpi_cols[0]:
                    store_orders = store_data.get("Order Count", pd.Series(dtype=float)).sum()
                    st.metric("Total Orders", f"{int(store_orders):,}")
                
                with store_kpi_cols[1]:
                    store_sales = store_data.get("Sales (incl. tax)", pd.Series(dtype=float)).sum()
                    st.metric("Total Sales", f"${store_sales:,.0f}")
                
                with store_kpi_cols[2]:
                    store_payout = store_data.get("Total payout ", pd.Series(dtype=float)).sum()
                    st.metric("Total Payout", f"${store_payout:,.0f}")
                
                with store_kpi_cols[3]:
                    store_aov = store_data.get("Average Order Value", pd.Series(dtype=float)).mean()
                    if pd.notnull(store_aov):
                        st.metric("Average Order Value", f"${store_aov:,.2f}")
                    else:
                        st.metric("Average Order Value", "—")
                
                # Financial breakdown for the store
                st.markdown("### 💰 Store Financial Breakdown")
                financial_breakdown_cols = st.columns(3)
                
                with financial_breakdown_cols[0]:
                    store_marketplace_fees = store_data.get("Marketplace Fee", pd.Series(dtype=float)).sum()
                    st.metric("Marketplace Fees", f"${store_marketplace_fees:,.0f}")
                
                with financial_breakdown_cols[1]:
                    store_delivery_fees = store_data.get("Delivery Network Fee", pd.Series(dtype=float)).sum()
                    st.metric("Delivery Fees", f"${store_delivery_fees:,.0f}")
                
                with financial_breakdown_cols[2]:
                    store_processing_fees = store_data.get("Order Processing Fee", pd.Series(dtype=float)).sum()
                    st.metric("Processing Fees", f"${store_processing_fees:,.0f}")
                
                # Time series analysis for the store
                st.markdown("### 📈 Store Performance Over Time")
                if "Payout Date" in store_data.columns:
                    # Sort by date
                    store_data = store_data.sort_values("Payout Date")
                    
                    # Time series metrics
                    time_series_cols = [c for c in ["Order Count", "Sales (incl. tax)", "Total payout ", "Average Order Value"] 
                                      if c in store_data.columns]
                    
                    if time_series_cols:
                        for col in time_series_cols:
                            fig = px.line(store_data, x="Payout Date", y=col, 
                                        title=f"{st.session_state.ubereats_selected_store} - {col} Over Time",
                                        markers=True)
                            fig.update_layout(xaxis_title="Payout Date", yaxis_title=col)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Store comparison with other stores
                st.markdown("### 🔍 Store Performance Comparison")
                
                # Calculate store rankings
                all_stores_summary = ubereats_df.groupby("Store Name").agg({
                    "Order Count": "sum",
                    "Sales (incl. tax)": "sum",
                    "Total payout ": "sum",
                    "Average Order Value": "mean"
                }).reset_index()
                
                # Only show ranking if a specific store is selected (not "All Stores")
                if st.session_state.ubereats_selected_store != "All Stores":
                    # Find current store's rank
                    store_rank = all_stores_summary[all_stores_summary["Store Name"] == st.session_state.ubereats_selected_store].index[0] + 1
                    total_stores = len(all_stores_summary)
                    
                    # Display ranking metrics
                    rank_cols = st.columns(4)
                    with rank_cols[0]:
                        st.metric("Store Rank", f"#{store_rank} of {total_stores}")
                    
                    with rank_cols[1]:
                        store_percentile = (store_rank / total_stores) * 100
                        st.metric("Percentile", f"{store_percentile:.1f}%")
                    
                    with rank_cols[2]:
                        if store_rank > 1:
                            next_store = all_stores_summary.iloc[store_rank - 2]
                            gap_orders = store_orders - next_store["Order Count"]
                            st.metric("Orders Gap to Next", f"{int(gap_orders):,}")
                        else:
                            st.metric("Orders Gap to Next", "🏆 Top Store")
                    
                    with rank_cols[3]:
                        if store_rank > 1:
                            gap_sales = store_sales - next_store["Sales (incl. tax)"]
                            st.metric("Sales Gap to Next", f"${gap_sales:,.0f}")
                        else:
                            st.metric("Sales Gap to Next", "🏆 Top Store")
                    
                    # Top 5 stores comparison
                    st.markdown("**🏆 Top 5 Stores by Orders**")
                    top_stores = all_stores_summary.nlargest(5, "Order Count")
                    fig_top = px.bar(top_stores, x="Store Name", y="Order Count", 
                                   title="Top 5 Stores - Orders Comparison",
                                   color="Order Count", color_continuous_scale="viridis")
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Top 5 stores by sales
                    st.markdown("**💰 Top 5 Stores by Sales**")
                    top_sales = all_stores_summary.nlargest(5, "Sales (incl. tax)")
                    fig_sales_top = px.bar(top_sales, x="Store Name", y="Sales (incl. tax)", 
                                         title="Top 5 Stores - Sales Comparison",
                                         color="Sales (incl. tax)", color_continuous_scale="plasma")
                    st.plotly_chart(fig_sales_top, use_container_width=True)
        
    else:
        st.info("Store Name column not found in UberEats data. Store-level analysis unavailable.")


# -------------------------------
# Section: TODC Pre/Post Analysis
# -------------------------------

def section_todc_analysis(marketing_df: pd.DataFrame, ops_df: pd.DataFrame, sales_df: pd.DataFrame, payout_df: pd.DataFrame):
    """Comprehensive analysis of TODC impact comparing pre (6/1-7/2) vs post (7/3-8/3) periods"""
    
    # Colourful section header
    if colored_header:
        colored_header(
            label="🚀 TODC Impact Analysis",
            description="Pre/Post analysis of Third Order Delivery Company impact from July 3rd onwards",
            color_name="blue-70"
        )
    else:
        st.markdown("## 🚀 TODC Impact Analysis")
        st.markdown("*Pre/Post analysis of Third Order Delivery Company impact from July 3rd onwards*")
    
    # Define TODC periods
    pre_start = pd.Timestamp("2025-06-01")
    pre_end = pd.Timestamp("2025-07-02")
    post_start = pd.Timestamp("2025-07-03")
    post_end = pd.Timestamp("2025-08-03")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 TODC Analysis Periods**")
    st.sidebar.info(f"📅 **Pre-TODC:** {pre_start.strftime('%Y-%m-%d')} to {pre_end.strftime('%Y-%m-%d')}")
    st.sidebar.info(f"📅 **Post-TODC:** {post_start.strftime('%Y-%m-%d')} to {post_end.strftime('%Y-%m-%d')}")
    
    # Validate input data
    if sales_df.empty:
        st.warning("⚠️ **No Sales Data:** Sales dataset is empty or not available.")
        return
    
    if "Start Date" not in sales_df.columns:
        st.warning("⚠️ **Missing Date Column:** Sales dataset missing 'Start Date' column.")
        return
    
    # Overall Performance Analysis
    st.markdown("### 📊 Overall Performance: Pre vs Post TODC")
    
    # Marketing Analysis
    if not marketing_df.empty:
        st.markdown("#### 📢 Marketing Performance")
        
        # Filter data for pre and post periods
        pre_marketing = marketing_df[
            (pd.to_datetime(marketing_df["Date"]) >= pre_start) & 
            (pd.to_datetime(marketing_df["Date"]) <= pre_end)
        ]
        post_marketing = marketing_df[
            (pd.to_datetime(marketing_df["Date"]) >= post_start) & 
            (pd.to_datetime(marketing_df["Date"]) <= post_end)
        ]
        
        # Calculate key marketing metrics
        marketing_metrics = {
            "Orders": "Orders",
            "Sales": "Sales",
            "ROAS": "ROAS",
            "New Customers": "New Customers Acquired"
        }
        
        marketing_comparison = []
        for metric_name, col_name in marketing_metrics.items():
            if col_name in marketing_df.columns:
                if metric_name == "ROAS":
                    pre_val = pre_marketing[col_name].mean()
                    post_val = post_marketing[col_name].mean()
                else:
                    pre_val = pre_marketing[col_name].sum()
                    post_val = post_marketing[col_name].sum()
                
                if pre_val != 0:
                    pct_change = ((post_val - pre_val) / pre_val) * 100
                else:
                    pct_change = 0
                
                marketing_comparison.append({
                    "Metric": metric_name,
                    "Pre-TODC": pre_val,
                    "Post-TODC": post_val,
                    "Absolute Change": post_val - pre_val,
                    "% Change": pct_change
                })
        
        if marketing_comparison:
            marketing_df_comparison = pd.DataFrame(marketing_comparison)
            st.dataframe(marketing_df_comparison, use_container_width=True)
    
    # Operations Analysis
    if not ops_df.empty:
        st.markdown("#### ⚙️ Operations Performance")
        
        # Filter data for pre and post periods
        pre_ops = ops_df[
            (pd.to_datetime(ops_df["Start Date"]) >= pre_start) & 
            (pd.to_datetime(ops_df["Start Date"]) <= pre_end)
        ]
        post_ops = ops_df[
            (pd.to_datetime(ops_df["Start Date"]) >= post_start) & 
            (pd.to_datetime(ops_df["Start Date"]) <= post_end)
        ]
        
        # Calculate key operations metrics
        ops_metrics = {
            "Delivered Orders": "Total Delivered or Picked Up Orders",
            "Cancellation Rate": "Total Cancellation Rate %",
            "Average Rating": "Average Rating",
            "Downtime (min)": "Total Downtime in Minutes"
        }
        
        ops_comparison = []
        for metric_name, col_name in ops_metrics.items():
            if col_name in ops_df.columns:
                if "Rate" in metric_name or "Rating" in metric_name:
                    pre_val = pre_ops[col_name].mean()
                    post_val = post_ops[col_name].mean()
                else:
                    pre_val = pre_ops[col_name].sum()
                    post_val = post_ops[col_name].sum()
                
                if pre_val != 0:
                    pct_change = ((post_val - pre_val) / pre_val) * 100
                else:
                    pct_change = 0
                
                ops_comparison.append({
                    "Metric": metric_name,
                    "Pre-TODC": pre_val,
                    "Post-TODC": post_val,
                    "Absolute Change": post_val - pre_val,
                    "% Change": pct_change
                })
        
        if ops_comparison:
            ops_df_comparison = pd.DataFrame(ops_comparison)
            st.dataframe(ops_df_comparison, use_container_width=True)
    
    # Payouts Analysis
    if not payout_df.empty:
        st.markdown("#### 💸 Payouts Performance")
        
        # Filter data for pre and post periods
        pre_payouts = payout_df[
            (pd.to_datetime(payout_df["Payout Date"]) >= pre_start) & 
            (pd.to_datetime(payout_df["Payout Date"]) <= pre_end)
        ]
        post_payouts = payout_df[
            (pd.to_datetime(payout_df["Payout Date"]) >= post_start) & 
            (pd.to_datetime(payout_df["Payout Date"]) <= post_end)
        ]
        
        # Calculate key payout metrics
        payout_metrics = {
            "Net Payout": "Net Payout",
            "Subtotal": "Subtotal",
            "Commission": "Commission",
            "Marketing Fees": "Marketing Fees | (Including any applicable taxes)"
        }
        
        payout_comparison = []
        for metric_name, col_name in payout_metrics.items():
            if col_name in payout_df.columns:
                pre_val = pre_payouts[col_name].sum()
                post_val = post_payouts[col_name].sum()
                
                if pre_val != 0:
                    pct_change = ((post_val - pre_val) / pre_val) * 100
                else:
                    pct_change = 0
                
                payout_comparison.append({
                    "Metric": metric_name,
                    "Pre-TODC": pre_val,
                    "Post-TODC": post_val,
                    "Absolute Change": post_val - pre_val,
                    "% Change": pct_change
                })
        
        if payout_comparison:
            payout_df_comparison = pd.DataFrame(payout_comparison)
            st.dataframe(payout_df_comparison, use_container_width=True)
    
    # Sales Analysis
    st.markdown("#### 💰 Sales Performance")
    
    # Filter data for pre and post periods
    pre_sales = sales_df[
        (pd.to_datetime(sales_df["Start Date"]) >= pre_start) & 
        (pd.to_datetime(sales_df["Start Date"]) <= pre_end)
    ]
    post_sales = sales_df[
        (pd.to_datetime(sales_df["Start Date"]) >= post_start) & 
        (pd.to_datetime(sales_df["Start Date"]) <= post_end)
    ]
    
    # Validate filtered data
    if pre_sales.empty:
        st.warning("⚠️ **No Pre-TODC Data:** No sales data found for the pre-TODC period.")
        return
    
    if post_sales.empty:
        st.warning("⚠️ **No Post-TODC Data:** No sales data found for the post-TODC period.")
        return
    
    # Calculate key metrics
    sales_metrics = {
        "Gross Sales": "Gross Sales",
        "Total Orders": "Total Orders Including Cancelled Orders",
        "Delivered Orders": "Total Delivered or Picked Up Orders",
        "AOV": "AOV"
    }
    
    sales_comparison = []
    for metric_name, col_name in sales_metrics.items():
        if col_name in sales_df.columns:
            pre_val = pre_sales[col_name].sum() if metric_name != "AOV" else pre_sales[col_name].mean()
            post_val = post_sales[col_name].sum() if metric_name != "AOV" else post_sales[col_name].mean()
            
            if pre_val != 0:
                pct_change = ((post_val - pre_val) / pre_val) * 100
            else:
                pct_change = 0
            
            sales_comparison.append({
                "Metric": metric_name,
                "Pre-TODC": pre_val,
                "Post-TODC": post_val,
                "Absolute Change": post_val - pre_val,
                "% Change": pct_change
            })
    
    if sales_comparison:
        sales_df_comparison = pd.DataFrame(sales_comparison)
        st.dataframe(sales_df_comparison, use_container_width=True)
        
        # Visualize key metrics
        if px is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gross Sales comparison
                fig_sales = px.bar(
                    sales_df_comparison[sales_df_comparison["Metric"] == "Gross Sales"],
                    x="Metric",
                    y=["Pre-TODC", "Post-TODC"],
                    title="💰 Gross Sales: Pre vs Post TODC",
                    barmode="group"
                )
                fig_sales.update_layout(
                    title_font_size=16,
                    title_font_color="#FF6B35",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_sales, use_container_width=True)
            
            with col2:
                # Orders comparison
                fig_orders = px.bar(
                    sales_df_comparison[sales_df_comparison["Metric"] == "Total Orders"],
                    x="Metric",
                    y=["Pre-TODC", "Post-TODC"],
                    title="📦 Total Orders: Pre vs Post TODC",
                    barmode="group"
                )
                fig_orders.update_layout(
                    title_font_size=16,
                    title_font_color="#FF6B35",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_orders, use_container_width=True)
    
    # Time Series Analysis
    st.markdown("### 📈 Time Series: TODC Impact Over Time")
    
    if not sales_df.empty and "Start Date" in sales_df.columns:
        # Add week column for time series
        sales_time = sales_df.copy()
        sales_time["Week"] = pd.to_datetime(sales_time["Start Date"]).dt.to_period("W-MON").apply(lambda r: r.start_time)
        
        # Aggregate by week
        weekly_sales = sales_time.groupby("Week").agg({
            "Gross Sales": "sum",
            "Total Orders Including Cancelled Orders": "sum",
            "Total Delivered or Picked Up Orders": "sum"
        }).reset_index()
        
        # Add period indicator
        weekly_sales["Period"] = weekly_sales["Week"].apply(
            lambda x: "Pre-TODC" if x < post_start else "Post-TODC"
        )
        
        if px is not None:
            # Gross Sales over time
            fig_time_sales = px.line(
                weekly_sales,
                x="Week",
                y="Gross Sales",
                color="Period",
                title="💰 Gross Sales Over Time: Pre vs Post TODC",
                color_discrete_map={"Pre-TODC": "#FF6B35", "Post-TODC": "#4ECDC4"}
            )
            fig_time_sales.update_layout(
                title_font_size=16,
                title_font_color="#FF6B35",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_time_sales, use_container_width=True)
            
            # Orders over time
            fig_time_orders = px.line(
                weekly_sales,
                x="Week",
                y="Total Delivered or Picked Up Orders",
                color="Period",
                title="📦 Delivered Orders Over Time: Pre vs Post TODC",
                color_discrete_map={"Pre-TODC": "#FF6B35", "Post-TODC": "#4ECDC4"}
            )
            fig_time_orders.update_layout(
                title_font_size=16,
                title_font_color="#FF6B35",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_time_orders, use_container_width=True)
    
    # Summary and Insights
    st.markdown("### 💡 Key Insights & Recommendations")
    
    # Calculate overall impact - ensure variables are defined
    if not sales_df.empty:
        # Define pre_sales and post_sales at function level if not already defined
        if 'pre_sales' not in locals():
            pre_sales = sales_df[
                (pd.to_datetime(sales_df["Start Date"]) >= pre_start) & 
                (pd.to_datetime(sales_df["Start Date"]) <= pre_end)
            ]
        if 'post_sales' not in locals():
            post_sales = sales_df[
                (pd.to_datetime(sales_df["Start Date"]) >= post_start) & 
                (pd.to_datetime(sales_df["Start Date"]) <= post_end)
            ]
        
        pre_total_sales = pre_sales["Gross Sales"].sum() if not pre_sales.empty else 0
        post_total_sales = post_sales["Gross Sales"].sum() if not post_sales.empty else 0
        
        if pre_total_sales > 0:
            overall_growth = ((post_total_sales - pre_total_sales) / pre_total_sales) * 100
            
            st.info(f"""
            **📊 Overall TODC Impact Summary:**
            - **Pre-TODC Period (6/1-7/2):** ${pre_total_sales:,.0f} in sales
            - **Post-TODC Period (7/3-8/3):** ${post_total_sales:,.0f} in sales
            - **Overall Growth:** {overall_growth:+.1f}%
            """)
            
            if overall_growth > 0:
                st.success("🎉 **Positive Impact:** TODC implementation shows positive results!")
            else:
                st.warning("⚠️ **Needs Attention:** TODC implementation may need optimization.")
        else:
            st.warning("⚠️ **Insufficient Data:** Not enough pre-TODC data to calculate impact.")
    else:
        st.warning("⚠️ **No Sales Data:** Sales dataset is empty or not available.")
    
    # Recommendations based on data
    st.markdown("""
    **🔍 Analysis Recommendations:**
    1. **Monitor Store Performance:** Track individual store performance post-TODC
    2. **Identify Success Factors:** Analyze what's working in top-performing stores
    3. **Optimize Operations:** Focus on areas showing decline or stagnation
    4. **Customer Experience:** Monitor ratings and delivery times for quality assurance
    5. **Marketing ROI:** Evaluate marketing spend effectiveness in the new delivery model
    """)
    
    # Store-Level Analysis
    st.markdown("### 🏪 Store-Level TODC Impact Analysis")
    
    if not sales_df.empty and "Store Name" in sales_df.columns:
        # Get unique stores
        stores = sales_df["Store Name"].unique()
        
        # Store selection
        selected_store = st.selectbox("Select Store for Detailed Analysis:", ["All Stores"] + list(stores))
        
        if selected_store != "All Stores":
            store_sales = sales_df[sales_df["Store Name"] == selected_store]
            
            # Filter by periods
            pre_store = store_sales[
                (pd.to_datetime(store_sales["Start Date"]) >= pre_start) & 
                (pd.to_datetime(store_sales["Start Date"]) <= pre_end)
            ]
            post_store = store_sales[
                (pd.to_datetime(store_sales["Start Date"]) >= post_start) & 
                (pd.to_datetime(store_sales["Start Date"]) <= post_end)
            ]
            
            # Calculate store-specific metrics
            store_metrics = {
                "Gross Sales": "Gross Sales",
                "Total Orders": "Total Orders Including Cancelled Orders",
                "Delivered Orders": "Total Delivered or Picked Up Orders",
                "AOV": "AOV"
            }
            
            store_comparison = []
            for metric_name, col_name in store_metrics.items():
                if col_name in store_sales.columns:
                    pre_val = pre_store[col_name].sum() if metric_name != "AOV" else pre_store[col_name].mean()
                    post_val = post_store[col_name].sum() if metric_name != "AOV" else post_store[col_name].mean()
                    
                    if pre_val != 0:
                        pct_change = ((post_val - pre_val) / pre_val) * 100
                    else:
                        pct_change = 0
                    
                    store_comparison.append({
                        "Metric": metric_name,
                        "Pre-TODC": pre_val,
                        "Post-TODC": post_val,
                        "Absolute Change": post_val - pre_val,
                        "% Change": pct_change
                    })
            
            if store_comparison:
                st.markdown(f"#### 📊 {selected_store} Performance")
                store_df_comparison = pd.DataFrame(store_comparison)
                st.dataframe(store_df_comparison, use_container_width=True)
                
                # Visualize store performance
                if px is not None:
                    fig_store = px.bar(
                        store_df_comparison,
                        x="Metric",
                        y=["Pre-TODC", "Post-TODC"],
                        title=f"📈 {selected_store}: Pre vs Post TODC Performance",
                        barmode="group"
                    )
                    fig_store.update_layout(
                        title_font_size=16,
                        title_font_color="#FF6B35",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_store, use_container_width=True)
        
        # Top Performers Analysis
        st.markdown("#### 🏆 Top Performing Stores Post-TODC")
        
        # Calculate growth for all stores
        store_growth = []
        for store in stores:
            store_data = sales_df[sales_df["Store Name"] == store]
            
            pre_store_data = store_data[
                (pd.to_datetime(store_data["Start Date"]) >= pre_start) & 
                (pd.to_datetime(store_data["Start Date"]) <= pre_end)
            ]
            post_store_data = store_data[
                (pd.to_datetime(store_data["Start Date"]) >= post_start) & 
                (pd.to_datetime(store_data["Start Date"]) <= post_end)
            ]
            
            pre_store_sales = pre_store_data["Gross Sales"].sum() if not pre_store_data.empty else 0
            post_store_sales = post_store_data["Gross Sales"].sum() if not post_store_data.empty else 0
            
            if pre_store_sales > 0:
                growth_pct = ((post_store_sales - pre_store_sales) / pre_store_sales) * 100
                store_growth.append({
                    "Store": store,
                    "Pre-TODC Sales": pre_store_sales,
                    "Post-TODC Sales": post_store_sales,
                    "Growth %": growth_pct
                })
        
        if store_growth:
            growth_df = pd.DataFrame(store_growth)
            growth_df = growth_df.sort_values("Growth %", ascending=False)
            
            # Top 5 and Bottom 5
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚀 Top 5 Growth Stores**")
                top_5 = growth_df.head(5)
                st.dataframe(top_5, use_container_width=True)
            
            with col2:
                st.markdown("**📉 Bottom 5 Growth Stores**")
                bottom_5 = growth_df.tail(5)
                st.dataframe(bottom_5, use_container_width=True)
            
            # Growth visualization
            if px is not None:
                fig_growth = px.bar(
                    growth_df.head(10),
                    x="Store",
                    y="Growth %",
                    title="📈 Top 10 Stores by Sales Growth Post-TODC",
                    color="Growth %",
                    color_continuous_scale="RdYlGn"
                )
                fig_growth.update_layout(
                    title_font_size=16,
                    title_font_color="#FF6B35",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_growth, use_container_width=True)
    
    # Time Series Analysis


# -------------------------------
# Overview cards combining key stats
# -------------------------------

def section_overview(marketing_df: pd.DataFrame, ops_df: pd.DataFrame, sales_df: pd.DataFrame, payout_df: pd.DataFrame, ubereats_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="📊 Data Overview Dashboard",
            description="Comprehensive insights across all DoorDash and UberEats datasets",
            color_name="blue-70"
        )
    else:
        st.markdown("## 📊 Data Overview Dashboard")
        st.markdown("*Comprehensive insights across all DoorDash and UberEats datasets*")
    
    # Enhanced KPI cards with icons and colours
    st.markdown("### 🎯 Key Performance Indicators")
    
    # Calculate metrics
    total_orders = np.nan
    if "Total Delivered or Picked Up Orders" in sales_df.columns:
        total_orders = pd.to_numeric(sales_df["Total Delivered or Picked Up Orders"], errors="coerce").sum()
    elif "Orders" in marketing_df.columns:
        total_orders = pd.to_numeric(marketing_df["Orders"], errors="coerce").sum()
    val = int(total_orders) if pd.notnull(total_orders) else 0
    
    gross_sales = pd.to_numeric(sales_df.get("Gross Sales", pd.Series(dtype=float)), errors="coerce").sum()
    avg_rating = pd.to_numeric(ops_df.get("Average Rating", pd.Series(dtype=float)), errors="coerce").mean()
    net_payout_col = find_column_by_keywords(payout_df, ["net payout"]) or "Net Payout"
    net_payout = pd.to_numeric(payout_df.get(net_payout_col, pd.Series(dtype=float)), errors="coerce").sum()
    
    # Use regular metrics with styling
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("📦 Total Orders", f"{val:,}")
    
    with c2:
        st.metric("💰 Gross Sales", f"${gross_sales:,.0f}")
    
    with c3:
        st.metric("💸 Net Payout", f"${net_payout:,.0f}")
    
    # Apply metric card styling if available
    if style_metric_cards:
        style_metric_cards()

    # Data Overview with clean, aligned metric cards
    st.markdown("### 📁 Data Overview")
    
    # Calculate data overview metrics
    files_count = sum([
        1 if not df.empty else 0 
        for df in [marketing_df, ops_df, sales_df, payout_df, ubereats_df]
    ])
    
    total_rows = sum([
        len(df) for df in [marketing_df, ops_df, sales_df, payout_df, ubereats_df]
    ])
    
    total_columns = sum([
        len(df.columns) for df in [marketing_df, ops_df, sales_df, payout_df, ubereats_df]
    ])
    
    all_stores = set()
    for df in [marketing_df, ops_df, sales_df, payout_df, ubereats_df]:
        if "Store Name" in df.columns:
            all_stores.update(df["Store Name"].dropna().unique())
        if "Store ID" in df.columns:
            all_stores.update(df["Store ID"].dropna().astype(str).unique())
    
    # Create clean, aligned metric cards without images
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    
    with overview_col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; margin-bottom: 10px;">📂</div>
            <div style="font-weight: bold; font-size: 18px; margin-bottom: 5px;">Files Loaded</div>
            <div style="font-size: 16px;">{files_count}/4 datasets</div>
        </div>
        """.format(files_count=files_count), unsafe_allow_html=True)
    
    with overview_col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; margin-bottom: 10px;">📊</div>
            <div style="font-weight: bold; font-size: 18px; margin-bottom: 5px;">Total Rows</div>
            <div style="font-size: 16px;">{total_rows:,} data points</div>
        </div>
        """.format(total_rows=total_rows), unsafe_allow_html=True)
    
    with overview_col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; margin-bottom: 10px;">📋</div>
            <div style="font-weight: bold; font-size: 18px; margin-bottom: 5px;">Total Columns</div>
            <div style="font-size: 16px;">{total_columns} unique fields</div>
        </div>
        """.format(total_columns=total_columns), unsafe_allow_html=True)

    # Campaign & Feature Analysis with colourful badges
    st.markdown("### 🎯 Campaign & Feature Analysis")
    
    # Calculate campaign and feature metrics
    all_campaigns = set()
    for df in [marketing_df, ops_df, sales_df, payout_df, ubereats_df]:
        campaign_cols = [col for col in df.columns if 'campaign' in col.lower()]
        for col in campaign_cols:
            all_campaigns.update(df[col].dropna().unique())
    
    important_features = set()
    key_metrics = [
        'orders', 'sales', 'revenue', 'commission', 'payout', 'rating', 
        'delivery', 'cancellation', 'downtime', 'promotion', 'ad', 'discount'
    ]
    for df in [marketing_df, ops_df, sales_df, payout_df, ubereats_df]:
        for col in df.columns:
            col_lower = col.lower()
            if any(metric in col_lower for metric in key_metrics):
                important_features.add(col)
    
    # Display campaign and feature counts
    campaign_cols = st.columns(2)
    with campaign_cols[0]:
        st.metric("📢 Campaigns", len(all_campaigns))
    with campaign_cols[1]:
        st.metric("🔧 Features", len(important_features))

    # File-specific breakdown with detailed analysis
    st.markdown("### 📋 File Breakdown")
    st.markdown("*Detailed analysis of each dataset's characteristics*")
    file_data = []
    
    for name, df in [("Marketing", marketing_df), ("Operations", ops_df), ("Sales", sales_df), ("Payouts", payout_df), ("UberEats", ubereats_df)]:
        # Find date column for each file
        date_col = None
        if name == "Marketing" and "Date" in df.columns:
            date_col = "Date"
        elif name == "Operations" and "Start Date" in df.columns:
            date_col = "Start Date"
        elif name == "Sales" and "Start Date" in df.columns:
            date_col = "Start Date"
        elif name == "Payouts" and "Payout Date" in df.columns:
            date_col = "Payout Date"
        elif name == "UberEats" and "Payout Date" in df.columns:
            date_col = "Payout Date"
        
        # Calculate date range
        date_range = "N/A"
        if date_col and not df.empty:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
        
        file_data.append({
            "File": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Date Range": date_range
        })
    
    file_df = pd.DataFrame(file_data)
    st.dataframe(file_df, use_container_width=True)
    
    # Detailed column analysis by file
    st.markdown("### Important Columns by File")
    
    for name, df in [("Marketing", marketing_df), ("Operations", ops_df), ("Sales", sales_df), ("Payouts", payout_df), ("ubereats", ubereats_df)]:
        if not df.empty:
            st.markdown(f"**{name} File:**")
            
            # Find important columns
            important_cols = []
            key_metrics = [
                'orders', 'sales', 'revenue', 'commission', 'payout', 'rating', 
                'delivery', 'cancellation', 'downtime', 'promotion', 'ad', 'discount',
                'campaign', 'store', 'date', 'customer', 'fee', 'charge'
            ]
            
            for col in df.columns:
                col_lower = col.lower()
                if any(metric in col_lower for metric in key_metrics):
                    unique_count = len(df[col].dropna().unique())
                    important_cols.append({
                        "Column": col,
                        "Unique Values": unique_count,
                        "Data Type": str(df[col].dtype)
                    })
            
            if important_cols:
                col_df = pd.DataFrame(important_cols).sort_values("Unique Values", ascending=False)
                st.dataframe(col_df, use_container_width=True)
            else:
                st.info("No important columns found in this file.")
            
            st.markdown("---")
    
    # Enhanced Visualizations with colourful themes
    if px is not None:
        st.markdown("### 📊 Data Visualizations")
        st.markdown("*Interactive charts showing data distribution across files*")
        
        # Create a 2x2 grid for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # File sizes comparison with enhanced styling
            fig1 = px.bar(
                file_df, 
                x="File", 
                y="Rows", 
                title="📈 Number of Rows by File",
                color="File",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_dark"
            )
            fig1.update_layout(
                title_font_size=16,
                title_font_color="#FF6B35",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig1, use_container_width=True)
            

        
        with col2:
            # Column count comparison with enhanced styling
            fig2 = px.bar(
                file_df, 
                x="File", 
                y="Columns", 
                title="📋 Number of Columns by File",
                color="File",
                color_discrete_sequence=px.colors.qualitative.Set1,
                template="plotly_dark"
            )
            fig2.update_layout(
                title_font_size=16,
                title_font_color="#FF6B35",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)
            

        
        # Enhanced Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        st.markdown("*Overall data summary across all datasets*")
        
        summary_data = {
            "📁 Metric": ["Total Files", "Total Rows", "Total Columns"],
            "📈 Value": [
                files_count,
                total_rows,
                total_columns
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Style the dataframe with custom CSS
        st.markdown("""
        <style>
        .summary-table {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            summary_df, 
            use_container_width=True,
            hide_index=True
        )


# -------------------------------
# Main App
# -------------------------------

def main():
    st.set_page_config(
        page_title="🚀 DoorDash & UberEats Performance Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .success-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        text-align: center;
    }
    .stMetric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        color: white !important;
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] label {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] div {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] span {
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Colourful header with icon
    if colored_header:
        colored_header(
            label="🚀 DoorDash & UberEats Performance Dashboard",
            description="Comprehensive analytics and insights for DoorDash and UberEats operations",
            color_name="orange-70"
        )
    else:
        st.markdown('<div class="main-header"><h1>🚀 DoorDash & UberEats Performance Dashboard</h1><p>Comprehensive analytics and insights for DoorDash and UberEats operations</p></div>', unsafe_allow_html=True)

    # Load data
    with st.spinner("🚀 Loading DoorDash and UberEats datasets..."):
        mk = load_csv(CSV_FILES["marketing"])  # marketing
        ops = load_csv(CSV_FILES["operations"])  # operations
        pay = load_csv(CSV_FILES["payouts"])  # payouts
        sal = load_csv(CSV_FILES["sales"])  # sales
        ue = load_csv(CSV_FILES["ubereats"])  # ubereats
        
        # Debug information for UberEats as requested by user
        st.info(f"🔍 DEBUG: UberEats file path: {CSV_FILES['ubereats']}")
        st.info(f"🔍 DEBUG: UberEats file exists: {CSV_FILES['ubereats'].exists()}")
        st.info(f"🔍 DEBUG: UberEats DataFrame loaded - Shape: {ue.shape}")
        st.info(f"🔍 DEBUG: UberEats DataFrame columns: {list(ue.columns) if not ue.empty else 'Empty DataFrame'}")
        if not ue.empty:
            st.info(f"🔍 DEBUG: UberEats first few rows: {ue.head(3).to_dict()}")
    
    # Apply date filtering based on user requirements
    # 1. Remove July 29 from payout date
    # 2. Filter all data to end on August 10, 2025 (remove August 11 and beyond)
    
    # Filter payouts to exclude July 29, 2025
    if not pay.empty and "Payout Date" in pay.columns:
        pay = parse_datetime_column(pay, "Payout Date")
        july_29_filter = pay["Payout Date"].dt.date != pd.to_datetime("2025-07-29").date()
        pay = pay[july_29_filter]
    
    # Filter all datasets to end on August 10, 2025
    august_10_end = pd.to_datetime("2025-08-10")
    
    if not mk.empty and "Date" in mk.columns:
        mk = parse_datetime_column(mk, "Date")
        mk = filter_df_by_date(mk, "Date", None, august_10_end)
    
    if not ops.empty and "Start Date" in ops.columns:
        ops = parse_datetime_column(ops, "Start Date")
        ops = filter_df_by_date(ops, "Start Date", None, august_10_end)
    
    if not pay.empty and "Payout Date" in pay.columns:
        pay = filter_df_by_date(pay, "Payout Date", None, august_10_end)
    
    if not sal.empty and "Start Date" in sal.columns:
        sal = parse_datetime_column(sal, "Start Date")
        sal = filter_df_by_date(sal, "Start Date", None, august_10_end)
    
    if not ue.empty and "Payout Date" in ue.columns:
        st.info(f"🔍 DEBUG: Before date filtering - UberEats shape: {ue.shape}")
        ue = parse_datetime_column(ue, "Payout Date")
        st.info(f"🔍 DEBUG: After datetime parsing - UberEats shape: {ue.shape}")
        ue = filter_df_by_date(ue, "Payout Date", None, august_10_end)
        st.info(f"🔍 DEBUG: After date filtering - UberEats shape: {ue.shape}")
    else:
        st.info(f"🔍 DEBUG: UberEats DataFrame empty or missing 'Payout Date' column")
        st.info(f"🔍 DEBUG: UberEats columns available: {list(ue.columns) if not ue.empty else 'Empty DataFrame'}")
    
    # Success message with animation
    if st_lottie:
        try:
            # Simple success animation
            st_lottie(
                "https://assets5.lottiefiles.com/packages/lf20_49rdyysj.json",
                height=100,
                key="success"
            )
        except:
            st.success("✅ All datasets loaded successfully!")
    else:
        st.success("✅ All datasets loaded successfully!")

    # Enhanced sidebar with clickable buttons
    with st.sidebar:
        st.header("📊 Dashboard Navigation")
        
        # Initialize session state for navigation
        if "selected_section" not in st.session_state:
            st.session_state.selected_section = "Overview"
        
        # Create buttons for navigation in one column
        if st.button("🏠 Overview", key="btn_overview", use_container_width=True):
            st.session_state.selected_section = "Overview"
        if st.button("📢 Marketing", key="btn_marketing", use_container_width=True):
            st.session_state.selected_section = "Marketing"
        if st.button("⚙️ Operations", key="btn_operations", use_container_width=True):
            st.session_state.selected_section = "Operations"
        if st.button("💰 Sales", key="btn_sales", use_container_width=True):
            st.session_state.selected_section = "Sales"
        if st.button("💸 Payouts", key="btn_payouts", use_container_width=True):
            st.session_state.selected_section = "Payouts"
        if st.button("🚗 UberEats", key="btn_ubereats", use_container_width=True):
            st.session_state.selected_section = "UberEats"
        
        # Add separator and TODC Analysis button
        st.markdown("---")
        if st.button("🚀 TODC Analysis", key="btn_todc", use_container_width=True):
            st.session_state.selected_section = "TODC"

        st.markdown("---")
        st.header("🔍 Overall Date Filter")
        
        # Determine min/max dates across datasets for convenience
        date_candidates: List[Tuple[pd.Series, str]] = []
        if "Date" in mk.columns:
            date_candidates.append((pd.to_datetime(mk["Date"], errors="coerce"), "Date"))
        if "Start Date" in ops.columns:
            date_candidates.append((pd.to_datetime(ops["Start Date"], errors="coerce"), "Start Date"))
        if "Payout Date" in pay.columns:
            date_candidates.append((pd.to_datetime(pay["Payout Date"], errors="coerce"), "Payout Date"))
        if "Start Date" in sal.columns:
            date_candidates.append((pd.to_datetime(sal["Start Date"], errors="coerce"), "Start Date"))
        if "Payout Date" in ue.columns:
            try:
                date_candidates.append((pd.to_datetime(ue["Payout Date"], format='%m/%d/%y', errors="coerce"), "Payout Date"))
            except:
                date_candidates.append((pd.to_datetime(ue["Payout Date"], errors="coerce"), "Payout Date"))

        all_dates = pd.concat([s for s, _ in date_candidates], axis=0) if date_candidates else pd.Series([], dtype="datetime64[ns]")
        min_date = pd.to_datetime(all_dates.min()) if not all_dates.empty else None
        max_date = pd.to_datetime(all_dates.max()) if not all_dates.empty else None

        date_filter = None
        if min_date is not None and max_date is not None:
            date_filter = st.date_input("Overall Date Range", (min_date, max_date))

        if isinstance(date_filter, tuple) and len(date_filter) == 2:
            start_date = pd.to_datetime(date_filter[0])
            end_date = pd.to_datetime(date_filter[1])
            # Apply dataset-specific filters
            if "Date" in mk.columns:
                mk = filter_df_by_date(mk, "Date", start_date, end_date)
            if "Start Date" in ops.columns:
                ops = filter_df_by_date(ops, "Start Date", start_date, end_date)
            if "Payout Date" in pay.columns:
                pay = filter_df_by_date(pay, "Payout Date", start_date, end_date)
            if "Start Date" in sal.columns:
                sal = filter_df_by_date(sal, "Start Date", start_date, end_date)
            if "Payout Date" in ue.columns:
                ue = filter_df_by_date(ue, "Payout Date", start_date, end_date)
        
    # Enhanced navigation based on selected option
    if st.session_state.selected_section == "Overview":
        section_overview(mk, ops, sal, pay, ue)
    elif st.session_state.selected_section == "Marketing":
        section_marketing(mk)
    elif st.session_state.selected_section == "Operations":
        section_operations(ops)
    elif st.session_state.selected_section == "Sales":
        section_sales(sal)
    elif st.session_state.selected_section == "Payouts":
        section_payouts(pay)
    elif st.session_state.selected_section == "UberEats":
        section_ubereats(ue)
    elif st.session_state.selected_section == "TODC":
        section_todc_analysis(mk, ops, sal, pay)


if __name__ == "__main__":
    main()
