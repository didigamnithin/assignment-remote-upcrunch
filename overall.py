import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="UpChurch Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
        color: #2c3e50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .data-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #2c3e50;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# File paths
CSV_FILES = {
    "marketing": Path("upchurch doordash marketing.csv"),
    "operations": Path("upchurch doordash operations.csv"),
    "sales": Path("upchurch doordash sales.csv"),
    "payouts": Path("upchurch doordash payouts.csv"),
    "ubereats": Path("upchurch ubereats sales and payouts.csv")
}

def load_csv(file_path):
    """Load CSV file with error handling"""
    try:
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df
        else:
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def parse_datetime_column(df, column_name):
    """Parse datetime column with multiple format support"""
    if column_name in df.columns:
        try:
            # Try different date formats
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        except:
            pass
    return df

def calculate_period_metrics(df, date_column, value_columns, period_start, period_end):
    """Calculate metrics for a specific time period"""
    period_df = df[
        (df[date_column] >= period_start) & 
        (df[date_column] <= period_end)
    ].copy()
    
    metrics = {}
    for col in value_columns:
        if col in period_df.columns:
            # Use average for rating metrics, uptime, and average metrics, sum for others
            if "Average" in col or "Rating" in col or "Uptime" in col:
                metrics[col] = period_df[col].mean()
            else:
                metrics[col] = period_df[col].sum()
    
    return metrics

def create_blended_metrics_chart(all_dataframes, period1_start, period1_end, period2_start, period2_end, metric_group):
    """Create blended metrics chart for a specific metric group"""
    
    # Define metric groups and their corresponding data sources
    metric_groups = {
        "orders": {
            "metrics": [
                "Orders", "Total Orders Including Cancelled Orders", "Total Delivered or Picked Up Orders",
                "Total Orders", "Total Delivered Orders"
            ],
            "sources": ["marketing", "operations", "sales", "payouts", "ubereats"],
            "date_columns": ["Date", "Start Date", "Start Date", "Payout Date", "Payout Date"]
        },
        "sales": {
            "metrics": [
                "Sales", "Gross Sales", "Sales (incl. tax)", "Total Sales after Adjustments (incl tax)",
                "AOV", "Total Promotion Sales | (for historical reference only)", "Total Ad Sales | (for historical reference only)"
            ],
            "sources": ["marketing", "sales", "payouts", "ubereats"],
            "date_columns": ["Date", "Start Date", "Payout Date", "Payout Date"]
        },
        "financial_performance": {
            "metrics": [
                "Net Payout", "Total payout ",
                "Total Promotion Fees | (for historical reference only)", "Total Ad Fees | (for historical reference only)",
                "Marketing Fees (for historical reference only) | (All discounts and fees)",
                "Marketing Spend", "Customer Acquisition Cost",
                "ROAS", "Return on Ad Spend", "Conversion Rate", "Average Rating",
                "Average AHT"
            ],
            "sources": ["marketing", "sales", "payouts", "ubereats", "operations"],
            "date_columns": ["Date", "Start Date", "Payout Date", "Payout Date", "Start Date"]
        },
        "customers": {
            "metrics": [
                "New Customers Acquired", "Existing Customers Acquired", "Total Customers Acquired",
                "New DP Customers Acquired", "Existing DP Customers Acquired", "Total DP Customers Acquired"
            ],
            "sources": ["marketing"],
            "date_columns": ["Date"]
        },
        "quality": {
            "metrics": [
                "Total Missing", "Missing/Incomplete", "Total Error", "Total Avoidable",
                "Total 1 Star", "Total 5 Star"
            ],
            "sources": ["operations"],
            "date_columns": ["Start Date"]
        }
    }
    
    if metric_group not in metric_groups:
        return None, None
    
    group_config = metric_groups[metric_group]
    
    # Collect all metrics data from different sources
    all_metrics_data = []
    
    for i, source in enumerate(group_config["sources"]):
        if source in all_dataframes and not all_dataframes[source].empty:
            df = all_dataframes[source]
            date_col = group_config["date_columns"][i]
            
            # Parse datetime if needed
            df = parse_datetime_column(df, date_col)
            
            if date_col in df.columns:
                # Calculate metrics for both periods
                period1_metrics = calculate_period_metrics(df, date_col, group_config["metrics"], period1_start, period1_end)
                period2_metrics = calculate_period_metrics(df, date_col, group_config["metrics"], period2_start, period2_end)
                
                # Add to collection
                for metric in group_config["metrics"]:
                    if metric in period1_metrics and metric in period2_metrics:
                        all_metrics_data.append({
                            "Metric": metric,
                            "Source": source,
                            "Pre TODC": period1_metrics[metric],
                            "Post TODC": period2_metrics[metric],
                            "Delta": period2_metrics[metric] - period1_metrics[metric],
                            "% Change": ((period2_metrics[metric] - period1_metrics[metric]) / period1_metrics[metric] * 100) if period1_metrics[metric] != 0 else 0
                        })
    
    if not all_metrics_data:
        return None, None
    
    # Create DataFrame and remove duplicates (same metric from different sources)
    metrics_df = pd.DataFrame(all_metrics_data)
    metrics_df = metrics_df.drop_duplicates(subset=['Metric'], keep='first')
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Pre TODC (Jun 6 - Jul 6)",
        x=metrics_df["Metric"],
        y=metrics_df["Pre TODC"],
        marker_color='#3498db',
        text=metrics_df["Pre TODC"].round(2),
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name="Post TODC (Jul 7 - Aug 6)",
        x=metrics_df["Metric"],
        y=metrics_df["Post TODC"],
        marker_color='#e74c3c',
        text=metrics_df["Post TODC"].round(2),
        textposition='auto'
    ))
    
    # Update layout based on metric group
    group_titles = {
        "orders": "📦 Orders & Delivery Metrics",
        "sales": "💰 Sales & Revenue Metrics", 
        "financial_performance": "💸 Financial & Performance Metrics",
        "customers": "👥 Customer Acquisition Metrics",
        "quality": "⭐ Quality & Error Metrics"
    }
    
    fig.update_layout(
        title=group_titles.get(metric_group, f"{metric_group.title()} Metrics"),
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig, metrics_df

def create_campaign_comparison_chart(df, date_column, value_columns, period_start, period_end):
    """Create comparison chart for Self Serve Campaign = FALSE vs TRUE"""
    # Filter data for the specified period
    period_df = df[
        (df[date_column] >= period_start) & 
        (df[date_column] <= period_end)
    ].copy()
    
    # Check if 'Is Self Serve Campaign' column exists
    if 'Is Self Serve Campaign' not in period_df.columns:
        return None, None
    
    # Calculate metrics for both campaign types
    false_campaign_metrics = {}
    true_campaign_metrics = {}
    
    for col in value_columns:
        if col in period_df.columns:
            # Filter for FALSE campaigns
            false_campaigns = period_df[period_df['Is Self Serve Campaign'] == False]
            if "Average" in col or "Rating" in col:
                false_campaign_metrics[col] = false_campaigns[col].mean() if not false_campaigns.empty else 0
            else:
                false_campaign_metrics[col] = false_campaigns[col].sum() if not false_campaigns.empty else 0
            
            # Filter for TRUE campaigns
            true_campaigns = period_df[period_df['Is Self Serve Campaign'] == True]
            if "Average" in col or "Rating" in col:
                true_campaign_metrics[col] = true_campaigns[col].mean() if not true_campaigns.empty else 0
            else:
                true_campaign_metrics[col] = true_campaigns[col].sum() if not true_campaigns.empty else 0
    
    # Create comparison data
    comparison_data = []
    for col in value_columns:
        if col in false_campaign_metrics and col in true_campaign_metrics:
            false_val = false_campaign_metrics[col]
            true_val = true_campaign_metrics[col]
            
            # Calculate percentage difference
            if false_val != 0:
                pct_diff = ((true_val - false_val) / false_val * 100)
            else:
                pct_diff = 0 if true_val == 0 else float('inf')
            
            comparison_data.append({
                "Metric": col,
                "Internal Campaigns": false_val,
                "TODC Campaigns": true_val,
                "Delta": true_val - false_val,
                "% Difference": f"{pct_diff:+.2f}%" if pct_diff != float('inf') else "N/A"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name="Internal Campaigns",
            x=comparison_df["Metric"],
            y=comparison_df["Internal Campaigns"],
            marker_color='#3498db',
            text=comparison_df["Internal Campaigns"].round(2),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name="TODC Campaigns",
            x=comparison_df["Metric"],
            y=comparison_df["TODC Campaigns"],
            marker_color='#e74c3c',
            text=comparison_df["TODC Campaigns"].round(2),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Campaign Type Comparison: {period_start.strftime('%b %d')} - {period_end.strftime('%b %d, %Y')}",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            height=500
        )
        
        return fig, comparison_df
    
    return None, None

def main():
    st.markdown('<h1 class="main-header">📊 UpChurch Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load all datasets silently
    all_dataframes = {
        "marketing": load_csv(CSV_FILES["marketing"]),
        "operations": load_csv(CSV_FILES["operations"]),
        "sales": load_csv(CSV_FILES["sales"]),
        "payouts": load_csv(CSV_FILES["payouts"]),
        "ubereats": load_csv(CSV_FILES["ubereats"])
    }
    
    # Metrics Growth Analysis Section
    st.markdown('<h2 class="section-header">📈 Blended Metrics Growth Analysis</h2>', unsafe_allow_html=True)
    
    # Define time periods
    period1_start = pd.Timestamp("2025-06-06")
    period1_end = pd.Timestamp("2025-07-06")
    period2_start = pd.Timestamp("2025-07-07")
    period2_end = pd.Timestamp("2025-08-06")
    
    st.markdown(f"""
    <div class="metric-card">
        <strong>Analysis Periods:</strong><br>
        • <strong>Pre TODC:</strong> June 6, 2025 - July 6, 2025 (31 days)<br>
        • <strong>Post TODC:</strong> July 7, 2025 - August 6, 2025 (31 days)
    </div>
    """, unsafe_allow_html=True)
    
    # Create blended metrics charts for each group
    metric_groups = ["orders", "sales", "financial_performance", "customers", "quality"]
    
    for group in metric_groups:
        fig, comparison_df = create_blended_metrics_chart(
            all_dataframes, period1_start, period1_end, period2_start, period2_end, group
        )
        
        if fig and comparison_df is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed comparison table
            group_names = {
                "orders": "Orders & Delivery",
                "sales": "Sales & Revenue", 
                "financial_performance": "Financial & Performance",
                "customers": "Customer Acquisition",
                "quality": "Quality & Error"
            }
            
            st.markdown(f"**Detailed {group_names[group]} Comparison:**")
            display_df = comparison_df[["Metric", "Pre TODC", "Post TODC", "Delta", "% Change"]].copy()
            display_df["% Change"] = display_df["% Change"].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) else str(x))
            st.dataframe(display_df.set_index("Metric"), use_container_width=True)
            st.markdown("---")
    
    # Campaign Type Analysis (if marketing data available)
    if not all_dataframes["marketing"].empty:
        st.markdown('<h2 class="section-header">🎯 Campaign Type Analysis</h2>', unsafe_allow_html=True)
        
        marketing_df = all_dataframes["marketing"]
        marketing_df = parse_datetime_column(marketing_df, "Date")
        
        if "Date" in marketing_df.columns:
            # Define marketing metrics for campaign comparison
            marketing_metrics = [
                "Sales", "Orders", "AOV", "ROAS", "New Customers Acquired", 
                "Existing Customers Acquired", "Total Customers Acquired",
                "Marketing Spend", "Customer Acquisition Cost", "Conversion Rate"
            ]
            
            valid_marketing_metrics = [metric for metric in marketing_metrics if metric in marketing_df.columns]
            
            if valid_marketing_metrics:
                campaign_fig, campaign_comparison_df = create_campaign_comparison_chart(
                    marketing_df, "Date", valid_marketing_metrics,
                    period2_start, period2_end
                )
                
                if campaign_fig:
                    st.plotly_chart(campaign_fig, use_container_width=True)
                    
                    st.markdown("**Detailed Campaign Type Comparison:**")
                    st.dataframe(campaign_comparison_df.set_index("Metric"), use_container_width=True)
                else:
                    st.warning("No 'Is Self Serve Campaign' column found in marketing data for campaign comparison.")

if __name__ == "__main__":
    main()
