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
            # Use average for rating metrics, sum for others
            if "Average" in col or "Rating" in col:
                metrics[col] = period_df[col].mean()
            else:
                metrics[col] = period_df[col].sum()
    
    return metrics

def create_growth_comparison_chart(df, date_column, value_columns, period1_start, period1_end, period2_start, period2_end):
    """Create comparison chart for two periods"""
    # Calculate metrics for both periods
    period1_metrics = calculate_period_metrics(df, date_column, value_columns, period1_start, period1_end)
    period2_metrics = calculate_period_metrics(df, date_column, value_columns, period2_start, period2_end)
    
    # Create comparison data
    comparison_data = []
    for col in value_columns:
        if col in period1_metrics and col in period2_metrics:
            growth_pct = ((period2_metrics[col] - period1_metrics[col]) / period1_metrics[col] * 100) if period1_metrics[col] != 0 else 0
            comparison_data.append({
                "Metric": col,
                "Pre TODC (Jun 6 - Jul 6)": period1_metrics[col],
                "Post TODC (Jul 7 - Aug 6)": period2_metrics[col],
                "Delta": period2_metrics[col] - period1_metrics[col],
                "% Change": f"{growth_pct:+.2f}%"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name="Pre TODC (Jun 6 - Jul 6)",
            x=comparison_df["Metric"],
            y=comparison_df["Pre TODC (Jun 6 - Jul 6)"],
            marker_color='#3498db',
            text=comparison_df["Pre TODC (Jun 6 - Jul 6)"].round(2),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name="Post TODC (Jul 7 - Aug 6)",
            x=comparison_df["Metric"],
            y=comparison_df["Post TODC (Jul 7 - Aug 6)"],
            marker_color='#e74c3c',
            text=comparison_df["Post TODC (Jul 7 - Aug 6)"].round(2),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Metrics Comparison: Pre TODC vs Post TODC",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            height=500
        )
        
        return fig, comparison_df
    
    return None, None

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
    marketing_df = load_csv(CSV_FILES["marketing"])
    operations_df = load_csv(CSV_FILES["operations"])
    sales_df = load_csv(CSV_FILES["sales"])
    payouts_df = load_csv(CSV_FILES["payouts"])
    ubereats_df = load_csv(CSV_FILES["ubereats"])
    
    # Metrics Growth Analysis Section
    st.markdown('<h2 class="section-header">📈 Metrics Growth Analysis</h2>', unsafe_allow_html=True)
    
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
    
    # Marketing Analysis
    if not marketing_df.empty:
        st.markdown("### 📢 Marketing Growth")
        marketing_df = parse_datetime_column(marketing_df, "Date")
        
        if "Date" in marketing_df.columns:
            # Check available columns in marketing data
            # available_columns = marketing_df.columns.tolist()
            # st.markdown(f"**Available columns in marketing data:** {', '.join(available_columns)}")
            
            # Define comprehensive marketing metrics to analyze
            marketing_metrics = [
                "Sales", 
                "Orders", 
                "AOV",
                "ROAS",
                "New Customers Acquired", 
                "Existing Customers Acquired",
                "Total Customers Acquired",
                "New DP Customers Acquired",
                "Existing DP Customers Acquired",
                "Total DP Customers Acquired",
                "Marketing Spend",
                "Customer Acquisition Cost",
                "Return on Ad Spend",
                "Conversion Rate"
            ]
            
            # Filter to only include metrics that exist in the dataframe
            valid_marketing_metrics = [metric for metric in marketing_metrics if metric in marketing_df.columns]
            
            if valid_marketing_metrics:
                # 1. Pre vs Post TODC Comparison
                st.markdown("#### 📅 Pre vs Post TODC Comparison")
                fig, comparison_df = create_growth_comparison_chart(
                    marketing_df, "Date", valid_marketing_metrics, 
                    period1_start, period1_end, period2_start, period2_end
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed comparison table
                    st.markdown("**Detailed Pre vs Post TODC Comparison:**")
                    st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)
                
                # 2. Campaign Type Comparison (Post TODC Period)
                st.markdown("#### 🎯 Campaign Type Comparison (Post TODC Period)")
                campaign_fig, campaign_comparison_df = create_campaign_comparison_chart(
                    marketing_df, "Date", valid_marketing_metrics,
                    period2_start, period2_end  # Using the latest month (Jul 7 - Aug 6)
                )
                
                if campaign_fig:
                    st.plotly_chart(campaign_fig, use_container_width=True)
                    
                    # Show detailed campaign comparison table
                    st.markdown("**Detailed Campaign Type Comparison:**")
                    st.dataframe(campaign_comparison_df.set_index("Metric"), use_container_width=True)
                    

                else:
                    st.warning("No 'Is Self Serve Campaign' column found in marketing data for campaign comparison.")
                

            else:
                st.warning("No valid marketing metrics found in the dataset.")
        else:
            st.warning("No 'Date' column found in marketing data.")
    
    # Operations Analysis
    if not operations_df.empty:
        st.markdown("### ⚙️ Operations Growth")
        operations_df = parse_datetime_column(operations_df, "Start Date")
        
        if "Start Date" in operations_df.columns:
            # Check available columns in operations data
            # available_columns = operations_df.columns.tolist()
            # st.markdown(f"**Available columns in operations data:** {', '.join(available_columns)}")
            
            # Define comprehensive operations metrics to analyze
            operations_metrics = [
                "Total Delivered or Picked Up Orders",
                "Total Orders Including Cancelled Orders",
                "Total Missing",
                "Missing/Incomplete",
                "Total Error",
                "Total Avoidable",
                "Average AHT",
                "Average Delivery Time (ASAP)",
                "Uptime %",
                "Average Rating",
                "Total 1 Star",
                "Total 5 Star"
            ]
            
            # Filter to only include metrics that exist in the dataframe
            valid_operations_metrics = [metric for metric in operations_metrics if metric in operations_df.columns]
            
            if valid_operations_metrics:
                fig, comparison_df = create_growth_comparison_chart(
                    operations_df, "Start Date", valid_operations_metrics,
                    period1_start, period1_end, period2_start, period2_end
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed comparison table
                    st.markdown("**Detailed Operations Comparison:**")
                    st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)
                    

            else:
                st.warning("No valid operations metrics found in the dataset. Available columns: " + ", ".join(available_columns))
        else:
            st.warning("No 'Start Date' column found in operations data. Available columns: " + ", ".join(operations_df.columns.tolist()))
    
    # Sales Analysis
    if not sales_df.empty:
        st.markdown("### 💰 Sales Growth")
        sales_df = parse_datetime_column(sales_df, "Start Date")
        
        if "Start Date" in sales_df.columns:
            # Check available columns in sales data
            # available_columns = sales_df.columns.tolist()
            # st.markdown(f"**Available columns in sales data:** {', '.join(available_columns)}")
            
            # Define sales metrics to analyze - using actual column names from the dataset
            sales_metrics = [
                "Gross Sales", 
                "Total Orders Including Cancelled Orders", 
                "Total Delivered or Picked Up Orders",
                "AOV",
                "Total Commission",
                "Total Promotion Fees | (for historical reference only)",
                "Total Promotion Sales | (for historical reference only)",
                "Total Ad Fees | (for historical reference only)",
                "Total Ad Sales | (for historical reference only)"
            ]
            
            # Filter to only include metrics that exist in the dataframe
            valid_sales_metrics = [metric for metric in sales_metrics if metric in sales_df.columns]
            
            if valid_sales_metrics:
                fig, comparison_df = create_growth_comparison_chart(
                    sales_df, "Start Date", valid_sales_metrics,
                    period1_start, period1_end, period2_start, period2_end
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed comparison table with percentage changes
                    st.markdown("**Detailed Sales Comparison:**")
                    st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)
                    

            else:
                st.warning("No valid sales metrics found in the dataset. Available columns: " + ", ".join(available_columns))
        else:
            st.warning("No 'Start Date' column found in sales data. Available columns: " + ", ".join(sales_df.columns.tolist()))
    
    # Payouts Analysis
    if not payouts_df.empty:
        st.markdown("### 💸 Payouts Metrics Growth")
        payouts_df = parse_datetime_column(payouts_df, "Payout Date")
        
        if "Payout Date" in payouts_df.columns:
            # Check available columns in payouts data
            # available_columns = payouts_df.columns.tolist()
            # st.markdown(f"**Available columns in payouts data:** {', '.join(available_columns)}")
            
            # Define comprehensive payouts metrics to analyze
            payouts_metrics = [
                "Subtotal", 
                "Net Payout", 
                "Commission",
                "Gross Sales",
                "Total Orders",
                "Total Delivered Orders",
                "AOV",
                "Total Commission",
                "Total Promotion Fees",
                "Total Promotion Sales",
                "Total Ad Fees",
                "Total Ad Sales",
                "Marketing Fees (for historical reference only) | (All discounts and fees)"
            ]
            
            # Filter to only include metrics that exist in the dataframe
            valid_payouts_metrics = [metric for metric in payouts_metrics if metric in payouts_df.columns]
            
            if valid_payouts_metrics:
                fig, comparison_df = create_growth_comparison_chart(
                    payouts_df, "Payout Date", valid_payouts_metrics,
                    period1_start, period1_end, period2_start, period2_end
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed comparison table
                    st.markdown("**Detailed Payouts Comparison:**")
                    st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)
                    

            else:
                st.warning("No valid payouts metrics found in the dataset. Available columns: " + ", ".join(available_columns))
        else:
            st.warning("No 'Payout Date' column found in payouts data. Available columns: " + ", ".join(payouts_df.columns.tolist()))
    
    # UberEats Analysis
    if not ubereats_df.empty:
        st.markdown("### 🚗 UberEats Metrics Growth")
        ubereats_df = parse_datetime_column(ubereats_df, "Payout Date")
        
        if "Payout Date" in ubereats_df.columns:
            # Check available columns in UberEats data
            # available_columns = ubereats_df.columns.tolist()
            # st.markdown(f"**Available columns in UberEats data:** {', '.join(available_columns)}")
            
            # Define comprehensive UberEats metrics to analyze
            ubereats_metrics = [
                "Sales (incl. tax)", 
                "Total Sales after Adjustments (incl tax)",
                "Order Count", 
                "Total payout ",
                "Gross Sales",
                "Total Orders",
                "Commission",
                "Net Payout"
            ]
            
            # Filter to only include metrics that exist in the dataframe
            valid_ubereats_metrics = [metric for metric in ubereats_metrics if metric in ubereats_df.columns]
            
            if valid_ubereats_metrics:
                fig, comparison_df = create_growth_comparison_chart(
                    ubereats_df, "Payout Date", valid_ubereats_metrics,
                    period1_start, period1_end, period2_start, period2_end
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed comparison table
                    st.markdown("**Detailed UberEats Comparison:**")
                    st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)
                    

            else:
                st.warning("No valid UberEats metrics found in the dataset. Available columns: " + ", ".join(available_columns))
        else:
            st.warning("No 'Payout Date' column found in UberEats data. Available columns: " + ", ".join(ubereats_df.columns.tolist()))
    


if __name__ == "__main__":
    main()
