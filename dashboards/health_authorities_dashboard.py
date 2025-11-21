"""
Health Authorities Dashboard
Public health risk monitoring dashboard for health authorities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboards.utils import (
    load_all_nodes_data,
    load_node_data,
    get_node_summary,
    get_regional_comparison,
    calculate_risk_trends,
    get_alerts,
    format_timestamp
)

# Page configuration
st.set_page_config(
    page_title="Health Risk Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff0000;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff4cc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffcc00;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 Public Health Risk Monitoring Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_cached():
    return load_all_nodes_data()

try:
    df = load_data_cached()
    
    if df.empty:
        st.error("⚠️ No data found. Please run the data pipeline first: `python src/data_ingestion/main.py`")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.info("💡 Make sure you have run the data pipeline first: `python src/data_ingestion/main.py`")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_nodes = st.sidebar.multiselect(
    "Select Regions",
    options=['hospital_A', 'hospital_B', 'hospital_C'],
    default=['hospital_A', 'hospital_B', 'hospital_C']
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
    min_value=df['timestamp'].min().date(),
    max_value=df['timestamp'].max().date()
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[
        (df['location'].isin(selected_nodes)) &
        (df['timestamp'].dt.date >= start_date) &
        (df['timestamp'].dt.date <= end_date)
    ]
else:
    df_filtered = df[df['location'].isin(selected_nodes)]

# Calculate summary statistics
total_records = len(df_filtered)
total_high_risk = df_filtered['health_risk'].sum() if 'health_risk' in df_filtered.columns else 0
risk_percentage = (total_high_risk / total_records * 100) if total_records > 0 else 0
avg_aqi = df_filtered['aqi'].mean() if 'aqi' in df_filtered.columns else 0
avg_pm25 = df_filtered['pm25'].mean() if 'pm25' in df_filtered.columns else 0

# Key Metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Records",
        value=f"{total_records:,}",
        delta=f"From {len(selected_nodes)} regions"
    )

with col2:
    st.metric(
        label="High Risk Rate",
        value=f"{risk_percentage:.1f}%",
        delta=f"{total_high_risk:,} cases",
        delta_color="inverse" if risk_percentage < 30 else "normal"
    )

with col3:
    aqi_status = "Good" if avg_aqi < 50 else "Moderate" if avg_aqi < 100 else "Unhealthy"
    st.metric(
        label="Average AQI",
        value=f"{avg_aqi:.0f}",
        delta=aqi_status,
        delta_color="normal" if avg_aqi < 50 else "off"
    )

with col4:
    st.metric(
        label="Average PM2.5",
        value=f"{avg_pm25:.1f}",
        delta="μg/m³"
    )

st.markdown("---")

# Alerts Section
st.subheader("🚨 Active Alerts")
alerts = []
for node in selected_nodes:
    node_df = df_filtered[df_filtered['location'] == node]
    if not node_df.empty:
        node_alerts = get_alerts(node_df)
        alerts.extend(node_alerts)

if alerts:
    for alert in alerts:
        severity_class = "alert-high" if alert['severity'] == 'high' else "alert-medium"
        st.markdown(f"""
            <div class="{severity_class}">
                <strong>⚠️ {alert['type'].upper()} Alert</strong> - {alert['location']}<br>
                {alert['message']}<br>
                <small>Time: {format_timestamp(alert['timestamp'])}</small>
            </div>
        """, unsafe_allow_html=True)
else:
    st.success("✅ No active alerts. All regions are within normal parameters.")

st.markdown("---")

# Regional Comparison
st.subheader("🌍 Regional Comparison")
comparison_df = get_regional_comparison()
comparison_df = comparison_df[comparison_df['node_name'].isin(selected_nodes)]

if not comparison_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # High Risk Percentage by Region
        fig_risk = px.bar(
            comparison_df,
            x='node_name',
            y='high_risk_percentage',
            title='High Risk Percentage by Region',
            labels={'node_name': 'Region', 'high_risk_percentage': 'High Risk %'},
            color='high_risk_percentage',
            color_continuous_scale='Reds'
        )
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Average AQI by Region
        fig_aqi = px.bar(
            comparison_df,
            x='node_name',
            y='avg_aqi',
            title='Average Air Quality Index by Region',
            labels={'node_name': 'Region', 'avg_aqi': 'Average AQI'},
            color='avg_aqi',
            color_continuous_scale='YlOrRd'
        )
        fig_aqi.update_layout(showlegend=False)
        st.plotly_chart(fig_aqi, use_container_width=True)

st.markdown("---")

# Risk Trends
st.subheader("📈 Risk Trends Over Time")
trends_df = calculate_risk_trends(df_filtered)

if not trends_df.empty:
    # Group by date and location
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily Risk Percentage
        fig_trend = px.line(
            trends_df,
            x='date',
            y='risk_percentage',
            color='node' if 'node' in trends_df.columns else None,
            title='Daily High Risk Percentage',
            labels={'date': 'Date', 'risk_percentage': 'High Risk %', 'node': 'Region'},
            markers=True
        )
        fig_trend.update_layout(hovermode='x unified')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Daily AQI Trend
        if 'avg_aqi' in trends_df.columns:
            fig_aqi_trend = px.line(
                trends_df,
                x='date',
                y='avg_aqi',
                color='node' if 'node' in trends_df.columns else None,
                title='Daily Average AQI',
                labels={'date': 'Date', 'avg_aqi': 'Average AQI', 'node': 'Region'},
                markers=True,
                color_discrete_map={'hospital_A': '#1f77b4', 'hospital_B': '#ff7f0e', 'hospital_C': '#2ca02c'}
            )
            fig_aqi_trend.update_layout(hovermode='x unified')
            st.plotly_chart(fig_aqi_trend, use_container_width=True)

st.markdown("---")

# Environmental Factors
st.subheader("🌡️ Environmental Factors")
env_cols = ['aqi', 'pm25', 'temperature', 'humidity']
available_env_cols = [col for col in env_cols if col in df_filtered.columns]

if available_env_cols:
    # Create subplots for environmental factors
    fig_env = make_subplots(
        rows=2, cols=2,
        subplot_titles=available_env_cols[:4],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    for idx, col in enumerate(available_env_cols[:4]):
        row, col_pos = positions[idx]
        for node in selected_nodes:
            node_df = df_filtered[df_filtered['location'] == node]
            if not node_df.empty:
                node_df_sorted = node_df.sort_values('timestamp')
                fig_env.add_trace(
                    go.Scatter(
                        x=node_df_sorted['timestamp'],
                        y=node_df_sorted[col],
                        name=f"{col} - {node}",
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=row, col=col_pos
                )
    
    fig_env.update_layout(
        height=600,
        title_text="Environmental Factors Over Time",
        showlegend=True,
        hovermode='x unified'
    )
    st.plotly_chart(fig_env, use_container_width=True)

st.markdown("---")

# Health Metrics
st.subheader("💓 Health Metrics")
health_cols = ['heart_rate_mean', 'spo2_mean', 'active_users_count']
available_health_cols = [col for col in health_cols if col in df_filtered.columns]

if available_health_cols:
    col1, col2 = st.columns(2)
    
    with col1:
        # Heart Rate Trends
        if 'heart_rate_mean' in available_health_cols:
            fig_hr = px.line(
                df_filtered.sort_values('timestamp'),
                x='timestamp',
                y='heart_rate_mean',
                color='location',
                title='Average Heart Rate Over Time',
                labels={'timestamp': 'Time', 'heart_rate_mean': 'Heart Rate (bpm)', 'location': 'Region'},
                markers=True
            )
            st.plotly_chart(fig_hr, use_container_width=True)
    
    with col2:
        # SpO2 Trends
        if 'spo2_mean' in available_health_cols:
            fig_spo2 = px.line(
                df_filtered.sort_values('timestamp'),
                x='timestamp',
                y='spo2_mean',
                color='location',
                title='Average SpO2 Over Time',
                labels={'timestamp': 'Time', 'spo2_mean': 'SpO2 (%)', 'location': 'Region'},
                markers=True
            )
            st.plotly_chart(fig_spo2, use_container_width=True)

st.markdown("---")

# Data Table
st.subheader("📋 Detailed Data View")
if st.checkbox("Show detailed data table"):
    display_cols = ['timestamp', 'location', 'health_risk', 'aqi', 'pm25', 'temperature', 
                   'heart_rate_mean', 'spo2_mean']
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    
    st.dataframe(
        df_filtered[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True,
        height=400
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🏥 Health Risk Monitoring Dashboard | Last updated: {}</p>
        <p>Data from Federated Learning System | MLOps Project</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)

