"""
Citizens Dashboard
Personal health risk monitoring dashboard for citizens
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
    load_node_data,
    get_alerts,
    get_risk_category,
    format_timestamp
    
)



# Page configuration
st.set_page_config(
    page_title="Personal Health Risk Dashboard",
    page_icon="👤",
    
    
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
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
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
    
    
    .alert-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00cc00;
        margin: 0.5rem 0;
    }
    
    
    
    .risk-high {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    
    
    .risk-medium {
        color: #ff9900;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    
    
    .risk-low {
        color: #00cc00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    
    
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">👤 Personal Health Risk Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Region Selection
st.sidebar.header("📍 Your Location")
selected_region = st.sidebar.selectbox(
    "Select Your Region",
    options=['hospital_A', 'hospital_B', 'hospital_C'],
    index=0,
    help="Select the region you are in to see personalized health information"
)

# Load data for selected region
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_region_data_cached(region):
    return load_node_data(region)

try:
    df = load_region_data_cached(selected_region)
    
    if df is None or df.empty:
        st.error(f"⚠️ No data found for {selected_region}. Please run the data pipeline first: `python src/data_ingestion/main.py`")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.info("💡 Make sure you have run the data pipeline first: `python src/data_ingestion/main.py`")
    st.stop()

# Sidebar - Date Range
st.sidebar.header("📅 Date Range")
default_end = df['timestamp'].max().date()
default_start = default_end - timedelta(days=7)

date_range = st.sidebar.date_input(
    "View data from",
    value=(default_start, default_end),
    min_value=df['timestamp'].min().date(),
    max_value=df['timestamp'].max().date()
)

# Filter data by date
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[
        (df['timestamp'].dt.date >= start_date) &
        (df['timestamp'].dt.date <= end_date)
    ].copy()
else:
    df_filtered = df.copy()

# Get latest data point for current status
latest = df_filtered.iloc[-1] if not df_filtered.empty else None

# Current Status Section
st.subheader("🎯 Current Health Status")

if latest is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Current Risk Level
        if 'health_risk' in latest:
            risk_value = latest['health_risk']
            risk_cat = get_risk_category(risk_value) if isinstance(risk_value, (int, float)) else "N/A"
            
            risk_class = "risk-high" if risk_cat == "High Risk" else "risk-medium" if risk_cat == "Medium Risk" else "risk-low"
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Risk</h3>
                    <p class="{risk_class}">{risk_cat}</p>
                    <small>Updated: {format_timestamp(latest['timestamp'])}</small>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Current AQI
        if 'aqi' in latest:
            aqi = latest['aqi']
            aqi_status = "Good" if aqi < 50 else "Moderate" if aqi < 100 else "Unhealthy"
            st.metric(
                label="Air Quality",
                value=f"AQI: {aqi:.0f}",
                delta=aqi_status,
                delta_color="normal" if aqi < 50 else "off"
            )
    
    with col3:
        # Current Temperature
        if 'temperature' in latest:
            temp = latest['temperature']
            st.metric(
                label="Temperature",
                value=f"{temp:.1f}°C",
                delta="Comfortable" if 15 <= temp <= 30 else "Extreme",
                delta_color="normal" if 15 <= temp <= 30 else "off"
            )
    
    with col4:
        # Current PM2.5
        if 'pm25' in latest:
            pm25 = latest['pm25']
            pm25_status = "Safe" if pm25 < 25 else "Moderate" if pm25 < 50 else "Unhealthy"
            st.metric(
                label="PM2.5",
                value=f"{pm25:.1f}",
                delta=pm25_status,
                delta_color="normal" if pm25 < 25 else "off"
            )

st.markdown("---")

# Personal Alerts Section
st.subheader("🚨 Your Personal Health Alerts")

if latest is not None:
    alerts = get_alerts(df_filtered, threshold_high_risk=0.3)
    
    if alerts:
        # Filter alerts for current region
        region_alerts = [a for a in alerts if a.get('location', selected_region) == selected_region]
        
        if region_alerts:
            for alert in region_alerts:
                severity_class = "alert-high" if alert['severity'] == 'high' else "alert-medium"
                st.markdown(f"""
                    <div class="{severity_class}">
                        <strong>⚠️ {alert['type'].upper()} Alert</strong><br>
                        {alert['message']}<br>
                        <small>Time: {format_timestamp(alert['timestamp'])}</small>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts for your region. Conditions are within normal parameters.")
    else:
        st.success("✅ No active alerts. Your region is within normal parameters.")
else:
    st.info("📊 No recent data available for alerts.")

st.markdown("---")

# Health Trends Section
st.subheader("📈 Your Health Trends")

if not df_filtered.empty:
    # Daily aggregation for trends
    df_filtered['date'] = df_filtered['timestamp'].dt.date
    daily_stats = df_filtered.groupby('date').agg({
        'health_risk': 'mean',
        'aqi': 'mean',
        'pm25': 'mean',
        'temperature': 'mean',
        'heart_rate_mean': 'mean',
        'spo2_mean': 'mean'
    }).reset_index()
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats['risk_percentage'] = daily_stats['health_risk'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Health Risk Trend
        fig_risk = px.line(
            daily_stats,
            x='date',
            y='risk_percentage',
            title='Your Health Risk Trend',
            labels={'date': 'Date', 'risk_percentage': 'Risk %'},
            markers=True,
            line_shape='spline'
        )
        fig_risk.update_traces(line_color='#ff6b6b', line_width=3)
        fig_risk.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.1, annotation_text="High Risk Zone")
        fig_risk.update_layout(
            hovermode='x unified',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Air Quality Trend
        if 'aqi' in daily_stats.columns:
            fig_aqi = px.line(
                daily_stats,
                x='date',
                y='aqi',
                title='Air Quality Trend',
                labels={'date': 'Date', 'aqi': 'AQI'},
                markers=True,
                line_shape='spline'
            )
            fig_aqi.update_traces(line_color='#4ecdc4', line_width=3)
            fig_aqi.add_hrect(y0=100, y1=200, fillcolor="orange", opacity=0.1, annotation_text="Moderate")
            fig_aqi.add_hrect(y0=200, y1=300, fillcolor="red", opacity=0.1, annotation_text="Unhealthy")
            fig_aqi.update_layout(
                hovermode='x unified',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_aqi, use_container_width=True)

st.markdown("---")

# Environmental Factors
st.subheader("🌡️ Environmental Conditions")

if not df_filtered.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature and Humidity
        if 'temperature' in df_filtered.columns and 'humidity' in df_filtered.columns:
            fig_temp = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature', 'Humidity'),
                vertical_spacing=0.15,
                shared_xaxes=True
            )
            
            df_sorted = df_filtered.sort_values('timestamp')
            
            fig_temp.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['temperature'],
                    name='Temperature',
                    mode='lines',
                    line=dict(color='#ff6b6b', width=2),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            fig_temp.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['humidity'],
                    name='Humidity',
                    mode='lines',
                    line=dict(color='#4a90e2', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig_temp.update_layout(
                height=500,
                title_text="Temperature & Humidity",
                showlegend=True,
                hovermode='x unified'
            )
            fig_temp.update_xaxes(title_text="Time", row=2, col=1)
            fig_temp.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig_temp.update_yaxes(title_text="Humidity (%)", row=2, col=1)
            
            st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # PM2.5 and AQI
        if 'pm25' in df_filtered.columns and 'aqi' in df_filtered.columns:
            fig_pollution = make_subplots(
                rows=2, cols=1,
                subplot_titles=('PM2.5', 'AQI'),
                vertical_spacing=0.15,
                shared_xaxes=True
            )
            
            df_sorted = df_filtered.sort_values('timestamp')
            
            fig_pollution.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['pm25'],
                    name='PM2.5',
                    mode='lines',
                    line=dict(color='#95a5a6', width=2),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            fig_pollution.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['aqi'],
                    name='AQI',
                    mode='lines',
                    line=dict(color='#e74c3c', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig_pollution.update_layout(
                height=500,
                title_text="Pollution Levels",
                showlegend=True,
                hovermode='x unified'
            )
            fig_pollution.update_xaxes(title_text="Time", row=2, col=1)
            fig_pollution.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
            fig_pollution.update_yaxes(title_text="AQI", row=2, col=1)
            
            st.plotly_chart(fig_pollution, use_container_width=True)

st.markdown("---")

# Personal Health Metrics
st.subheader("💓 Personal Health Metrics")

if not df_filtered.empty:
    health_metrics = []
    
    if 'heart_rate_mean' in df_filtered.columns:
        health_metrics.append(('heart_rate_mean', 'Average Heart Rate', 'bpm'))
    if 'spo2_mean' in df_filtered.columns:
        health_metrics.append(('spo2_mean', 'Average SpO2', '%'))
    if 'active_users_count' in df_filtered.columns:
        health_metrics.append(('active_users_count', 'Active Users', 'users'))
    
    if health_metrics:
        daily_health = df_filtered.groupby('date').agg({
            metric[0]: 'mean' for metric in health_metrics
        }).reset_index()
        daily_health['date'] = pd.to_datetime(daily_health['date'])
        
        fig_health = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for idx, (col, label, unit) in enumerate(health_metrics):
            fig_health.add_trace(
                go.Scatter(
                    x=daily_health['date'],
                    y=daily_health[col],
                    name=f"{label} ({unit})",
                    mode='lines+markers',
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=5)
                )
            )
        
        fig_health.update_layout(
            title="Health Metrics Trend",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_health, use_container_width=True)

st.markdown("---")

# Recommendations Section
st.subheader("💡 Health Recommendations")

if latest is not None:
    recommendations = []
    
    # Air quality recommendations
    if 'aqi' in latest:
        aqi = latest['aqi']
        if aqi > 100:
            recommendations.append({
                'icon': '🌬️',
                'title': 'Air Quality Alert',
                'message': f'AQI is {aqi:.0f}. Consider staying indoors or wearing a mask if going outside.',
                'priority': 'high' if aqi > 150 else 'medium'
            })
    
    # PM2.5 recommendations
    if 'pm25' in latest:
        pm25 = latest['pm25']
        if pm25 > 50:
            recommendations.append({
                'icon': '😷',
                'title': 'High PM2.5',
                'message': f'PM2.5 levels are elevated ({pm25:.1f} μg/m³). Reduce outdoor activities.',
                'priority': 'high' if pm25 > 75 else 'medium'
            })
    
    # Temperature recommendations
    if 'temperature' in latest:
        temp = latest['temperature']
        if temp > 35:
            recommendations.append({
                'icon': '☀️',
                'title': 'High Temperature',
                'message': f'Temperature is {temp:.1f}°C. Stay hydrated and avoid prolonged sun exposure.',
                'priority': 'medium'
            })
        elif temp < 5:
            recommendations.append({
                'icon': '❄️',
                'title': 'Low Temperature',
                'message': f'Temperature is {temp:.1f}°C. Dress warmly and avoid prolonged outdoor exposure.',
                'priority': 'medium'
            })
    
    # Health risk recommendations
    if 'health_risk' in latest:
        risk = latest['health_risk']
        if isinstance(risk, (int, float)) and risk > 0.4:
            recommendations.append({
                'icon': '🏥',
                'title': 'Elevated Health Risk',
                'message': 'Health risk indicators are elevated in your area. Monitor your health closely.',
                'priority': 'high' if risk > 0.7 else 'medium'
            })
    
    if recommendations:
        for rec in recommendations:
            priority_class = "alert-high" if rec['priority'] == 'high' else "alert-medium"
            st.markdown(f"""
                <div class="{priority_class}">
                    <strong>{rec['icon']} {rec['title']}</strong><br>
                    {rec['message']}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No specific recommendations. Current conditions are favorable for outdoor activities.")

st.markdown("---")

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>👤 Personal Health Risk Dashboard | Region: {}</p>
        <p>Data from Federated Learning System | MLOps Project</p>
        <p><small>Last updated: {}</small></p>
    </div>
    """.format(selected_region, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)

