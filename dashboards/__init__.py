"""
Dashboards Package
Contains health risk monitoring dashboards for authorities and citizens
"""

__version__ = "1.0.0"

from dashboards.utils import (
    load_node_data,
    load_all_nodes_data,
    get_node_summary,
    get_regional_comparison,
    calculate_risk_trends,
    get_alerts,
    get_risk_category,
    format_timestamp
)

__all__ = [
    'load_node_data',
    'load_all_nodes_data',
    'get_node_summary',
    'get_regional_comparison',
    'calculate_risk_trends',
    'get_alerts',
    'get_risk_category',
    'format_timestamp'
]

