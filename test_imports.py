#!/usr/bin/env python3
"""Quick test for dashboard builder"""

print("Testing imports...")

print("1. Plotly...")
import plotly.graph_objects as go
print("  ✓ plotly.graph_objects imported")

import plotly.subplots
print("  ✓ plotly.subplots imported")

print("2. UMAP...")
from umap import UMAP
print("  ✓ UMAP imported")

print("3. Dashboard...")
from src.dashboard_builder import DashboardBuilder
print("  ✓ DashboardBuilder imported")

print("\n✓ All imports OK!")
