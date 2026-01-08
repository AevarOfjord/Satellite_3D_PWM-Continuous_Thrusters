"""
Streamlit Dashboard for Satellite Control System

V4.0.0: Phase 3 - Advanced Visualization & GUI

Provides a web-based dashboard for real-time monitoring, analysis, and control.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Core imports - must be at top level for Streamlit
from src.satellite_control.config.simulation_config import SimulationConfig
from src.satellite_control.config.io import ConfigIO
from src.satellite_control.visualization.unified_visualizer import UnifiedVisualizationGenerator

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Satellite Control Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_simulation_data(data_path: Path) -> Optional[pd.DataFrame]:
    """
    Load simulation data from CSV file.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with simulation data or None if loading fails
    """
    try:
        csv_path = data_path / "control_data.csv"
        if not csv_path.exists():
            return None
        
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return None


def find_simulation_directories(base_path: Path = Path("Data/Simulation")) -> List[Path]:
    """
    Find all simulation data directories.
    
    Args:
        base_path: Base path to search
        
    Returns:
        List of simulation directory paths
    """
    if not base_path.exists():
        return []
    
    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    # Sort by modification time (newest first)
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs


def create_3d_trajectory_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create 3D trajectory visualization using Plotly.
    
    Args:
        df: DataFrame with simulation data
        
    Returns:
        Plotly figure with 3D trajectory
    """
    fig = go.Figure()
    
    # Extract position data
    if "Current_X" in df.columns and "Current_Y" in df.columns:
        x = df["Current_X"].values
        y = df["Current_Y"].values
        z = df["Current_Z"].values if "Current_Z" in df.columns else np.zeros_like(x)
        
        # Create trajectory trace
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                name="Trajectory",
                line=dict(color="blue", width=4),
                marker=dict(size=3, color=df.index if len(df) > 0 else [0]),
                hovertemplate="<b>Position</b><br>" +
                              "X: %{x:.3f} m<br>" +
                              "Y: %{y:.3f} m<br>" +
                              "Z: %{z:.3f} m<br>" +
                              "<extra></extra>",
            )
        )
        
        # Add start and end markers
        if len(x) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=[x[0]],
                    y=[y[0]],
                    z=[z[0]],
                    mode="markers",
                    name="Start",
                    marker=dict(size=10, color="green", symbol="circle"),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[x[-1]],
                    y=[y[-1]],
                    z=[z[-1]],
                    mode="markers",
                    name="End",
                    marker=dict(size=10, color="red", symbol="circle"),
                )
            )
        
        # Add target positions if available
        if "Target_X" in df.columns:
            target_x = df["Target_X"].values
            target_y = df["Target_Y"].values
            target_z = df["Target_Z"].values if "Target_Z" in df.columns else np.zeros_like(target_x)
            
            fig.add_trace(
                go.Scatter3d(
                    x=target_x,
                    y=target_y,
                    z=target_z,
                    mode="markers",
                    name="Target",
                    marker=dict(size=5, color="orange", symbol="diamond"),
                )
            )
    
    # Update layout
    fig.update_layout(
        title="3D Trajectory",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
    return fig


def create_thruster_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create thruster activity heatmap.
    
    Args:
        df: DataFrame with simulation data
        
    Returns:
        Plotly figure with thruster heatmap
    """
    # Extract thruster data
    thruster_cols = [col for col in df.columns if col.startswith("Thruster_") and col.endswith("_Val")]
    if not thruster_cols:
        # Try Cmd columns if Val not available
        thruster_cols = [col for col in df.columns if col.startswith("Thruster_") and col.endswith("_Cmd")]
    
    if not thruster_cols:
        # Create empty figure
        fig = go.Figure()
        fig.add_annotation(text="No thruster data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sort thruster columns by number
    thruster_cols.sort(key=lambda x: int(x.split("_")[1]))
    
    # Get time axis
    if "Control_Time" in df.columns:
        time = df["Control_Time"].values
    else:
        time = np.arange(len(df))
    
    # Create heatmap data
    heatmap_data = []
    thruster_labels = []
    for col in thruster_cols:
        heatmap_data.append(df[col].values)
        thruster_num = col.split("_")[1]
        thruster_labels.append(f"T{thruster_num}")
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=time,
        y=thruster_labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Thrust Level"),
        hovertemplate="<b>%{y}</b><br>" +
                      "Time: %{x:.2f} s<br>" +
                      "Thrust: %{z:.3f}<br>" +
                      "<extra></extra>",
    ))
    
    fig.update_layout(
        title="Thruster Activity Heatmap",
        xaxis_title="Time (s)",
        yaxis_title="Thruster",
        height=400,
    )
    
    return fig


def calculate_energy_consumption(df: pd.DataFrame, max_thrust: float = 0.1) -> pd.Series:
    """
    Calculate energy consumption from thruster activity.
    
    Simple model: Energy = sum(thrust^2) * dt * efficiency_factor
    
    Args:
        df: DataFrame with simulation data
        max_thrust: Maximum thrust per thruster (N)
        
    Returns:
        Series with cumulative energy consumption
    """
    # Get thruster value columns
    thruster_cols = [col for col in df.columns if col.startswith("Thruster_") and col.endswith("_Val")]
    if not thruster_cols:
        thruster_cols = [col for col in df.columns if col.startswith("Thruster_") and col.endswith("_Cmd")]
    
    if not thruster_cols:
        return pd.Series([0.0] * len(df))
    
    # Calculate dt
    if "Control_Time" in df.columns:
        dt = np.diff(df["Control_Time"].values)
        dt = np.concatenate([[dt[0] if len(dt) > 0 else 0.05], dt])
    else:
        dt = np.ones(len(df)) * 0.05  # Default 50ms
    
    # Calculate power for each timestep (simplified: power ~ thrust^2)
    power = np.zeros(len(df))
    for col in thruster_cols:
        thrust_values = df[col].values * max_thrust  # Convert to actual force
        power += thrust_values ** 2  # Power proportional to thrust squared
    
    # Calculate energy (integrate power)
    energy = np.cumsum(power * dt)
    
    return pd.Series(energy)


def create_energy_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create energy consumption plot.
    
    Args:
        df: DataFrame with simulation data
        
    Returns:
        Plotly figure with energy consumption
    """
    energy = calculate_energy_consumption(df)
    
    if "Control_Time" in df.columns:
        time = df["Control_Time"].values
    else:
        time = np.arange(len(df))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=energy,
        mode="lines",
        name="Cumulative Energy",
        line=dict(color="red", width=2),
        fill="tozeroy",
    ))
    
    fig.update_layout(
        title="Energy Consumption",
        xaxis_title="Time (s)",
        yaxis_title="Energy (arbitrary units)",
        height=400,
    )
    
    return fig


def create_telemetry_plots(df: pd.DataFrame) -> go.Figure:
    """
    Create telemetry plots (position, velocity, orientation).
    
    Args:
        df: DataFrame with simulation data
        
    Returns:
        Plotly figure with subplots
    """
    # Determine time axis
    if "Control_Time" in df.columns:
        time = df["Control_Time"].values
        time_label = "Time (s)"
    elif "Step" in df.columns:
        time = df["Step"].values
        time_label = "Step"
    else:
        time = np.arange(len(df))
        time_label = "Index"
    
    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Position", "Velocity", "Orientation (Euler)"),
        vertical_spacing=0.1,
    )
    
    # Position plot
    if "Current_X" in df.columns:
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_X"], name="X", line=dict(color="red")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_Y"], name="Y", line=dict(color="green")),
            row=1,
            col=1,
        )
        if "Current_Z" in df.columns:
            fig.add_trace(
                go.Scatter(x=time, y=df["Current_Z"], name="Z", line=dict(color="blue")),
                row=1,
                col=1,
            )
    
    # Velocity plot
    if "Current_VX" in df.columns:
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_VX"], name="VX", line=dict(color="red"), showlegend=False),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_VY"], name="VY", line=dict(color="green"), showlegend=False),
            row=2,
            col=1,
        )
        if "Current_VZ" in df.columns:
            fig.add_trace(
                go.Scatter(x=time, y=df["Current_VZ"], name="VZ", line=dict(color="blue"), showlegend=False),
                row=2,
                col=1,
            )
    
    # Orientation plot
    if "Current_Roll" in df.columns:
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_Roll"], name="Roll", line=dict(color="red"), showlegend=False),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_Pitch"], name="Pitch", line=dict(color="green"), showlegend=False),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=df["Current_Yaw"], name="Yaw", line=dict(color="blue"), showlegend=False),
            row=3,
            col=1,
        )
    
    # Update axes
    fig.update_xaxes(title_text=time_label, row=3, col=1)
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (rad)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True)
    
    return fig


def main():
    """Main dashboard function."""
    st.title("🛰️ Satellite Control Dashboard")
    st.markdown("Real-time monitoring and analysis for satellite control simulations")
    
    # Sidebar: Data selection
    st.sidebar.header("Data Selection")
    
    # Find simulation directories
    sim_dirs = find_simulation_directories()
    
    if not sim_dirs:
        st.error("No simulation data found. Run a simulation first.")
        st.info("Data should be in: `Data/Simulation/`")
        return
    
    # Select simulation
    sim_options = [f"{d.name} ({d.stat().st_mtime:.0f})" for d in sim_dirs]
    selected_idx = st.sidebar.selectbox(
        "Select Simulation",
        range(len(sim_options)),
        format_func=lambda x: sim_options[x],
    )
    
    selected_dir = sim_dirs[selected_idx]
    
    # Load data
    df = load_simulation_data(selected_dir)
    
    if df is None or df.empty:
        st.error(f"Failed to load data from {selected_dir}")
        return
    
    # Display basic info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Info")
    st.sidebar.write(f"**Directory:** {selected_dir.name}")
    st.sidebar.write(f"**Data Points:** {len(df)}")
    
    if "Control_Time" in df.columns:
        duration = df["Control_Time"].max() - df["Control_Time"].min()
        st.sidebar.write(f"**Duration:** {duration:.2f} s")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview", 
        "🎯 3D Trajectory", 
        "📈 Telemetry", 
        "⚙️ Configuration",
        "🎯 Mission Progress",
        "📊 Performance Metrics",
        "📈 Historical Analysis",
        "🔥 Enhanced Visualizations",
        "🚀 Run Simulation"
    ])
    
    with tab1:
        st.header("Simulation Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if "Current_X" in df.columns:
                final_x = df["Current_X"].iloc[-1]
                st.metric("Final X Position", f"{final_x:.3f} m")
        
        with col2:
            if "Current_Y" in df.columns:
                final_y = df["Current_Y"].iloc[-1]
                st.metric("Final Y Position", f"{final_y:.3f} m")
        
        with col3:
            if "MPC_Computation_Time" in df.columns:
                avg_solve_time = df["MPC_Computation_Time"].mean()
                st.metric("Avg MPC Time", f"{avg_solve_time*1000:.2f} ms")
        
        with col4:
            if "MPC_Status" in df.columns:
                success_rate = (df["MPC_Status"] == "OPTIMAL").sum() / len(df) * 100
                st.metric("MPC Success Rate", f"{success_rate:.1f}%")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
    
    with tab2:
        st.header("3D Trajectory Visualization")
        fig_3d = create_3d_trajectory_plot(df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab3:
        st.header("Telemetry Data")
        fig_telemetry = create_telemetry_plots(df)
        st.plotly_chart(fig_telemetry, use_container_width=True)
    
    with tab4:
        st.header("Configuration Editor")
        
        # Try to load config from simulation directory
        config_path = selected_dir / "config.yaml"
        if not config_path.exists():
            config_path = selected_dir / "config.json"
        
        if config_path.exists():
            try:
                loaded_config = ConfigIO.load(config_path)
                st.success(f"Loaded configuration from {config_path.name}")
            except Exception as e:
                st.warning(f"Could not load config from {config_path}: {e}")
                loaded_config = SimulationConfig.create_default()
                st.info("Using default configuration")
        else:
            loaded_config = SimulationConfig.create_default()
            st.info("No configuration file found. Using default configuration.")
        
        # Configuration editor
        config_dict = loaded_config.app_config.model_dump()
        
        # Create expandable sections for different config groups
        with st.expander("🔧 Physics Parameters", expanded=False):
            physics = config_dict.get("physics", {})
            
            col1, col2 = st.columns(2)
            with col1:
                mass = st.number_input("Mass (kg)", value=physics.get("mass", 12.0), min_value=0.1, step=0.1)
                inertia_xx = st.number_input("Inertia XX (kg·m²)", value=physics.get("inertia_xx", 0.5), min_value=0.01, step=0.01)
                inertia_yy = st.number_input("Inertia YY (kg·m²)", value=physics.get("inertia_yy", 0.5), min_value=0.01, step=0.01)
                inertia_zz = st.number_input("Inertia ZZ (kg·m²)", value=physics.get("inertia_zz", 0.5), min_value=0.01, step=0.01)
            
            with col2:
                max_thrust = st.number_input("Max Thrust (N)", value=physics.get("max_thrust", 0.1), min_value=0.01, step=0.01)
                thruster_count = st.number_input("Thruster Count", value=physics.get("thruster_count", 8), min_value=1, max_value=32, step=1)
        
        with st.expander("🎯 MPC Parameters", expanded=False):
            mpc = config_dict.get("mpc", {})
            
            col1, col2 = st.columns(2)
            with col1:
                prediction_horizon = st.number_input("Prediction Horizon", value=mpc.get("prediction_horizon", 20), min_value=1, max_value=100, step=1)
                dt = st.number_input("MPC dt (s)", value=mpc.get("dt", 0.05), min_value=0.001, max_value=1.0, step=0.001, format="%.3f")
                max_iterations = st.number_input("Max Iterations", value=mpc.get("max_iterations", 100), min_value=1, max_value=1000, step=1)
            
            with col2:
                position_weight = st.number_input("Position Weight", value=mpc.get("position_weight", 1.0), min_value=0.0, step=0.1)
                velocity_weight = st.number_input("Velocity Weight", value=mpc.get("velocity_weight", 0.1), min_value=0.0, step=0.1)
                control_weight = st.number_input("Control Weight", value=mpc.get("control_weight", 0.01), min_value=0.0, step=0.01)
        
        with st.expander("⏱️ Simulation Parameters", expanded=False):
            sim = config_dict.get("simulation", {})
            
            col1, col2 = st.columns(2)
            with col1:
                control_dt = st.number_input("Control dt (s)", value=sim.get("control_dt", 0.05), min_value=0.001, max_value=1.0, step=0.001, format="%.3f")
                dt = st.number_input("Simulation dt (s)", value=sim.get("dt", 0.01), min_value=0.001, max_value=1.0, step=0.001, format="%.3f")
                max_time = st.number_input("Max Time (s)", value=sim.get("max_time", 300.0), min_value=1.0, step=1.0)
            
            with col2:
                target_hold_time = st.number_input("Target Hold Time (s)", value=sim.get("target_hold_time", 5.0), min_value=0.0, step=0.1)
                default_target_speed = st.number_input("Default Target Speed (m/s)", value=sim.get("default_target_speed", 0.1), min_value=0.01, step=0.01)
        
        # Save configuration
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            save_format = st.selectbox("Save Format", ["YAML", "JSON"])
        
        with col2:
            save_filename = st.text_input("Filename", value="config.yaml" if save_format == "YAML" else "config.json")
        
        if st.button("💾 Save Configuration", type="primary"):
            try:
                # Create updated config
                updated_dict = {
                    "physics": {
                        "mass": mass,
                        "inertia_xx": inertia_xx,
                        "inertia_yy": inertia_yy,
                        "inertia_zz": inertia_zz,
                        "max_thrust": max_thrust,
                        "thruster_count": int(thruster_count),
                    },
                    "mpc": {
                        "prediction_horizon": int(prediction_horizon),
                        "dt": dt,
                        "max_iterations": int(max_iterations),
                        "position_weight": position_weight,
                        "velocity_weight": velocity_weight,
                        "control_weight": control_weight,
                    },
                    "simulation": {
                        "control_dt": control_dt,
                        "dt": dt,
                        "max_time": max_time,
                        "target_hold_time": target_hold_time,
                        "default_target_speed": default_target_speed,
                    },
                }
                
                # Create new config with overrides
                new_config = SimulationConfig.create_with_overrides(updated_dict)
                
                # Save to file
                save_path = selected_dir / save_filename
                ConfigIO.save(new_config, save_path, format="yaml" if save_format == "YAML" else "json")
                st.success(f"✅ Configuration saved to {save_path}")
            except Exception as e:
                st.error(f"❌ Failed to save configuration: {e}")
        
        # Show raw JSON
        with st.expander("📄 Raw Configuration (JSON)", expanded=False):
            st.json(loaded_config.app_config.model_dump())
    
    with tab5:
        st.header("Mission Progress Tracking")
        
        if "Mission_Phase" in df.columns:
            # Mission phase timeline
            phases = df["Mission_Phase"].unique()
            st.subheader("Mission Phases")
            
            phase_counts = df["Mission_Phase"].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Phases", len(phases))
                st.metric("Current Phase", df["Mission_Phase"].iloc[-1] if len(df) > 0 else "N/A")
            
            with col2:
                # Phase distribution chart
                fig_phases = go.Figure(data=[
                    go.Bar(x=phase_counts.index, y=phase_counts.values, name="Phase Count")
                ])
                fig_phases.update_layout(
                    title="Phase Distribution",
                    xaxis_title="Phase",
                    yaxis_title="Count",
                    height=300,
                )
                st.plotly_chart(fig_phases, use_container_width=True)
            
            # Waypoint progress
            if "Waypoint_Number" in df.columns:
                st.subheader("Waypoint Progress")
                waypoints = df["Waypoint_Number"].dropna().unique()
                if len(waypoints) > 0:
                    current_waypoint = df["Waypoint_Number"].iloc[-1] if len(df) > 0 else 0
                    total_waypoints = int(waypoints.max()) if len(waypoints) > 0 else 1
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Waypoint", int(current_waypoint))
                    with col2:
                        st.metric("Total Waypoints", total_waypoints)
                    with col3:
                        progress_pct = (current_waypoint / total_waypoints * 100) if total_waypoints > 0 else 0
                        st.metric("Progress", f"{progress_pct:.1f}%")
                    
                    # Progress bar
                    st.progress(progress_pct / 100)
        else:
            st.info("Mission phase data not available in this simulation.")
    
    with tab6:
        st.header("Performance Metrics")
        
        # MPC Performance
        if "MPC_Computation_Time" in df.columns:
            st.subheader("MPC Solver Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_time = df["MPC_Computation_Time"].mean() * 1000
                st.metric("Avg Solve Time", f"{avg_time:.2f} ms")
            
            with col2:
                max_time = df["MPC_Computation_Time"].max() * 1000
                st.metric("Max Solve Time", f"{max_time:.2f} ms")
            
            with col3:
                if "MPC_Status" in df.columns:
                    optimal_count = (df["MPC_Status"] == "OPTIMAL").sum()
                    success_rate = (optimal_count / len(df)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col4:
                if "MPC_Iterations" in df.columns:
                    avg_iterations = df["MPC_Iterations"].mean()
                    st.metric("Avg Iterations", f"{avg_iterations:.1f}")
            
            # Solve time distribution
            fig_solve = go.Figure()
            solve_times_ms = df["MPC_Computation_Time"].values * 1000
            fig_solve.add_trace(go.Histogram(x=solve_times_ms, nbinsx=50, name="Solve Time"))
            fig_solve.update_layout(
                title="MPC Solve Time Distribution",
                xaxis_title="Time (ms)",
                yaxis_title="Frequency",
                height=400,
            )
            st.plotly_chart(fig_solve, use_container_width=True)
        
        # Control Performance
        st.subheader("Control Performance")
        
        if "Error_X" in df.columns:
            # Position error
            error_x = df["Error_X"].abs()
            error_y = df["Error_Y"].abs()
            error_z = df["Error_Z"].abs() if "Error_Z" in df.columns else pd.Series([0] * len(df))
            
            total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Position Error", f"{total_error.mean():.4f} m")
                st.metric("Max Position Error", f"{total_error.max():.4f} m")
            
            with col2:
                if "Control_Time" in df.columns:
                    time = df["Control_Time"].values
                else:
                    time = np.arange(len(df))
                
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(x=time, y=total_error, name="Total Error", line=dict(color="red")))
                fig_error.update_layout(
                    title="Position Error Over Time",
                    xaxis_title="Time (s)",
                    yaxis_title="Error (m)",
                    height=400,
                )
                st.plotly_chart(fig_error, use_container_width=True)
        
        # Thruster Activity
        if "Total_Active_Thrusters" in df.columns:
            st.subheader("Thruster Activity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_active = df["Total_Active_Thrusters"].mean()
                st.metric("Avg Active Thrusters", f"{avg_active:.1f}")
                max_active = df["Total_Active_Thrusters"].max()
                st.metric("Max Active Thrusters", int(max_active))
            
            with col2:
                if "Control_Time" in df.columns:
                    time = df["Control_Time"].values
                else:
                    time = np.arange(len(df))
                
                fig_thrusters = go.Figure()
                fig_thrusters.add_trace(
                    go.Scatter(
                        x=time,
                        y=df["Total_Active_Thrusters"],
                        name="Active Thrusters",
                        fill="tozeroy",
                        line=dict(color="orange"),
                    )
                )
                fig_thrusters.update_layout(
                    title="Active Thrusters Over Time",
                    xaxis_title="Time (s)",
                    yaxis_title="Number of Active Thrusters",
                    height=400,
                )
                st.plotly_chart(fig_thrusters, use_container_width=True)
    
    with tab7:
        st.header("Historical Data Analysis")
        st.markdown("Compare multiple simulation runs side-by-side")
        
        # Multi-select simulations
        selected_sims = st.multiselect(
            "Select Simulations to Compare",
            options=sim_dirs,
            format_func=lambda x: f"{x.name}",
            default=sim_dirs[:min(5, len(sim_dirs))],  # Default to first 5
        )
        
        if len(selected_sims) < 2:
            st.info("Select at least 2 simulations to compare")
        else:
            # Load all selected simulations
            sim_data = {}
            for sim_dir in selected_sims:
                data = load_simulation_data(sim_dir)
                if data is not None and not data.empty:
                    sim_data[sim_dir.name] = data
            
            if len(sim_data) < 2:
                st.warning("Not enough valid simulations to compare")
            else:
                # Comparison metrics
                st.subheader("Performance Comparison")
                
                # Create comparison table
                comparison_data = []
                for name, df in sim_data.items():
                    metrics = {
                        "Simulation": name,
                        "Duration (s)": df["Control_Time"].max() - df["Control_Time"].min() if "Control_Time" in df.columns else "N/A",
                        "Avg MPC Time (ms)": f"{df['MPC_Computation_Time'].mean() * 1000:.2f}" if "MPC_Computation_Time" in df.columns else "N/A",
                        "MPC Success Rate (%)": f"{(df['MPC_Status'] == 'OPTIMAL').sum() / len(df) * 100:.1f}" if "MPC_Status" in df.columns else "N/A",
                        "Final X (m)": f"{df['Current_X'].iloc[-1]:.3f}" if "Current_X" in df.columns else "N/A",
                        "Final Y (m)": f"{df['Current_Y'].iloc[-1]:.3f}" if "Current_Y" in df.columns else "N/A",
                        "Final Z (m)": f"{df['Current_Z'].iloc[-1]:.3f}" if "Current_Z" in df.columns else "N/A",
                    }
                    comparison_data.append(metrics)
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Trajectory comparison
                st.subheader("Trajectory Comparison")
                
                fig_compare = go.Figure()
                
                colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
                for idx, (name, df) in enumerate(sim_data.items()):
                    if "Current_X" in df.columns and "Current_Y" in df.columns:
                        x = df["Current_X"].values
                        y = df["Current_Y"].values
                        z = df["Current_Z"].values if "Current_Z" in df.columns else np.zeros_like(x)
                        
                        color = colors[idx % len(colors)]
                        fig_compare.add_trace(
                            go.Scatter3d(
                                x=x,
                                y=y,
                                z=z,
                                mode="lines",
                                name=name,
                                line=dict(color=color, width=3),
                            )
                        )
                
                fig_compare.update_layout(
                    title="3D Trajectory Comparison",
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        zaxis_title="Z (m)",
                        aspectmode="data",
                    ),
                    height=600,
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Performance metrics comparison
                st.subheader("MPC Performance Comparison")
                
                if all("MPC_Computation_Time" in df.columns for df in sim_data.values()):
                    fig_perf = go.Figure()
                    
                    for idx, (name, df) in enumerate(sim_data.items()):
                        solve_times = df["MPC_Computation_Time"].values * 1000
                        color = colors[idx % len(colors)]
                        fig_perf.add_trace(
                            go.Box(
                                y=solve_times,
                                name=name,
                                marker_color=color,
                            )
                        )
                    
                    fig_perf.update_layout(
                        title="MPC Solve Time Distribution Comparison",
                        yaxis_title="Time (ms)",
                        height=400,
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                # Position error comparison
                st.subheader("Position Error Comparison")
                
                if all("Error_X" in df.columns for df in sim_data.values()):
                    fig_error = go.Figure()
                    
                    for idx, (name, df) in enumerate(sim_data.items()):
                        error_x = df["Error_X"].abs()
                        error_y = df["Error_Y"].abs()
                        error_z = df["Error_Z"].abs() if "Error_Z" in df.columns else pd.Series([0] * len(df))
                        total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
                        
                        time = df["Control_Time"].values if "Control_Time" in df.columns else np.arange(len(df))
                        color = colors[idx % len(colors)]
                        
                        fig_error.add_trace(
                            go.Scatter(
                                x=time,
                                y=total_error,
                                name=name,
                                line=dict(color=color, width=2),
                            )
                        )
                    
                    fig_error.update_layout(
                        title="Position Error Over Time (Comparison)",
                        xaxis_title="Time (s)",
                        yaxis_title="Error (m)",
                        height=400,
                    )
                    st.plotly_chart(fig_error, use_container_width=True)
    
    with tab8:
        st.header("🔥 Enhanced Visualizations")
        
        # Thruster Activity Heatmap
        st.subheader("Thruster Activity Heatmap")
        if any(col.startswith("Thruster_") for col in df.columns):
            fig_heatmap = create_thruster_heatmap(df)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Thruster statistics
            thruster_cols = [col for col in df.columns if col.startswith("Thruster_") and col.endswith("_Val")]
            if not thruster_cols:
                thruster_cols = [col for col in df.columns if col.startswith("Thruster_") and col.endswith("_Cmd")]
            
            if thruster_cols:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_thruster_usage = sum(df[col].sum() for col in thruster_cols)
                    st.metric("Total Thruster Usage", f"{total_thruster_usage:.2f}")
                with col2:
                    avg_active = sum((df[col] > 0).sum() for col in thruster_cols) / len(thruster_cols) / len(df) * 100
                    st.metric("Avg Active Time", f"{avg_active:.1f}%")
                with col3:
                    max_thruster = max(df[col].max() for col in thruster_cols)
                    st.metric("Max Thrust Level", f"{max_thruster:.3f}")
        else:
            st.info("Thruster data not available in this simulation.")
        
        # Energy Consumption
        st.subheader("Energy Consumption")
        if any(col.startswith("Thruster_") for col in df.columns):
            fig_energy = create_energy_plot(df)
            st.plotly_chart(fig_energy, use_container_width=True)
            
            energy = calculate_energy_consumption(df)
            col1, col2 = st.columns(2)
            with col1:
                total_energy = energy.iloc[-1] if len(energy) > 0 else 0
                st.metric("Total Energy Consumed", f"{total_energy:.2f}")
            with col2:
                if "Control_Time" in df.columns and len(df) > 1:
                    duration = df["Control_Time"].max() - df["Control_Time"].min()
                    avg_power = total_energy / duration if duration > 0 else 0
                    st.metric("Average Power", f"{avg_power:.2f}")
        else:
            st.info("Energy calculation requires thruster data.")
        
        # Interactive Plot Explorer
        st.subheader("Interactive Plot Explorer")
        st.markdown("Create custom plots by selecting data columns. **Plotly plots support zoom, pan, and hover by default!**")
        
        col1, col2 = st.columns(2)
        
        numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64, float, int]]
        
        with col1:
            x_col = st.selectbox(
                "X-axis",
                options=numeric_cols,
                index=numeric_cols.index("Control_Time") if "Control_Time" in numeric_cols else 0,
            )
        
        with col2:
            y_cols = st.multiselect(
                "Y-axis (select multiple)",
                options=[col for col in numeric_cols if col != x_col],
                default=["Current_X", "Current_Y"] if "Current_X" in numeric_cols else [],
            )
        
        if x_col and y_cols:
            fig_custom = go.Figure()
            
            colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
            for idx, y_col in enumerate(y_cols):
                fig_custom.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="lines+markers",
                    name=y_col,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=3, opacity=0.6),
                    hovertemplate=f"<b>{y_col}</b><br>" +
                                 f"{x_col}: %{{x}}<br>" +
                                 f"Value: %{{y}}<br>" +
                                 "<extra></extra>",
                ))
            
            fig_custom.update_layout(
                title=f"Custom Plot: {', '.join(y_cols)} vs {x_col}",
                xaxis_title=x_col,
                yaxis_title="Value",
                height=500,
                hovermode="x unified",
                # Enable zoom and pan
                dragmode="zoom",
            )
            
            # Add zoom/pan controls
            fig_custom.update_xaxes(rangeslider_visible=False)
            fig_custom.update_yaxes(autorange=True)
            
            st.plotly_chart(fig_custom, use_container_width=True, config={
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
                "displaylogo": False,
            })
            
            st.info("💡 **Tip:** Use the toolbar above the plot to zoom, pan, reset axes, and add annotations!")
            
            # Export plot as image
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Plot as PNG"):
                    img_bytes = fig_custom.to_image(format="png", width=1200, height=600)
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name=f"{selected_dir.name}_custom_plot.png",
                        mime="image/png",
                    )
        
        # Obstacle Avoidance Visualization (if data exists)
        if "Obstacle" in str(df.columns) or any("obstacle" in col.lower() for col in df.columns):
            st.subheader("Obstacle Avoidance")
            st.info("Obstacle data detected. Visualization coming soon!")
        else:
            st.subheader("Obstacle Avoidance")
            st.info("No obstacle data available in this simulation.")
    
    with tab9:
        st.header("🚀 Run Simulation")
        st.markdown("Configure and run a new simulation directly from the dashboard")
        
        st.info("⚠️ **Note:** Running a simulation will block the dashboard until completion. For long simulations, consider using the CLI: `satellite-control run`")
        
        # Mission selection
        st.subheader("Mission Configuration")
        
        from src.satellite_control.mission.plugin import get_registry
        registry = get_registry()
        available_plugins = registry.list_plugins()
        
        mission_type = st.selectbox(
            "Mission Type",
            options=["waypoint", "shape_following"] + available_plugins,
            index=0,
            help="Select the type of mission to run"
        )
        
        # Basic parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Start Position")
            start_x = st.number_input("Start X (m)", value=1.0, step=0.1)
            start_y = st.number_input("Start Y (m)", value=1.0, step=0.1)
            start_z = st.number_input("Start Z (m)", value=0.0, step=0.1)
        
        with col2:
            st.subheader("Target Position")
            target_x = st.number_input("Target X (m)", value=0.0, step=0.1)
            target_y = st.number_input("Target Y (m)", value=0.0, step=0.1)
            target_z = st.number_input("Target Z (m)", value=0.0, step=0.1)
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_duration = st.number_input("Max Duration (s)", value=300.0, min_value=1.0, step=10.0)
        
        with col2:
            control_dt = st.number_input("Control dt (s)", value=0.05, min_value=0.001, max_value=1.0, step=0.001, format="%.3f")
        
        with col3:
            prediction_horizon = st.number_input("MPC Horizon", value=20, min_value=1, max_value=100, step=1)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                use_preset = st.checkbox("Use Configuration Preset", value=False)
                preset_name = None
                if use_preset:
                    from src.satellite_control.config.presets import list_presets
                    presets = list_presets()
                    preset_name = st.selectbox("Preset", options=presets)
            
            with col2:
                auto_generate_viz = st.checkbox("Auto-generate visualizations", value=True)
                headless_mode = st.checkbox("Headless mode (no animation)", value=True)
        
        # Run button
        st.markdown("---")
        
        if st.button("🚀 Run Simulation", type="primary", width="stretch"):
            try:
                with st.spinner("Initializing simulation..."):
                    # Create simulation config
                    from src.satellite_control.core.simulation import SatelliteMPCLinearizedSimulation
                    
                    # Create base config
                    if use_preset and preset_name:
                        from src.satellite_control.config.presets import load_preset
                        preset_dict = load_preset(preset_name)
                        config = SimulationConfig.create_with_overrides(preset_dict)
                    else:
                        config = SimulationConfig.create_default()
                    
                    # Apply parameter overrides
                    config_overrides = {
                        "simulation": {
                            "max_duration": max_duration,
                            "control_dt": control_dt,
                        },
                        "mpc": {
                            "prediction_horizon": prediction_horizon,
                        }
                    }
                    config = SimulationConfig.create_with_overrides(config_overrides, base_config=config)
                    
                    # Create simulation
                    sim = SatelliteMPCLinearizedSimulation(
                        start_pos=(start_x, start_y, start_z),
                        target_pos=(target_x, target_y, target_z),
                        start_angle=(0.0, 0.0, 0.0),
                        target_angle=(0.0, 0.0, 0.0),
                        simulation_config=config,
                    )
                    
                    # Override max duration if needed
                    sim.max_simulation_time = max_duration
                    
                    st.success("✅ Simulation initialized!")
                    st.info("Running simulation... This may take a while.")
                    
                    # Run simulation (headless)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run simulation
                    sim.run_simulation(show_animation=not headless_mode)
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Simulation completed!")
                    
                    # Close simulation
                    sim.close()
                    
                    # Auto-generate visualizations if requested
                    if auto_generate_viz:
                        st.info("Generating visualizations...")
                        try:
                            from src.satellite_control.visualization.unified_visualizer import UnifiedVisualizationGenerator
                            
                            # Find the most recent simulation directory
                            sim_dirs = find_simulation_directories()
                            if sim_dirs:
                                latest_dir = sim_dirs[0]  # Most recent
                                
                                generator = UnifiedVisualizationGenerator(
                                    data_directory=str(latest_dir.parent),
                                    prefer_pandas=True,
                                )
                                generator.csv_path = latest_dir / "control_data.csv"
                                generator.output_dir = latest_dir
                                generator.load_csv_data()
                                generator.generate_performance_plots()
                                
                                st.success(f"✅ Visualizations generated in {latest_dir}")
                        except Exception as viz_err:
                            st.warning(f"⚠️ Visualization generation failed: {viz_err}")
                    
                    st.balloons()
                    st.success("🎉 Simulation completed successfully!")
                    
                    # Add refresh button
                    if st.button("🔄 Refresh Simulation List", use_container_width=True):
                        st.rerun()
                    
                    st.info("💡 The new simulation should appear in the sidebar. If not, refresh the page.")
                    
            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Export section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export")
    
    if st.sidebar.button("Export Data as CSV"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_dir.name}_data.csv",
            mime="text/csv",
        )
    
    # Video export option
    st.sidebar.markdown("---")
    st.sidebar.subheader("Video Export")
    
    if st.sidebar.button("Generate Animation Video"):
        with st.sidebar:
            with st.spinner("Generating video... This may take a while."):
                try:
                    # Use UnifiedVisualizationGenerator to create video
                    generator = UnifiedVisualizationGenerator(
                        data_directory=str(selected_dir.parent),
                        prefer_pandas=True,
                    )
                    
                    # Override paths to use current simulation data
                    generator.csv_path = selected_dir / "control_data.csv"
                    generator.output_dir = selected_dir
                    
                    # Load data
                    generator.load_csv_data()
                    
                    # Generate animation
                    generator.generate_animation()
                    
                    st.success("✅ Video generated successfully!")
                    st.info(f"Check: {selected_dir / 'Simulation_animation.mp4'}")
                except Exception as e:
                    st.error(f"❌ Video generation failed: {e}")
                    st.info("You can generate videos using the CLI: `satellite-control visualize`")


if __name__ == "__main__":
    main()
