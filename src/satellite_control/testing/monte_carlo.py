"""
Monte Carlo Testing Framework

Runs N simulations with random initial conditions to statistically
analyze controller performance, success rates, and robustness.

Usage:
    python -m src.satellite_control.testing.monte_carlo --trials 100
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrialConfig:
    """Configuration for a single Monte Carlo trial."""

    trial_id: int
    initial_x: float
    initial_y: float
    initial_vx: float
    initial_vy: float
    initial_theta: float
    initial_omega: float
    target_x: float = 0.0
    target_y: float = 0.0
    target_theta: float = 0.0


@dataclass
class TrialResult:
    """Results from a single Monte Carlo trial."""

    trial_id: int
    success: bool
    final_position_error: float
    final_angle_error: float
    time_to_target: float
    total_fuel_usage: float
    mean_solve_time: float
    max_solve_time: float
    failure_reason: Optional[str] = None
    config: Optional[Dict] = None


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo analysis."""

    n_trials: int = 100
    max_simulation_time: float = 60.0

    # Initial position variation (meters)
    pos_mean: float = 1.0
    pos_std: float = 0.3

    # Initial velocity variation (m/s)
    vel_mean: float = 0.0
    vel_std: float = 0.02

    # Initial angle variation (radians)
    angle_mean: float = 0.0
    angle_std: float = 0.5  # ~30 degrees

    # Angular velocity variation (rad/s)
    omega_std: float = 0.01

    # Target position
    target_x: float = 0.0
    target_y: float = 0.0
    target_theta: float = 0.0

    # Success criteria
    position_tolerance: float = 0.05  # meters
    angle_tolerance: float = 0.05  # radians (~3 degrees)

    # Random seed for reproducibility
    seed: Optional[int] = None


class MonteCarloRunner:
    """
    Executes Monte Carlo trials for satellite control system.

    Generates random initial conditions, runs simulations, and collects
    statistics for performance analysis.
    """

    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo runner.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config
        self.results: List[TrialResult] = []
        self.rng = np.random.default_rng(config.seed)

    def generate_trial_configs(self) -> List[TrialConfig]:
        """Generate random trial configurations."""
        configs = []

        for i in range(self.config.n_trials):
            # Random initial position (distance from origin)
            r = abs(self.rng.normal(self.config.pos_mean, self.config.pos_std))
            angle = self.rng.uniform(0, 2 * np.pi)
            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Random initial velocity
            vx = self.rng.normal(self.config.vel_mean, self.config.vel_std)
            vy = self.rng.normal(self.config.vel_mean, self.config.vel_std)

            # Random initial orientation
            theta = self.rng.normal(self.config.angle_mean, self.config.angle_std)

            # Random angular velocity
            omega = self.rng.normal(0, self.config.omega_std)

            configs.append(
                TrialConfig(
                    trial_id=i,
                    initial_x=x,
                    initial_y=y,
                    initial_vx=vx,
                    initial_vy=vy,
                    initial_theta=theta,
                    initial_omega=omega,
                    target_x=self.config.target_x,
                    target_y=self.config.target_y,
                    target_theta=self.config.target_theta,
                )
            )

        return configs

    def run_single_trial(self, trial_config: TrialConfig) -> TrialResult:
        """
        Run a single simulation trial.

        Args:
            trial_config: Trial configuration

        Returns:
            TrialResult with success/failure and metrics
        """
        from src.satellite_control.core.simulation import (
            SatelliteMPCLinearizedSimulation,
        )

        try:
            # Create simulation with trial initial conditions
            sim = SatelliteMPCLinearizedSimulation(
                start_pos=(trial_config.initial_x, trial_config.initial_y),
                target_pos=(trial_config.target_x, trial_config.target_y),
                start_angle=(0.0, 0.0, trial_config.initial_theta),
                target_angle=(0.0, 0.0, trial_config.target_theta),
                start_vx=trial_config.initial_vx,
                start_vy=trial_config.initial_vy,
                start_omega=trial_config.initial_omega,
                use_mujoco_viewer=False,  # Headless mode
            )

            # Set mission parameters for point-to-point
            # Override max simulation time
            if hasattr(sim, "max_simulation_time"):
                sim.max_simulation_time = self.config.max_simulation_time
            if hasattr(sim, "max_time"):
                sim.max_time = self.config.max_simulation_time

            # Run the simulation in headless mode
            start_time = time.time()
            # Disable auto-visualization to prevent memory issues and speed up trials
            sim.auto_generate_visualizations = lambda: None  # type: ignore[method-assign]
            sim.run_simulation(show_animation=False)
            wall_time = time.time() - start_time

            # Extract final results
            final_state = sim.get_current_state()
            pos_error = np.sqrt(
                (final_state[0] - trial_config.target_x) ** 2
                + (final_state[1] - trial_config.target_y) ** 2
            )
            from src.satellite_control.utils.orientation_utils import (
                euler_xyz_to_quat_wxyz,
                quat_angle_error,
            )

            target_quat = euler_xyz_to_quat_wxyz((0.0, 0.0, trial_config.target_theta))
            angle_error = quat_angle_error(target_quat, final_state[3:7])

            # Check success
            success = (
                pos_error < self.config.position_tolerance
                and angle_error < self.config.angle_tolerance
            )

            # Get solver stats if available
            solve_times = getattr(sim, "mpc_solve_times", [])
            mean_solve = np.mean(solve_times) if len(solve_times) > 0 else 0.0
            max_solve = np.max(solve_times) if len(solve_times) > 0 else 0.0

            # Get fuel usage if available
            fuel_usage = getattr(sim, "total_thruster_usage", 0.0)

            # Get simulation time
            sim_time = getattr(sim, "simulation_time", wall_time)

            # Cleanup
            sim.close()

            return TrialResult(
                trial_id=trial_config.trial_id,
                success=success,
                final_position_error=pos_error,
                final_angle_error=angle_error,
                time_to_target=sim_time,
                total_fuel_usage=fuel_usage,
                mean_solve_time=mean_solve,
                max_solve_time=max_solve,
                config=asdict(trial_config),
            )

        except Exception as e:
            logger.error(f"Trial {trial_config.trial_id} failed: {e}")
            print(f"Trial {trial_config.trial_id} failed: {e}")
            return TrialResult(
                trial_id=trial_config.trial_id,
                success=False,
                final_position_error=float("inf"),
                final_angle_error=float("inf"),
                time_to_target=float("inf"),
                total_fuel_usage=0.0,
                mean_solve_time=0.0,
                max_solve_time=0.0,
                failure_reason=str(e),
                config=asdict(trial_config),
            )

    def run_trials(self, parallel: bool = False, n_workers: int = 4) -> None:
        """
        Run all Monte Carlo trials.

        Args:
            parallel: Use parallel processing
            n_workers: Number of parallel workers
        """
        configs = self.generate_trial_configs()
        self.results = []

        print(f"\n{'='*60}")
        print(f"MONTE CARLO ANALYSIS: {self.config.n_trials} Trials")
        print(f"{'='*60}\n")

        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self.run_single_trial, cfg): cfg for cfg in configs}
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)
                    status = "✓" if result.success else "✗"
                    print(
                        f"Trial {i+1}/{self.config.n_trials}: {status} "
                        f"(err={result.final_position_error:.4f}m)"
                    )
        else:
            for i, cfg in enumerate(configs):
                result = self.run_single_trial(cfg)
                self.results.append(result)
                status = "✓" if result.success else "✗"
                print(
                    f"Trial {i+1}/{self.config.n_trials}: {status} "
                    f"(err={result.final_position_error:.4f}m)"
                )

    def analyze_results(self) -> Dict:
        """
        Analyze Monte Carlo results.

        Returns:
            Dictionary with statistical summary
        """
        if not self.results:
            return {}

        successes = [r for r in self.results if r.success]
        failures = [r for r in self.results if not r.success]

        pos_errors = [r.final_position_error for r in self.results if r.success]
        angle_errors = [r.final_angle_error for r in self.results if r.success]
        times = [r.time_to_target for r in self.results if r.success]
        fuels = [r.total_fuel_usage for r in self.results if r.success]
        solve_times = [r.mean_solve_time for r in self.results if r.success]

        analysis = {
            "total_trials": len(self.results),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(self.results) * 100,
            "position_error": {
                "mean": np.mean(pos_errors) if pos_errors else 0,
                "std": np.std(pos_errors) if pos_errors else 0,
                "max": np.max(pos_errors) if pos_errors else 0,
                "min": np.min(pos_errors) if pos_errors else 0,
            },
            "angle_error": {
                "mean": np.mean(angle_errors) if angle_errors else 0,
                "std": np.std(angle_errors) if angle_errors else 0,
            },
            "time_to_target": {
                "mean": np.mean(times) if times else 0,
                "std": np.std(times) if times else 0,
            },
            "fuel_usage": {
                "mean": np.mean(fuels) if fuels else 0,
                "std": np.std(fuels) if fuels else 0,
            },
            "solve_time_ms": {
                "mean": np.mean(solve_times) * 1000 if solve_times else 0,
                "std": np.std(solve_times) * 1000 if solve_times else 0,
            },
        }

        return analysis

    def print_summary(self) -> None:
        """Print analysis summary to console."""
        analysis = self.analyze_results()

        print(f"\n{'='*60}")
        print("MONTE CARLO RESULTS SUMMARY")
        print(f"{'='*60}\n")

        print(f"Total Trials:    {analysis['total_trials']}")
        print(f"Successes:       {analysis['successes']}")
        print(f"Failures:        {analysis['failures']}")
        print(f"Success Rate:    {analysis['success_rate']:.1f}%\n")

        print("Position Error (m):")
        pe = analysis["position_error"]
        print(f"  Mean: {pe['mean']:.4f} ± {pe['std']:.4f}")
        print(f"  Range: [{pe['min']:.4f}, {pe['max']:.4f}]\n")

        print("Time to Target (s):")
        tt = analysis["time_to_target"]
        print(f"  Mean: {tt['mean']:.2f} ± {tt['std']:.2f}\n")

        print("Fuel Usage (thruster-seconds):")
        fu = analysis["fuel_usage"]
        print(f"  Mean: {fu['mean']:.2f} ± {fu['std']:.2f}\n")

        print("MPC Solve Time (ms):")
        st = analysis["solve_time_ms"]
        print(f"  Mean: {st['mean']:.2f} ± {st['std']:.2f}\n")

    def save_results(self, output_dir: str = "Data/MonteCarlo") -> str:
        """
        Save results to files.

        Args:
            output_dir: Output directory

        Returns:
            Path to results directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(output_dir) / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results as CSV
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(results_dir / "results.csv", index=False)

        # Save analysis as JSON
        analysis = self.analyze_results()
        with open(results_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Save config
        with open(results_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"\nResults saved to: {results_dir}")
        return str(results_dir)

    def generate_plots(self, output_dir: str) -> None:
        """Generate analysis plots."""
        import matplotlib.pyplot as plt

        if not self.results:
            return

        output_path = Path(output_dir)

        # 1. Success rate pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        successes = sum(1 for r in self.results if r.success)
        failures = len(self.results) - successes
        ax.pie(
            [successes, failures],
            labels=["Success", "Failure"],
            colors=["#2ca02c", "#d62728"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"Monte Carlo Success Rate (N={len(self.results)})")
        plt.savefig(output_path / "success_rate.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Position error histogram
        pos_errors = [r.final_position_error for r in self.results if r.success]
        if pos_errors:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(pos_errors, bins=20, color="#1f77b4", edgecolor="black", alpha=0.7)
            ax.axvline(float(np.mean(pos_errors)), color="red", linestyle="--", label="Mean")
            ax.set_xlabel("Final Position Error (m)")
            ax.set_ylabel("Count")
            ax.set_title("Position Error Distribution")
            ax.legend()
            plt.savefig(output_path / "position_error_hist.png", dpi=150, bbox_inches="tight")
            plt.close()

        # 3. Initial position scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        for r in self.results:
            if r.config:
                color = "#2ca02c" if r.success else "#d62728"
                marker = "o" if r.success else "x"
                ax.scatter(
                    r.config["initial_x"],
                    r.config["initial_y"],
                    c=color,
                    marker=marker,
                    alpha=0.6,
                )
        ax.scatter(0, 0, c="blue", marker="*", s=200, label="Target")
        ax.set_xlabel("Initial X (m)")
        ax.set_ylabel("Initial Y (m)")
        ax.set_title("Initial Positions (Green=Success, Red=Failure)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        plt.savefig(output_path / "initial_positions.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Plots saved to: {output_path}")


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo Testing")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-time", type=float, default=60.0, help="Max simulation time")
    parser.add_argument("--parallel", action="store_true", help="Use parallel")
    args = parser.parse_args()

    config = MonteCarloConfig(
        n_trials=args.trials,
        max_simulation_time=args.max_time,
        seed=args.seed,
    )

    runner = MonteCarloRunner(config)
    runner.run_trials(parallel=args.parallel)
    runner.print_summary()

    output_dir = runner.save_results()
    runner.generate_plots(output_dir)


if __name__ == "__main__":
    main()
