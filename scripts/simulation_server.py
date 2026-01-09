#!/usr/bin/env python3
"""
WebSocket Server for Real-Time MPC Simulation

Wraps the standard SatelliteMPCLinearizedSimulation to ensure exact parity
with the CLI (run_simulation.py).
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# WebSocket library
try:
    import websockets
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite_control.core.simulation import SatelliteMPCLinearizedSimulation
from src.satellite_control.config.simulation_config import SimulationConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy websockets errors from browser prefetch
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("websockets.asyncio.server").setLevel(logging.WARNING)


class SimulationServer:
    """WebSocket server wrapping SatelliteMPCLinearizedSimulation."""

    def __init__(self):
        self.sim: Optional[SatelliteMPCLinearizedSimulation] = None
        self.running = False
        self.paused = False
        self.clients = set()
        self.simulation_task: Optional[asyncio.Task] = None

    def initialize_simulation(self, start_pos=None):
        """Initialize the core simulation backend."""
        logger.info("Initializing SatelliteMPCLinearizedSimulation...")

        # Create default config configuration
        sim_config = SimulationConfig.create_default()

        # Initialize simulation (headless mode)
        self.sim = SatelliteMPCLinearizedSimulation(
            start_pos=start_pos if start_pos else (5.0, 0.0, 0.0),
            simulation_config=sim_config,
            use_mujoco_viewer=False,
        )

        # Manually initialize components usually handled by SimulationLoop
        if not hasattr(self.sim, "context"):
            from src.satellite_control.core.simulation_context import SimulationContext

            self.sim.context = SimulationContext()
            self.sim.context.dt = self.sim.satellite.dt
            self.sim.context.control_dt = self.sim.control_update_interval

        logger.info("Simulation backend initialized")

    async def simulation_loop(self):
        """Main simulation loop driving the backend."""
        if not self.sim:
            self.initialize_simulation()

        logger.info("Starting simulation loop")

        # Timing
        sim_dt = self.sim.satellite.dt
        control_dt = self.sim.control_update_interval
        last_control_time = 0.0

        while self.running:
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            # Check if simulation finished
            if self.sim.simulation_time >= self.sim.max_simulation_time:
                await self.broadcast({"type": "mission_complete", "time": self.sim.simulation_time})
                self.running = False
                break

            # 1. Update Target
            current_state = self.sim.get_current_state()
            self.sim.update_target_state_for_mode(current_state)

            # 2. Control Step (MPC)
            # Run MPC only when enough time has passed
            if self.sim.simulation_time - last_control_time >= control_dt:
                self.sim.update_mpc_control()
                last_control_time = self.sim.simulation_time

                # Broadcast state after control update
                await self.broadcast_state()

            # 3. Apply Delayed Commands
            self.sim.process_command_queue()

            # 4. Physics Step
            self.sim.satellite.simulation_time = self.sim.simulation_time
            self.sim.satellite.update_physics(sim_dt)
            self.sim.simulation_time = self.sim.satellite.simulation_time

            # Simple pacing (could be improved)
            await asyncio.sleep(sim_dt)

    async def broadcast_state(self):
        """Extract state from simulation and send to clients."""
        if not self.clients:
            return

        # Extract State
        # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        state = self.sim.get_current_state()

        pos = state[0:3]
        quat = state[3:7]
        vel = state[7:10]
        ang_vel = state[10:13]

        # Get Control Action (Thrust levels)
        control = self.sim.thruster_actual_output

        # Get Target
        target = self.sim.target_state[0:3]

        await self.broadcast(
            {
                "type": "state",
                "time": self.sim.simulation_time,
                "position": pos.tolist(),
                "quaternion": quat.tolist(),
                "velocity": vel.tolist(),
                "angular_velocity": ang_vel.tolist(),
                "rw_speeds": [0.0, 0.0, 0.0],  # No RWs in this model
                "target": target.tolist(),
                "control": control.tolist(),
                "solve_time": 0.0,  # TODO: extract from mpc controller stats
            }
        )

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        if self.clients:
            msg = json.dumps(message)
            await asyncio.gather(
                *[client.send(msg) for client in self.clients], return_exceptions=True
            )

    async def handle_client(self, websocket):
        """Handle a client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self.clients)}")
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(data, websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total: {len(self.clients)}")

    async def handle_message(self, data: dict, websocket):
        """Handle incoming command."""
        msg_type = data.get("type")

        if msg_type == "start":
            self.initialize_simulation(start_pos=data.get("start_position"))
            self.running = True
            self.paused = False
            if self.simulation_task:
                self.simulation_task.cancel()
            self.simulation_task = asyncio.create_task(self.simulation_loop())

        elif msg_type == "stop":
            self.running = False
            if self.simulation_task:
                self.simulation_task.cancel()
            await self.broadcast({"type": "stopped"})

        elif msg_type == "pause":
            self.paused = True
            await self.broadcast({"type": "paused"})

        elif msg_type == "resume":
            self.paused = False
            await self.broadcast({"type": "resumed"})

    async def run(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on ws://{host}:{port}")

        async def handler(websocket):
            await self.handle_client(websocket)

        async with websockets.serve(handler, host, port):
            await asyncio.Future()  # Run forever


def main():
    server = SimulationServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
