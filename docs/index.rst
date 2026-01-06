Satellite Thruster Control System
=================================

Model Predictive Control system for satellite thruster control with MuJoCo physics simulation.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   ARCHITECTURE
   DEVELOPMENT_GUIDE
   TESTING_GUIDE
   TROUBLESHOOTING

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Documentation

   SIMULATION
   VISUALIZATION
   PHYSICS_ENGINE
   MATHEMATICS


Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Clone repository
   git clone https://github.com/AevarOfjord/SatelliteProject
   cd SatelliteProject

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -e ".[dev]"

Running a Simulation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python run_simulation.py


Key Components
--------------

**MPC Controllers**
   - ``PWMMPC``: Continuous PWM controller with L1 fuel-optimal mode
   - ``BinaryMPC``: Binary (on/off) thruster controller

**Simulation**
   - ``SatelliteMPCLinearizedSimulation``: Main simulation class
   - ``ThrusterManager``: Thruster valve physics

**Testing**
   - ``MonteCarloRunner``: Statistical analysis framework


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
