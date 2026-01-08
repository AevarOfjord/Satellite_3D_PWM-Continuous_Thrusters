# Aerospace Simulators: Complete Market Analysis
## What Software Is Used to Simulate Satellite Missions?

**Understanding the competitive landscape to position your product correctly**

---

## Table of Contents

1. [Commercial Enterprise Simulators](#1-commercial-enterprise-simulators)
2. [NASA & Open Source Tools](#2-nasa--open-source-tools)
3. [Specialized Mission Analysis Tools](#3-specialized-mission-analysis-tools)
4. [Control-Specific Simulators](#4-control-specific-simulators)
5. [Python/Research Tools](#5-pythonresearch-tools)
6. [Comparison Matrix](#6-comparison-matrix)
7. [Where Your Product Fits](#7-where-your-product-fits)

---

## 1. Commercial Enterprise Simulators

### **1.1 AGI STK (Systems Tool Kit)**
**Company:** Ansys (formerly Analytical Graphics, Inc.)  
**Price:** $50,000 - $500,000/year  
**Market:** Enterprise aerospace, government, large satellite operators

**What It Does:**
- Comprehensive mission analysis and visualization
- Orbit propagation and planning
- Communication link analysis
- Coverage analysis
- Visualization (2D/3D)
- Sensor modeling
- Launch vehicle integration

**Capabilities:**
- ✅ Orbit propagation (high-fidelity)
- ✅ Mission planning
- ✅ Visualization (excellent 3D graphics)
- ✅ Communication analysis
- ✅ Coverage analysis
- ✅ Ground station tracking
- ❌ Limited control system design (basic ADCS)
- ❌ Not tunable for control (black box)
- ❌ Complex setup (days to weeks)

**Used By:**
- Lockheed Martin
- Boeing
- Northrop Grumman
- NASA centers
- US Air Force
- International space agencies

**Strengths:**
- Industry standard (everyone knows it)
- Comprehensive (everything in one tool)
- Professional visualization
- Excellent documentation
- Enterprise support

**Weaknesses:**
- Expensive ($50K-$500K/year)
- Complex (steep learning curve)
- Not for control system design
- Windows-only (mostly)
- Not tunable (black box)

---

### **1.2 FreeFlyer**
**Company:** a.i. solutions  
**Price:** $30,000 - $200,000/year  
**Market:** Mission design, satellite operations, government

**What It Does:**
- Satellite mission design and analysis
- Orbit propagation
- Maneuver planning
- Mission visualization
- Operations planning
- Ground station scheduling

**Capabilities:**
- ✅ Mission design (specialized)
- ✅ Orbit propagation
- ✅ Maneuver planning
- ✅ Visualization
- ✅ Operations planning
- ❌ Basic control (PID only, not MPC)
- ❌ Not tunable
- ❌ Complex scripting language

**Used By:**
- NASA Goddard
- Air Force Research Lab
- Commercial satellite operators
- International Space Station operations

**Strengths:**
- Mission design focus (better than STK for some tasks)
- Good maneuver planning
- Used by NASA (credibility)
- Python API (limited)

**Weaknesses:**
- Expensive ($30K-$200K/year)
- Limited control capabilities
- Not for control system design
- Complex setup
- Windows-only

---

### **1.3 ASTOS**
**Company:** Astos Solutions (German)  
**Price:** $50,000 - $200,000/year (licenses)  
**Market:** European Space Agency, European aerospace industry

**What It Does:**
- Mission analysis and trajectory optimization
- Launch vehicle design
- Re-entry analysis
- Propulsion system design
- End-to-end mission design

**Capabilities:**
- ✅ Trajectory optimization (specialized)
- ✅ Launch vehicle analysis
- ✅ Re-entry analysis
- ✅ Mission design
- ❌ Not for control system design
- ❌ Expensive
- ❌ Europe-focused

**Used By:**
- European Space Agency (ESA)
- European aerospace companies
- Research institutions (Europe)

**Strengths:**
- Trajectory optimization (excellent)
- End-to-end mission design
- Used by ESA (credibility)

**Weaknesses:**
- Expensive
- Not for control design
- Europe-focused
- Limited adoption in US

---

## 2. NASA & Open Source Tools

### **2.1 GMAT (General Mission Analysis Tool)**
**Developer:** NASA Goddard + Private Industry  
**Price:** Free (open source)  
**Market:** Research, education, NASA missions

**What It Does:**
- Space mission analysis
- Orbit propagation
- Trajectory optimization
- Mission planning
- Visualization

**Capabilities:**
- ✅ Orbit propagation (high-fidelity)
- ✅ Trajectory optimization
- ✅ Mission planning
- ✅ Visualization (basic)
- ✅ Free (open source)
- ❌ Complex setup (weeks)
- ❌ Command-line interface (CLI)
- ❌ Poor documentation
- ❌ Limited control system design
- ❌ No MPC control

**Used By:**
- NASA missions (LCROSS, OSIRIS-REx, LRO)
- Research institutions
- Universities
- Students (learning)

**Strengths:**
- Free (no cost)
- NASA-developed (credibility)
- Used in real missions
- Open source (can modify)
- High-fidelity orbit propagation

**Weaknesses:**
- Complex (weeks to learn)
- CLI-based (no GUI)
- Poor documentation
- Limited control capabilities
- No commercial support
- Not tunable

**Languages:** C++ (core), MATLAB/Python (limited bindings)

---

### **2.2 Basilisk (BSK)**
**Developer:** University of Colorado Boulder  
**Price:** Free (open source)  
**Market:** Research, CubeSat development, education

**What It Does:**
- Multi-body dynamics simulation
- Attitude determination and control (ADCS)
- Flight software framework
- Mission simulation
- Hardware-in-the-loop (HIL) testing

**Capabilities:**
- ✅ Multi-body dynamics
- ✅ ADCS simulation (good)
- ✅ Flight software framework
- ✅ HIL support
- ✅ Free (open source)
- ✅ Modular (C++/Python)
- ❌ Complex setup (weeks)
- ❌ Requires C++/Python programming
- ❌ Limited MPC support
- ❌ Poor documentation
- ❌ Steep learning curve

**Used By:**
- University of Colorado
- CubeSat developers (research)
- NASA centers (some)
- Research labs

**Strengths:**
- Free (no cost)
- Good ADCS capabilities
- Modular architecture
- HIL support
- Open source (can modify)

**Weaknesses:**
- Complex (weeks to set up)
- Requires programming (C++/Python)
- Poor documentation
- Limited MPC
- No commercial support
- Not user-friendly

**Languages:** C++ (core), Python (bindings)

---

### **2.3 OreSat / PySquared**
**Developer:** Portland State University / Other Universities  
**Price:** Free (open source)  
**Market:** CubeSat development, education

**What It Does:**
- CubeSat flight software framework
- Mission simulation (basic)
- Hardware integration
- Python-based (easy for students)

**Capabilities:**
- ✅ Python-based (accessible)
- ✅ CubeSat-focused
- ✅ Free (open source)
- ✅ Educational (good for learning)
- ❌ Basic simulation only
- ❌ Limited control capabilities
- ❌ Not comprehensive
- ❌ No MPC
- ❌ Student project (less mature)

**Used By:**
- University students
- CubeSat teams (education)
- Research projects (small)

**Strengths:**
- Python-based (accessible)
- Educational (good for learning)
- Free (no cost)

**Weaknesses:**
- Basic capabilities
- Not comprehensive
- Limited control
- Student project (less mature)

---

## 3. Specialized Mission Analysis Tools

### **3.1 Orbiter Space Flight Simulator**
**Developer:** Martin Schweiger  
**Price:** Free  
**Market:** Hobbyists, education, enthusiasts

**What It Does:**
- Space flight simulation
- Realistic physics
- Interactive 3D visualization
- Educational tool

**Capabilities:**
- ✅ Realistic physics (good)
- ✅ 3D visualization (excellent)
- ✅ Interactive (playable)
- ✅ Free
- ❌ Not for engineering analysis
- ❌ No mission planning tools
- ❌ Not for control design
- ❌ Gaming-focused (not professional)

**Used By:**
- Space enthusiasts
- Students (learning)
- Hobbyists
- Educators (visualization)

**Strengths:**
- Realistic physics
- Great visualization
- Interactive
- Free

**Weaknesses:**
- Not for professional use
- No engineering tools
- Gaming-focused
- Not for control design

---

### **3.2 FlightGear**
**Developer:** FlightGear Project  
**Price:** Free (open source)  
**Market:** Aviation, atmospheric flight

**What It Does:**
- Atmospheric flight simulation
- Realistic aerodynamics
- 3D visualization
- Some orbital capability (limited)

**Capabilities:**
- ✅ Atmospheric flight (excellent)
- ✅ Realistic aerodynamics
- ✅ 3D visualization
- ✅ Free
- ❌ Limited orbital capability
- ❌ Not for satellite missions
- ❌ Atmospheric focus
- ❌ Not for control design

**Used By:**
- Aviation enthusiasts
- Flight training (some)
- Research (atmospheric)

**Not Relevant For:**
- Satellite missions
- Control system design
- Orbital mechanics

---

### **3.3 COSMOS**
**Developer:** Ball Aerospace (open source)  
**Price:** Free (open source)  
**Market:** Small satellite operations, CubeSats

**What It Does:**
- Mission operations framework
- Command and control
- Telemetry processing
- Ground station integration

**Capabilities:**
- ✅ Mission operations (specialized)
- ✅ Command/control
- ✅ Telemetry processing
- ✅ Ground station integration
- ✅ Free (open source)
- ❌ Not for simulation
- ❌ Operations-focused (not design)
- ❌ Not for control design
- ❌ Complex setup

**Used By:**
- CubeSat missions
- Small satellite operators
- University CubeSat programs

**Strengths:**
- Operations-focused (good for mission ops)
- Free (no cost)
- Used by CubeSats

**Weaknesses:**
- Not for simulation
- Not for control design
- Operations tool (not design tool)

---

## 4. Control-Specific Simulators

### **4.1 MATLAB/Simulink Aerospace Toolbox**
**Company:** MathWorks  
**Price:** $5,000 - $10,000/year (toolbox) + MATLAB license  
**Market:** Research, education, industry

**What It Does:**
- Control system design (Simulink)
- Orbit propagation (Aerospace Toolbox)
- ADCS simulation
- Flight dynamics
- Mission analysis (basic)

**Capabilities:**
- ✅ Control system design (excellent)
- ✅ Simulink (visual programming)
- ✅ ADCS simulation
- ✅ MATLAB integration (data analysis)
- ✅ Orbit propagation (basic)
- ❌ Expensive (MATLAB + toolboxes)
- ❌ Not specialized for satellites
- ❌ Limited MPC support
- ❌ Requires MATLAB knowledge
- ❌ Not integrated (separate tools)

**Used By:**
- Research labs
- Universities (teaching)
- Industry (some)
- Control engineers

**Strengths:**
- Control design (excellent)
- Simulink (visual)
- MATLAB integration
- Widely used

**Weaknesses:**
- Expensive ($5K-$10K/year)
- Not specialized
- Requires MATLAB
- Limited MPC

---

### **4.2 Poliastro**
**Developer:** Open source community  
**Price:** Free  
**Market:** Python users, research, education

**What It Does:**
- Orbital mechanics (Python)
- Orbit propagation
- Mission analysis (basic)
- Python library (programmatic)

**Capabilities:**
- ✅ Python-based (accessible)
- ✅ Orbital mechanics (good)
- ✅ Free
- ✅ Easy to use
- ❌ No control system design
- ❌ No ADCS
- ❌ Limited visualization
- ❌ Basic only

**Used By:**
- Python users
- Students
- Research labs (simple missions)

**Strengths:**
- Python (accessible)
- Free
- Easy to use

**Weaknesses:**
- No control design
- Basic only
- Limited capabilities

---

## 5. Python/Research Tools

### **5.1 Skyfield**
**Developer:** Brandon Rhodes  
**Price:** Free  
**Market:** Python users, hobbyists, research

**What It Does:**
- Astronomical calculations
- Planet positions
- Ephemeris calculations
- Python library

**Capabilities:**
- ✅ Python-based
- ✅ Astronomical calculations
- ✅ Free
- ❌ Not for satellite missions
- ❌ Not for control design
- ❌ Limited to astronomy

**Not Relevant For:**
- Satellite control design
- Mission simulation
- ADCS design

---

### **5.2 OreKit**
**Developer:** CS Group / Open source  
**Price:** Free (open source)  
**Market:** Research, Java/Python users

**What It Does:**
- Orbital mechanics
- Orbit propagation
- Mission analysis
- Java library (Python bindings)

**Capabilities:**
- ✅ Orbital mechanics (good)
- ✅ Orbit propagation
- ✅ Free
- ✅ Java/Python bindings
- ❌ No control system design
- ❌ No ADCS
- ❌ Limited visualization
- ❌ Java-based (less accessible)

**Used By:**
- Research labs
- Java users
- Some European projects

**Strengths:**
- Orbital mechanics
- Free
- Java/Python

**Weaknesses:**
- No control design
- Java-based
- Limited capabilities

---

## 6. Comparison Matrix

| Tool | Type | Price | Control Design | MPC | Tunability | UI | Target Market |
|------|------|-------|---------------|-----|------------|----|--------------| 
| **AGI STK** | Enterprise | $50K-$500K | ❌ Basic | ❌ No | ❌ No | ✅ GUI | Enterprise |
| **FreeFlyer** | Enterprise | $30K-$200K | ❌ PID only | ❌ No | ❌ No | ✅ GUI | Mission Design |
| **GMAT** | Open Source | Free | ❌ Limited | ❌ No | ⚠️ Code | ❌ CLI | Research/NASA |
| **Basilisk** | Open Source | Free | ✅ ADCS | ⚠️ Basic | ⚠️ Code | ❌ CLI | Research |
| **MATLAB Toolbox** | Commercial | $5K-$10K | ✅ Simulink | ⚠️ Limited | ⚠️ Code | ✅ GUI | Research |
| **Poliastro** | Open Source | Free | ❌ No | ❌ No | ⚠️ Code | ❌ Library | Python Users |
| **Orbiter** | Free | Free | ❌ No | ❌ No | ❌ No | ✅ 3D | Hobbyists |
| **COSMOS** | Open Source | Free | ❌ No | ❌ No | ❌ No | ✅ GUI | Operations |
| **Your Product** | **Hybrid** | **$2K-$10K** | **✅ MPC** | **✅ Yes** | **✅ Yes** | **✅ Web** | **Research/Startups** |

---

## 7. Where Your Product Fits

### **Market Gap Analysis:**

**Control System Design:**
- ❌ AGI STK: No control design (black box)
- ❌ FreeFlyer: PID only (not MPC)
- ❌ GMAT: No control design
- ⚠️ Basilisk: ADCS but complex, no MPC
- ⚠️ MATLAB: Simulink but expensive, limited MPC
- ✅ **Your Product: MPC with full tunability** ← **Unique!**

**Pricing:**
- ❌ Enterprise tools: $30K-$500K (too expensive)
- ✅ Open source: Free but complex/no support
- ✅ **Your Product: $2K-$10K (sweet spot)** ← **Perfect!**

**Tunability:**
- ❌ Most tools: Black boxes (can't tune)
- ⚠️ Open source: Can modify code (but complex)
- ✅ **Your Product: Visual tuning + API** ← **Best of both!**

**Usability:**
- ❌ Enterprise tools: Complex (days to weeks)
- ❌ Open source: CLI, requires programming
- ✅ **Your Product: Web dashboard (5 minutes)** ← **Easiest!**

---

## Your Competitive Position

### **What Makes You Different:**

**1. MPC Control Focus**
- Most tools: Orbit propagation, mission planning
- Your product: **Control system design with MPC** ← **Unique focus!**

**2. Tunability**
- Most tools: Black boxes (can't tune)
- Your product: **Every parameter exposed and tunable** ← **Unique!**

**3. Pricing**
- Enterprise: $30K-$500K (too expensive)
- Open source: Free but complex
- Your product: **$2K-$10K (affordable)** ← **Perfect price!**

**4. Usability**
- Enterprise: Complex (days to weeks)
- Open source: CLI, programming required
- Your product: **Web dashboard (5 minutes)** ← **Easiest!**

**5. Format**
- Enterprise: Desktop apps (platform-specific)
- Open source: Libraries (requires programming)
- Your product: **Web dashboard + Python API** ← **Best of both!**

---

## Target Market Positioning

### **You're Not Competing With:**

**AGI STK / FreeFlyer:**
- They do: Mission planning, orbit propagation
- You do: Control system design
- Market: Different (they're $50K+, enterprise)
- Overlap: Minimal (they don't do control design)

**GMAT / Basilisk:**
- They do: Orbit propagation, ADCS (complex)
- You do: Control system design (easy)
- Market: Different (they're free but complex)
- Overlap: Some (both research, but you're easier)

**MATLAB/Simulink:**
- They do: Control design (general)
- You do: Satellite control design (specialized)
- Market: Some overlap (research)
- Advantage: You're specialized, cheaper, easier

---

### **You're Competing For:**

**Research Labs & Universities:**
- Need: Affordable control design tools
- Options: MATLAB ($5K-$10K), Basilisk (free but complex)
- Your advantage: Easier, specialized, better price

**CubeSat Startups:**
- Need: Validate control before launch
- Options: AGI STK ($50K+), Basilisk (complex)
- Your advantage: Affordable, easy, MPC focus

**Control Engineers:**
- Need: Tune control parameters
- Options: MATLAB (expensive), code from scratch
- Your advantage: Visual tuning, MPC ready

---

## Summary: Market Landscape

### **What Exists:**

**Mission Planning Tools:**
- AGI STK, FreeFlyer (orbit propagation, mission design)
- GMAT, Orbiter (orbit propagation, free)
- Focus: Where satellite goes (trajectory), not how (control)

**Control Design Tools:**
- MATLAB/Simulink (general control, expensive)
- Basilisk (ADCS, complex, free)
- Focus: Control design, but not specialized for satellites

**Your Niche:**
- **Satellite control system design with MPC**
- **Tunable, affordable, easy to use**
- **Perfect for: Research, startups, education**

---

## Key Insights

### **1. Market Gap:**
- Most tools focus on **orbit propagation** (where satellite goes)
- Few tools focus on **control design** (how satellite moves)
- **Your focus: Control design** ← **Niche opportunity!**

### **2. Pricing Gap:**
- Enterprise: $30K-$500K (too expensive for research/startups)
- Open source: Free but complex (requires expertise)
- **Your price: $2K-$10K** ← **Perfect sweet spot!**

### **3. Usability Gap:**
- Enterprise: Complex (days to weeks setup)
- Open source: CLI, programming required
- **Your format: Web dashboard** ← **Easiest!**

### **4. Tunability Gap:**
- Most tools: Black boxes (can't tune)
- Open source: Can modify code (but complex)
- **Your approach: Visual tuning + API** ← **Best of both!**

---

## Positioning Statement

> **"Satellite Control Studio is the only affordable, tunable MPC control simulator designed specifically for satellite control system design. Unlike mission planning tools (AGI STK, GMAT) or complex open-source tools (Basilisk), we focus on making control system design accessible with visual tuning and MPC control."**

**Key Differentiators:**
1. ✅ **MPC Control Focus** (not just orbit propagation)
2. ✅ **Fully Tunable** (not black box)
3. ✅ **Affordable** ($2K-$10K vs $50K-$500K)
4. ✅ **Easy to Use** (web dashboard vs CLI)
5. ✅ **Specialized** (satellite control, not general)

---

## Bottom Line

**Most aerospace simulators focus on:**
- Orbit propagation (where satellite goes)
- Mission planning (what satellite does)
- Visualization (showing the mission)

**Your product focuses on:**
- **Control system design** (how satellite moves)
- **MPC tuning** (optimize control parameters)
- **Validation** (test before deploying)

**This is a unique niche!** Most tools don't do what you do (tunable MPC control design), and the ones that try are either too expensive (MATLAB) or too complex (Basilisk).

**You're filling a real gap in the market!** 🚀
