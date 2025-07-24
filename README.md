# Reinforcement Learning for Fleet Management in Urban Traffic

This repository contains the simulation framework and learning environment developed as part of a thesis project exploring reinforcement learning (RL) for centralized fleet management in urban traffic scenarios. Built on top of the [Flow](https://github.com/flow-project/flow) framework and the SUMO traffic simulator, the system extends standard Flow capabilities with a custom environment, agent logic, and dynamic demand modeling.

---

## Overview

This project focuses on intelligent fleet coordination in dense, city-like traffic environments. It uses a modified **MiniCity** network to simulate realistic urban layouts, intersections, and demand zones. The goal is to train a centralized RL agent to optimize:

- Vehicle dispatching  
- Ride assignment  
- Proactive movements based on learned spatiotemporal demand patterns  

### Key Objectives

- Maximize vehicle utilization  
- Minimize passenger wait times  
- Handle dynamic request streams with spatial and temporal variation  
- Support both reactive assignment and proactive repositioning of idle vehicles  

---

## Key Features

### ✅ Custom Environment (`FleetManagerEnv`)
A tailored Gym-compatible environment extending Flow’s base classes. Integrates request queues, trip lifecycle management, and vehicle-idle tracking.

### 🔁 Flexible Agent Interface
Supports both non-RL control policies (e.g., heuristic-based) and RL policies trained using **stable-baselines3** and **RLlib**.

### 🔥 Demand Heatmap Encoding
Observation space includes a dynamic 2D heatmap of demand distribution, enabling proactive fleet behavior and pre-positioning of idle vehicles.

### 🏙️ Realistic City Network (MiniCity)
Simulation is set in a city-style network with realistic road layouts, intersections, and demand zones, adapted from the Flow `minicity` network.

### 🧪 Dual Training Modes
- **Non-RL Mode:** Baseline comparisons with deterministic control (e.g., nearest vehicle matching).  
- **RL Mode:** Agents trained using algorithms such as TD3 and PPO with either Stable-Baselines or RLlib.

---

## Project Structure

flow/ # Modified Flow source with custom env, network, and controllers
exp_configs/ # Experiment configuration files for RL and non-RL experiments
scripts/ # Utility scripts for training, testing, and logging
training_data/ # TensorBoard logs, trained models, and evaluation outputs

---

## Visualization and Evaluation

- Training statistics (e.g., reward, utilization, wait time) are visualized using **TensorBoard**
- Evaluation logs and metrics are saved as `.csv` files for analysis
- Models are checkpointed regularly to allow resumption or testing

---

## Acknowledgements

This work builds upon the open-source [Flow](https://github.com/flow-project/flow) framework, developed by UC Berkeley and ICSI.  
Traffic simulation is powered by [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/).

---
