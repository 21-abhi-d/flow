# examples/run_fleet_manager.py

from flow.fleet_manager.rule_based_manager import FleetManagerAgent

def run(env):
    manager = FleetManagerAgent()

    for step in range(env.env_params.horizon):
        print(f"\n--- Step {step} ---")

        vehicle_states = env.get_vehicle_states()
        requests = env.get_current_requests()

        assignments = manager.assign_requests(vehicle_states, requests)

        env.apply_fleet_actions(assignments)

        env.step()  # advance the simulation

