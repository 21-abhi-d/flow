
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import FlattenObservation
from log_callback import RewardLoggingCallback

from flow.envs.fleet_manager_env import FleetManagerEnv
from flow.core.params import EnvParams, SimParams, VehicleParams, NetParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.networks import TrafficLightGridNetwork
from flow.controllers import IDMController, ContinuousRouter
from collections import defaultdict
network_cls = TrafficLightGridNetwork

def build_routes(net_params, valid_edges):
    routes = defaultdict(list)

    row_num = net_params.additional_params["grid_array"]["row_num"]
    col_num = net_params.additional_params["grid_array"]["col_num"]

    # Horizontal routes (left to right and back)
    for i in range(row_num):
        left_edge = f"bot{i}_0"
        right_edge = f"top{i}_{col_num}"

        if left_edge in valid_edges:
            routes[left_edge] = [f"bot{i}_{j}" for j in range(1, col_num)]

        if right_edge in valid_edges:
            routes[right_edge] = [f"top{i}_{j}" for j in reversed(range(1, col_num))]

    # Vertical routes (top to bottom and back)
    for j in range(col_num):
        down_edge = f"right0_{j}"
        up_edge = f"left{row_num - 1}_{j}"

        if down_edge in valid_edges:
            routes[down_edge] = [f"right{i}_{j}" for i in range(1, row_num)]

        if up_edge in valid_edges:
            routes[up_edge] = [f"left{i}_{j}" for i in reversed(range(1, row_num))]

    # Additional: rightmost bot edges and top edges (e.g., bot0_3, top0_3)
    for i in range(row_num):
        outer_bot = f"bot{i}_{col_num}"
        outer_top = f"top{i}_{col_num}"

        if outer_bot in valid_edges and outer_bot not in routes:
            routes[outer_bot] = [f"bot{i}_{j}" for j in reversed(range(col_num))]

        if outer_top in valid_edges and outer_top not in routes:
            routes[outer_top] = [f"top{i}_{j}" for j in reversed(range(col_num))]

    # Additional: bottommost left/right vertical inflow edges
    for j in range(col_num):
        outer_left = f"left{row_num}_{j}"
        outer_right = f"right{row_num}_{j}"

        if outer_left in valid_edges and outer_left not in routes:
            routes[outer_left] = [f"left{i}_{j}" for i in reversed(range(row_num))]

        if outer_right in valid_edges and outer_right not in routes:
            routes[outer_right] = [f"right{i}_{j}" for i in reversed(range(row_num))]

    # Final safety: dummy single-edge routes for any remaining unhandled inflow edges
    for i in range(row_num + 1):
        for j in range(col_num + 1):
            for prefix in ["top", "bot", "left", "right"]:
                edge = f"{prefix}{i}_{j}"
                if edge in valid_edges and edge not in routes:
                    routes[edge] = [edge]  # fallback route to avoid crashing

    print("[DEBUG] Final validated route keys:")
    print("  ->", list(routes.keys()))

    return routes

def make_env():
    def _init():
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(min_gap=2.5, max_speed=30.0),
            lane_change_params=SumoLaneChangeParams(lane_change_mode=0, model="SL2015"),
            num_vehicles=12
        )

        env_params = EnvParams(additional_params={"num_vehicles": 6})
        sim_params = SimParams()
        sim_params.render = False
        sim_params.sim_step = 0.1
        sim_params.num_clients = 1
        sim_params.use_ballistic = False
        sim_params.no_step_log = True
        sim_params.lateral_resolution = 0.25
        sim_params.overtake_right = False
        sim_params.seed = 42
        sim_params.print_warnings = True
        sim_params.teleport_time = -1
        sim_params.restart_instance = True
        
        grid_array = {
            "row_num": 3,
            "col_num": 3,
            "inner_length": 50,
            "short_length": 35,
            "long_length": 100,
            "cars_left": 3,
            "cars_right": 3,
            "cars_top": 3,
            "cars_bot": 3,
        }

        net_params = NetParams(
            additional_params={
                "grid_array": grid_array,
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
                "speed_limit": {
                    "horizontal": 35,
                    "vertical": 35
                },
                "traffic_lights": False,
                "bidirectional": True,
            },
        )

        network = network_cls(
            name="fleet-manager-grid",
            vehicles=vehicles,
            net_params=net_params
        )
        
        edge_data_from_sumo = network.specify_edges(net_params)
        valid_edges = set(e["id"] for e in edge_data_from_sumo)
        
        print("[DEBUG] All edge IDs in SUMO net:")
        if hasattr(network, "specify_edges"):
            edges = network.specify_edges(network.net_params)
            print("  ->", edges)
        else:
            print("  -> Network has no specify_edges() method")
        
        network.routes = build_routes(net_params, valid_edges)
        print("[DEBUG] Built custom routes:")
        for rid, edges in network.routes.items():
            print(f"  Route ID '{rid}': {edges}")
        print("[DEBUG] Total routes built:", len(network.routes))
        
        print("[DEBUG] Routes passed to env.k.network.rts:")
        for rid, path in network.routes.items():
            print(f"  {rid} starts at {path[0]}")


        print("[DEBUG] Creating environment with the following routes:")
        print("  ->", list(network.routes.keys()))
        env = FleetManagerEnv(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator='traci'
        )
        env.k.network.rts = network.routes
        print("[DEBUG] Injected routes into env.k.network.rts:")
        print("  ->", list(env.k.network.rts.keys()))

        return FlattenObservation(env)

    return _init

env = DummyVecEnv([make_env()])
callback = RewardLoggingCallback()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    learning_rate=5e-5,
    clip_range=0.08,
    ent_coef=0.01,
    vf_coef=0.8,
    clip_range_vf=0.2,
    gamma=0.99,
    n_epochs=10,
    batch_size=64,
    gae_lambda=0.95
)

model.learn(total_timesteps=50000, callback=callback)
model.save("fleet_manager_model")