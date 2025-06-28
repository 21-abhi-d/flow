import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import FlattenObservation

# Make sure your env is importable
from flow.envs.fleet_manager_env import FleetManagerEnv
from flow.core.params import EnvParams, SimParams, VehicleParams, NetParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.networks import MiniCityNetwork
from flow.controllers import IDMController, ContinuousRouter


def make_env():
    def _init():
        # Create the base environment
        net_params = NetParams()

        vehicles = VehicleParams()
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=2.5,
                max_speed=30.0
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
                model="SL2015"
            ),
            num_vehicles=5
        )

        env_params = EnvParams()
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

        network = MiniCityNetwork(
            name="fleet-manager-net",
            vehicles=vehicles,
            net_params=net_params
        )

        env = FleetManagerEnv(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator='traci'
        )

        # Flatten observation before vectorizing
        return FlattenObservation(env)

    return _init

# Wrap in vectorized environment
env = DummyVecEnv([make_env()])

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("fleet_manager_model")
