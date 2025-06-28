from gym.envs.registration import register

register(
    id='FleetManager-v0',
    entry_point='flow.envs.fleet_manager_env:FleetManagerEnv',
)
