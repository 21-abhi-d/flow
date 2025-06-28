from flow.envs.base import Env
from flow.fleet_manager.rule_based_manager import FleetManagerAgent
from examples.run_fleet_manager import run as run_fleet_manager
from gym.spaces import Box, MultiDiscrete, Discrete
import numpy as np

import random


def ensure_list(val):
    return val if isinstance(val, list) else [val]

class FleetManagerEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        self.manager = FleetManagerAgent()
        self.active_requests = []
        self.req_id_counter = 0
        self.request_rate = 1.0
        
        self.time_counter = 0
        self.request_spawn_times = {}     # request_id -> time
        self.request_assign_times = {}    # request_id -> time
        self.metrics_log = [] 
        self.completed_requests = []
        self.request_slots = [None] * 50
        
        
    @property
    def observation_space(self):
        return Box(low=0.0, high=1e3, shape=(210,), dtype=np.float32)

    def step(self, rl_actions=None):
        print("\n[DEBUG] Vehicle busy status:")
        for vid in self.k.vehicle.get_ids():
            print(f" - {vid}: busy_until={self.manager.busy_until.get(vid, 0)}, time={self.time_counter}")
        self.request_slots = self.active_requests[:50]
        # Update kernel (required in every step)
        self.k.update(reset=False)

        # Track simulation time
        if not hasattr(self, "time_counter"):
            self.time_counter = 0
        self.time_counter += 1

                
        # Get vehicle states and active ride requests
        self.vehicles = self.get_vehicle_states()
        self.requests = self.get_active_requests()
        self.request_slots = self.active_requests[:50]
        
        for vid in list(self.manager.busy_until.keys()):
            if self.manager.busy_until[vid] <= self.time_counter:
                self.vehicles[vid]["status"] = "idle"
                del self.manager.busy_until[vid]
        
        for vid in list(self.manager.active_trips.keys()):
            if self.time_counter >= self.manager.active_trips[vid]["dropoff_time"]:
                self.vehicles[vid]["status"] = "idle"
                print(f"[RESET] Setting {vid} to idle at t={self.time_counter}")
                # del self.manager.active_trips[vid]

        assignments = {}
        unassigned_requests = self.requests

        if rl_actions is not None:
            # Use RL output to decide assignments
            print("[DEBUG] Using RL agent to assign requests")
            for idx, req in enumerate(self.request_slots):
                if req is not None:
                    print(f"[SLOT DEBUG] Slot {idx}: req_id={req['id']}, time={req['time']}, pos={req['pos']}")
            assignments = self._apply_rl_actions(rl_actions)
        else:
            # Default: use rule-based manager
            print("[DEBUG] Using rule-based assignment")
            assignments, unassigned_requests = self.manager.assign_requests(self.vehicles, self.requests, self.time_counter)
            for req_id in assignments.values():
                self.request_assign_times[req_id] = self.time_counter
            self.apply_actions(assignments)
        # elif Decision by LLM:
        # logic

        to_remove = []
        for vid, trip in self.manager.active_trips.items():
            if self.time_counter >= trip["dropoff_time"]:
                req_id = trip["request_id"]
                spawn_time = self.request_spawn_times.get(req_id)
                assign_time = self.request_assign_times.get(req_id)
                print(f"[DROP CHECK] {vid}: time={self.time_counter}, dropoff_time={trip['dropoff_time']}")

                if spawn_time is not None and assign_time is not None:
                    wait_time = assign_time - spawn_time
                    total_time = self.time_counter - spawn_time
                else:
                    wait_time = 0
                    total_time = 0

                print(f"[DEBUG] Completed request {req_id}: spawned at {spawn_time}, assigned at {assign_time}, wait: {wait_time}")

                self.completed_requests.append({
                    "request_id": req_id,
                    "vehicle_id": vid,
                    "spawn_time": spawn_time,
                    "assign_time": assign_time,
                    "dropoff_time": self.time_counter,
                    "wait_time": wait_time,
                    "total_time": total_time
                })

                print(f"[Complete] Vehicle {vid} completed request {req_id} (wait: {wait_time}, total: {total_time})")
                to_remove.append(vid)

        # Remove completed trips
        for vid in to_remove:
            del self.manager.active_trips[vid]
            self.manager.busy_until[vid] = 0
            print(f"[IDLE] Vehicle {vid} is now idle again at t={self.time_counter}")

        # Remove served requests
        self.active_requests = unassigned_requests

        # Default return values
        done = False
        info = {
            "num_requests": len(self.active_requests),
            "time": self.time_counter,
        }

        print(f"[Step {self.time_counter}] Active requests in queue: {len(self.active_requests)}")

        obs = self.get_state()
        print(f"[DEBUG] Observation state shape: {obs.shape}")
        print(f"[DEBUG] First few values of state: {obs[:10]}")

        # 🔍 Check for NaNs or infs in observations
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("[ERROR] Observation contains NaN or Inf values!")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate wait times for newly assigned requests
        wait_times = [
            self.request_assign_times[req_id] - self.request_spawn_times[req_id]
            for req_id in assignments.values()
            if req_id in self.request_spawn_times
        ]
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
        num_assigned = len(assignments)

        reward = -avg_wait_time + 0.1 * num_assigned

        # 🔍 Check reward for invalid values
        if np.isnan(reward) or np.isinf(reward):
            print("[ERROR] Reward is NaN or Inf! Resetting to 0.")
            reward = 0.0

        step_metrics = {
            "time": self.time_counter,
            "assigned": num_assigned,
            "active_requests": len(self.active_requests),
            "avg_wait_time": round(avg_wait_time, 2),
            "reward": round(reward, 2)
        }
        self.metrics_log.append(step_metrics)

        print(f"[Metrics] {step_metrics}")
        return obs, reward, done, info

    def reset(self):
        super().reset()
        self.k.update(reset=True)

        self.time_counter = 0
        self.req_id_counter = 0
        self.active_requests = []
        self.request_spawn_times = {}
        self.request_assign_times = {}
        self.completed_requests = []
        self.metrics_log = []

        # Spawn 5 fake requests
        for _ in range(5):
            x = np.random.uniform(100, 900)
            y = np.random.uniform(100, 900)
            self.active_requests.append({
                "id": self.req_id_counter,
                "pos": (x, y),
                "time": self.time_counter
            })
            self.request_spawn_times[self.req_id_counter] = self.time_counter
            self.req_id_counter += 1

        # Spawn at least 5 RL vehicles
        for i in range(5):
            veh_id = f"rl_{i}"
            if veh_id not in self.k.vehicle.get_ids():
                try:
                    self.k.vehicle.add(
                        veh_id=veh_id,
                        type_id="rl",
                        edge="e_10",  # replace with valid edge
                        lane=0,
                        pos="0",
                        speed=0
                    )
                    self.manager.busy_until[veh_id] = 0
                    
                    print(f"[RESET DEBUG] Vehicles after spawn: {self.k.vehicle.get_ids()}")
                except Exception as e:
                    print(f"[WARN] Vehicle add failed for {veh_id}: {e}")

        obs = self.get_state()
        print(f"[DEBUG] reset(): state shape = {obs.shape}, first 10 vals: {obs[:10]}")
        return obs

    def get_vehicle_states(self):
        veh_ids = self.k.vehicle.get_ids()
        print(f"[DEBUG] Vehicles in sim: {len(veh_ids)}")
        states = {}
        for vid in veh_ids:
            pos = self.k.vehicle.get_position(vid)
            if not isinstance(pos, tuple):
                pos = (pos, 0.0)

            busy_until = self.manager.busy_until.get(vid, 0)
            if busy_until <= self.time_counter:
                status = "idle"
            else:
                status = "busy"

            states[vid] = {"pos": pos, "status": status}
        return states

    def get_active_requests(self):
        print("[DEBUG] Checking for new request spawn...")
        if self.time_counter >= self.env_params.horizon - 50:
            print(f"[INFO] Request spawning disabled at t={self.time_counter}")
            return self.active_requests
        
        if random.random() < self.request_rate:
            print("[DEBUG] Spawn condition met. Attempting to add a request...")
            all_edges = ensure_list(self.k.network.get_edge_list())
            print(f"[DEBUG] Available edges: {all_edges[:5]}... (total: {len(all_edges)})")
            if all_edges:
                edge_id = random.choice(all_edges)
                try:
                    lane_id = f"{edge_id}_0"
                    shape = self.k.kernel_api.lane.getShape(lane_id)
                    if shape:
                        mid_idx = len(shape) // 2
                        x, y = shape[mid_idx]
                        request = {
                            "id": f"req{self.req_id_counter}",
                            "pos": (x, y),
                            "time": getattr(self, "time_counter", 0)
                        }
                        self.request_spawn_times[request["id"]] = self.time_counter
                        print(f"[Spawn] New request: {request}")
                        self.active_requests.append(request)
                        # for i in range(len(self.request_slots)):
                        #     if self.request_slots[i] is None:
                        #         self.request_slots[i] = request
                        #         break
                        self.req_id_counter += 1
                    else:
                        print(f"[WARN] Empty shape for lane {lane_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to get position for {edge_id}: {e}")

        return self.active_requests

    def assign_vehicle_to_request(self, vehicle_id, req_id, current_time):
        trip_duration = 5
        # trip_duration = int(np.linalg.norm(np.array(req["pos"]) - np.array(veh["pos"])) / speed)
        self.busy_until[vehicle_id] = current_time + trip_duration
        self.active_trips[vehicle_id] = {
            "request_id": req_id,
            "dropoff_time": current_time + trip_duration
        }
        print(f"[ASSIGN] Vehicle {vehicle_id} assigned to {req_id}, dropoff at {current_time + trip_duration}")

    def _dispatch_actions(self, assignments):
        for vehicle_id, req_id in assignments.items():
            self.manager.assign_vehicle_to_request(vehicle_id, req_id, self.time_counter)
            # print(f"[DEBUG] current_time passed: {current_time}, self.time_counter: {self.time_counter}")
    
    
    def apply_actions(self, rl_actions):
        assignments = {}

        # Get up to 20 idle vehicles (in consistent order)
        idle_vehicles = [
            vid for vid in sorted(self.k.vehicle.get_ids())
            if self.manager.busy_until.get(vid, 0) <= self.time_counter
        ]

        # Interpret RL action vector
        for i, vehicle_id in enumerate(idle_vehicles[:20]):
            raw_action = int(rl_actions[i])
            req_idx = raw_action - 1  # shift: 0 = idle, 1..50 = req_slots[0..49]

            if 0 <= req_idx < len(self.request_slots):
                req = self.request_slots[req_idx]
                if req is not None:
                    assignments[vehicle_id] = req["id"]
                    self.request_assign_times[req["id"]] = self.time_counter
                    self.request_slots[req_idx] = None
            print(f"[DEBUG] raw_action: {raw_action}, req_idx: {req_idx}, req: {req}")

        
        # Call the existing logic for applying dispatches
        self._dispatch_actions(assignments)

        # Remove assigned requests from active queue
        assigned_ids = set(assignments.values())
        self.active_requests = [
            req for req in self.active_requests if req["id"] not in assigned_ids
        ]

    
    @property
    def action_space(self):
        # Each vehicle picks a request index (-1 for idle, 0–49 for request)
        max_requests = 50
        max_vehicles = 20
        # Add 2 to handle -1 offset (shift range to 0–50 internally)
        return MultiDiscrete([max_requests + 2] * max_vehicles)
    
    def _apply_rl_actions(self, rl_actions):
        """
        RL assigns each vehicle to a request index, or -1 to stay idle.
        """
            
        print(f"[DEBUG] RL Actions received: {rl_actions}")
        
        assignments = {}
        
        # DEBUG: Inspect request slots
        for idx, r in enumerate(self.request_slots):
            print(f"[DEBUG] SLOT {idx}: {r['id'] if r else 'None'}")
            
        idle_vehicles = {
            vid: v for vid, v in self.vehicles.items()
            if self.manager.busy_until.get(vid, 0) <= self.time_counter
        }
        
        print(f"[DEBUG] Idle vehicles: {list(idle_vehicles.keys())}")

        veh_ids = list(idle_vehicles.keys())
        for i, veh_id in enumerate(veh_ids):
            if i >= len(rl_actions):
                break

            req_idx = int(rl_actions[i]) - 1

            # 🚨 Guard against invalid index
            if not (0 <= req_idx < len(self.request_slots)):
                print(f"[SKIP] Vehicle {veh_id} -> invalid req_idx {req_idx}, skipping")
                continue

            print(f"[DEBUG] Vehicle {veh_id} assigned to action {rl_actions[i]} -> req_idx {req_idx}")
            req = self.request_slots[req_idx]
            if req is not None:
                assignments[veh_id] = req["id"]
                self.request_assign_times[req["id"]] = self.time_counter
                self.request_slots[req_idx] = None

                veh_pos = idle_vehicles[veh_id]["pos"]
                pickup_dist = ((veh_pos[0] - req["pos"][0])**2 + (veh_pos[1] - req["pos"][1])**2) ** 0.5
                AVERAGE_SPEED = 10  # meters/sec
                pickup_time = int(pickup_dist / AVERAGE_SPEED)
                trip_duration = random.randint(20, 60)
                dropoff_time = self.time_counter + pickup_time + trip_duration

                self.manager.busy_until[veh_id] = dropoff_time
                self.manager.active_trips[veh_id] = {
                    "request_id": req["id"],
                    "dropoff_time": dropoff_time
                }

                print(f"[RL Dispatch] Assigned {veh_id} to {req['id']} -> pickup in {pickup_time}s, drop-off at t={dropoff_time}")
            else: 
                print(f"[SKIP] Request index {req_idx} was empty.")
                        
        print(f"[DEBUG] Assignments from RL: {assignments}")

        self.apply_actions(assignments)
        print(f"[FINAL DEBUG] RL assigned {len(assignments)} vehicles this step.")
        
        return assignments
    
    def summarize_metrics(self):
        print("\n--- Simulation Summary ---")
        
        # Completed requests
        total_completed = len(self.completed_requests)
        if total_completed > 0:
            avg_wait = sum(req["wait_time"] for req in self.completed_requests) / total_completed
            avg_total = sum(req["total_time"] for req in self.completed_requests) / total_completed
            print(f"Total completed requests: {total_completed}")
            print(f"Average wait time: {round(avg_wait, 2)}")
            print(f"Average total time: {round(avg_total, 2)}")
        else:
            print("No requests were completed.")

        # Metrics log over time
        if self.metrics_log:
            print(f"\nFinal timestep recorded: {self.metrics_log[-1]['time']}")
            print("Sample step metrics (last 5):")
            for row in self.metrics_log[-5:]:
                print(row)
        
    def get_state(self):
        max_vehicles = 20
        max_requests = 50

        vehicle_obs = []
        for vid in sorted(self.k.vehicle.get_ids())[:max_vehicles]:
            pos = self.k.vehicle.get_position(vid)
            if not isinstance(pos, tuple) or any(p is None or np.isnan(p) or np.isinf(p) for p in pos):
                pos = (0.0, 0.0)

            is_busy = 1.0 if self.manager.busy_until.get(vid, 0) > self.time_counter else 0.0
            vehicle_obs.append([pos[0], pos[1], is_busy])

        while len(vehicle_obs) < max_vehicles:
            vehicle_obs.append([0.0, 0.0, 0.0])

        # self.request_slots = self.active_requests[:max_requests]

        request_obs = []
        for req in self.request_slots:
            if req is None:
                request_obs.append([0.0, 0.0, 0.0])
            else:
                x, y = req.get("pos", (0.0, 0.0))
                age = self.time_counter - req.get("time", self.time_counter)
                if any(v is None or np.isnan(v) or np.isinf(v) for v in [x, y, age]):
                    x, y, age = 0.0, 0.0, 0.0
                request_obs.append([x, y, float(age)])

        while len(request_obs) < max_requests:
            request_obs.append([0.0, 0.0, 0.0])

        flat_state = [val for sublist in (vehicle_obs + request_obs) for val in sublist]
        obs = np.array(flat_state, dtype=np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs, -1e3, 1e3)

        print(f"[DEBUG] get_state returned {obs.shape} - first 10 values: {obs[:10]}")
        return obs
