import random

class FleetManagerAgent:
    def __init__(self):
        self.busy_until = {}  # vehicle_id -> timestep when it becomes free
        self.active_trips = {} # vehicle_id -> {"request_id": ..., "dropoff_time": ...}


    def is_vehicle_idle(self, vid, current_time):
        return self.busy_until.get(vid, 0) <= current_time
    
    def assign_requests(self, vehicles, requests, current_time):
        assignments = {}
        unassigned = []

        # Filter idle vehicles not currently busy
        idle_vehicles = {
            vid: v for vid, v in vehicles.items()
            if self.is_vehicle_idle(vid, current_time)
        }
        print(f"[DEBUG] Idle vehicles: {list(idle_vehicles.keys())}")

        # Assume average driving speed in meters per second
        AVERAGE_SPEED = 10  # ~36 km/h

        for req in requests:
            closest_id = None
            min_dist = float("inf")

            for vid, v in idle_vehicles.items():
                dist = ((v["pos"][0] - req["pos"][0])**2 + (v["pos"][1] - req["pos"][1])**2) ** 0.5
                if dist < min_dist:
                    closest_id = vid
                    min_dist = dist

            if closest_id:
                # Simulate time to drive to the request and perform the trip
                # pickup_time = int(min_dist / AVERAGE_SPEED)
                # trip_duration = random.randint(20, 60)  # actual trip duration
                # dropoff_time = current_time + pickup_time + trip_duration
                pickup_time = 1  # or 0
                trip_duration = 5  # fixed small trip
                dropoff_time = current_time + pickup_time + trip_duration

                assignments[closest_id] = req["id"]
                del idle_vehicles[closest_id]

                self.busy_until[closest_id] = dropoff_time
                self.active_trips[closest_id] = {
                    "request_id": req["id"],
                    "dropoff_time": dropoff_time
                }

                print(f"[Dispatch] Assigning {closest_id} to {req['id']}, "
                    f"pickup in {pickup_time}s, drop-off at t={dropoff_time}")
            else:
                unassigned.append(req)  # No available vehicle

        return assignments, unassigned