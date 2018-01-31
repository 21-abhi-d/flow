import numpy as np

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario

# parameters required in net_params' additional_params attribute
REQUIRED_NET_PARAMS = ["radius_ring", "lanes", "speed_limit"]


class Figure8Scenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """
        Initializes a figure 8 scenario.
        Required net_params: radius_ring, lanes, speed_limit, resolution.
        In order for right-of-way dynamics to take place at the intersection,
        set "no_internal_links" in net_params to False.

        See Scenario.py for description of params.
        """
        for param in REQUIRED_NET_PARAMS:
            if param not in net_params.additional_params:
                raise ValueError("Figure eight network parameter {} not "
                                 "supplied".format(param))

        self.ring_edgelen = net_params.additional_params[
                                "radius_ring"] * np.pi / 2.
        self.intersection_len = 2 * net_params.additional_params["radius_ring"]
        self.junction_len = 2.9 + 3.3 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params.additional_params["length"] = \
            6 * self.ring_edgelen + 2 * self.intersection_len + \
            2 * self.junction_len + 10 * self.inner_space_len

        self.radius_ring = net_params.additional_params["radius_ring"]
        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.resolution = net_params.additional_params["resolution"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """
        See base class
        """
        edgestarts = \
            [("bottom_lower_ring",
              0 + self.inner_space_len),
             ("right_lower_ring_in",
              self.ring_edgelen + 2 * self.inner_space_len),
             ("right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 +
              self.junction_len + 3 * self.inner_space_len),
             ("left_upper_ring",
              self.ring_edgelen + self.intersection_len +
              self.junction_len + 4 * self.inner_space_len),
             ("top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 5 * self.inner_space_len),
             ("right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 6 * self.inner_space_len),
             ("bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 7 * self.inner_space_len),
             ("bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
              2 * self.junction_len + 8 * self.inner_space_len),
             ("top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 9 * self.inner_space_len),
             ("left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 10 * self.inner_space_len)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        """
        See base class
        """
        intersection_edgestarts = \
            [(":center_intersection_%s" % (1 + self.lanes),
              self.ring_edgelen + self.intersection_len / 2 +
              3 * self.inner_space_len),
             (":center_intersection_1",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
              self.junction_len + 8 * self.inner_space_len)]

        return intersection_edgestarts

    def specify_internal_edge_starts(self):
        """
        See base class
        """
        internal_edgestarts = \
            [(":bottom_lower_ring",
              0),
             (":right_lower_ring_in",
              self.ring_edgelen + self.inner_space_len),
             (":right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 +
              self.junction_len + 2 * self.inner_space_len),
             (":left_upper_ring",
              self.ring_edgelen + self.intersection_len +
              self.junction_len + 3 * self.inner_space_len),
             (":top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 4 * self.inner_space_len),
             (":right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 5 * self.inner_space_len),
             (":bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 6 * self.inner_space_len),
             (":bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
              2 * self.junction_len + 7 * self.inner_space_len),
             (":top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 8 * self.inner_space_len),
             (":left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 9 * self.inner_space_len)]

        return internal_edgestarts
