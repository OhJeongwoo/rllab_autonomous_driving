import os

from pandas.core.indexing import IndexSlice
import carla
import random
import argparse
import yaml
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode, NGSimTimeslot
from carla_real_traffic_scenarios.ngsim.scenario import NGSimLaneChangeScenario
from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario
from carla_real_traffic_scenarios.utils.topology import get_lane_id, get_lane_ids

DTOR = math.pi / 180
def norm_calculator(vector: carla.Vector3D):
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)



def wrt_ego(ego_transform: carla.Transform, vehicle_location: carla.Location):

    position = vehicle_location.__sub__(ego_transform.location)
    euler_angle = [ego_transform.rotation.yaw, ego_transform.rotation.pitch, ego_transform.rotation.roll]
    rotation_matrix = R.from_euler('zyx', euler_angle, degrees=True)
    position_wrt_ego = rotation_matrix.inv().apply([position.x, position.y, position.z])

    return np.asarray(position_wrt_ego)

def vehicle_detector(vehicles:carla.ActorList, scenario: Scenario, origin: carla.Location, lane_id : int, max_distance: float):
    
    indicator = 1
    if scenario._world_map.get_waypoint(origin).next(max_distance)[0].transform.location.x < origin.x:
        indicator = -1

    preceding_distance = max_distance * indicator
    following_distance = - max_distance * indicator
    preceding = False
    following = False

    for vehicle in vehicles:
        loc = vehicle.get_location()
        wp = scenario._world_map.get_waypoint(loc)
        if wp.lane_id == lane_id:
            dx = loc.x - origin.x
            if dx*indicator > 0.1 and abs(dx) <= abs(preceding_distance):
                preceding_vehicle = vehicle
                preceding_distance = dx
                preceding = True

            if dx * indicator < -0.1 and abs(dx) <= abs(following_distance):
                following_vehicle = vehicle
                following_distance = dx
                following = True

    result = {}

    result['preceding'] = preceding
    if preceding:
        result['preceding vehicle'] = preceding_vehicle

    result['following'] = following
    if following:
        result['following vehicle'] = following_vehicle

    return result


def state_update(state, ego_vehicle: carla.Actor, target_vehicle: carla.Actor, idx: int):

        state[idx*6-3 : idx*6-1] = wrt_ego(ego_vehicle.get_transform(), target_vehicle.get_transform().location)[0:2]
        state[idx*6-1] = - target_vehicle.get_transform().rotation.yaw + ego_vehicle.get_transform().rotation.yaw 
        state[idx*6-1] *= DTOR
        state[idx*6] = norm_calculator(target_vehicle.get_velocity())  # vehicle velocity
        state[idx*6+1] = norm_calculator(target_vehicle.get_acceleration())  # vehicle acceleration
        state[idx*6+2] = target_vehicle.get_angular_velocity().z # vehicle angular velocity
        

def get_state(world:carla.World, scenario: Scenario, ego_vehicle: carla.Actor, max_distance: float):

    ngsim_record = scenario._ngsim_recording
    vehicles = world.get_actors().filter('vehicle.*')  

    state = np.zeros(39)

    state[0] = norm_calculator(ego_vehicle.get_velocity())  # ego vehicle velocity
    state[1] = norm_calculator(ego_vehicle.get_acceleration())  # ego vehicle acceleration
    state[2] = ego_vehicle.get_angular_velocity().z # ego vehicle angular velocity
    
    # reset waypoint and lane info
    ego_wp = scenario._world_map.get_waypoint(ego_vehicle.get_location())
    left_wp = ego_wp.get_left_lane()
    right_wp = ego_wp.get_right_lane()

    preceding_loc = ego_wp.next(max_distance)[0].transform.location
    following_loc = ego_wp.previous(max_distance)[0].transform.location

    ego_lane_id = ego_wp.lane_id

    if left_wp == None:
        left_lane_id = -999

    else:
        left_lane_id = left_wp.lane_id

    if right_wp == None:
        right_lane_id = 999

    else:
        right_lane_id = right_wp.lane_id

    # ego line vehicle state update
    detection = vehicle_detector(vehicles, scenario, ego_vehicle.get_location(), ego_lane_id, max_distance)

    if detection['preceding']:
        state_update(state, ego_vehicle, detection['preceding vehicle'], 2)

    else:
        state[9:11] = wrt_ego(ego_vehicle.get_transform(), preceding_loc)[0:2]
        state[11] = 0
        state[12:15] = state[0:3]
    
    if detection['following']:
        state_update(state, ego_vehicle, detection['following vehicle'], 5)

    else:
        state[27:29] = wrt_ego(ego_vehicle.get_transform(), following_loc)[0:2]
        state[29] = 0
        state[30:33] = state[0:3]

    #left line vehicle state update

    lane_direction = preceding_loc.__sub__(ego_vehicle.get_location())
    rotation_matrix = R.from_euler('z', -90, degrees=True)
    to_the_left = np.asarray(rotation_matrix.apply([lane_direction.x, lane_direction.y, lane_direction.z]))
    to_the_left = to_the_left * ego_wp.next(max_distance)[0].lane_width / np.linalg.norm(to_the_left)

    left_preceding = to_the_left + np.asarray([preceding_loc.x, preceding_loc.y, preceding_loc.z])
    left_preceding_loc = carla.Location(left_preceding[0], left_preceding[1], left_preceding[2])
    

    back_lane_direction = following_loc.__sub__(ego_vehicle.get_location())
    rotation_matrix = R.from_euler('z', 90, degrees=True)
    to_the_leftb = np.asarray(rotation_matrix.apply([back_lane_direction.x, back_lane_direction.y, back_lane_direction.z]))
    to_the_leftb = to_the_leftb * ego_wp.previous(max_distance)[0].lane_width / np.linalg.norm(to_the_leftb)

    left_following = to_the_leftb + np.asarray([following_loc.x, following_loc.y, following_loc.z])
    left_following_loc = carla.Location(left_following[0], left_following[1], left_following[2])

    if left_lane_id == -999:
        state[3:5] = wrt_ego(ego_vehicle.get_transform(), left_preceding_loc)[0:2]
        state[5] = 0
        state[6:9] = state[0:3]

        state[21:23] = wrt_ego(ego_vehicle.get_transform(), left_following_loc)[0:2]
        state[23] = 0
        state[24:27] = state[0:3]

    else:
        left_detection = vehicle_detector(vehicles, scenario, left_wp.transform.location, left_lane_id, max_distance)

        if left_detection['preceding']:
            state_update(state, ego_vehicle, left_detection['preceding vehicle'], 1)

        else:
            state[3:5] = wrt_ego(ego_vehicle.get_transform(), left_preceding_loc)[0:2]
            state[5] = 0
            state[6:9] = state[0:3]
    
        if left_detection['following']:
            state_update(state, ego_vehicle, left_detection['following vehicle'], 4)

        else:
            state[21:23] = wrt_ego(ego_vehicle.get_transform(), left_following_loc)[0:2]
            state[23] = 0
            state[24:27] = state[0:3]

    #right line vehicle state update

    rotation_matrix = R.from_euler('z', 90, degrees=True)
    to_the_right = - to_the_left
    to_the_right = to_the_right * ego_wp.next(max_distance)[0].lane_width / np.linalg.norm(to_the_right)

    right_preceding = to_the_right + np.asarray([preceding_loc.x, preceding_loc.y, preceding_loc.z])
    right_preceding_loc = carla.Location(right_preceding[0], right_preceding[1], right_preceding[2])

    rotation_matrix = R.from_euler('z', -90, degrees=True)
    to_the_rightb = -to_the_leftb
    to_the_rightb = to_the_rightb * ego_wp.previous(max_distance)[0].lane_width / np.linalg.norm(to_the_rightb)

    right_following = to_the_rightb + np.asarray([following_loc.x, following_loc.y, following_loc.z])
    right_following_loc = carla.Location(right_following[0], right_following[1], right_following[2])

    if right_lane_id == 999:
        state[15:17] = wrt_ego(ego_vehicle.get_transform(), right_preceding_loc)[0:2]
        state[17] = 0
        state[18:21] = state[0:3]
        state[33:35] = wrt_ego(ego_vehicle.get_transform(), right_following_loc)[0:2]
        state[35] = 0
        state[36:39] = state[0:3]
    else:
        right_detection = vehicle_detector(vehicles, scenario, right_wp.transform.location, right_lane_id, max_distance)

        if right_detection['preceding']:
            state_update(state, ego_vehicle, right_detection['preceding vehicle'], 3)

        else:
            state[15:17] = wrt_ego(ego_vehicle.get_transform(), right_preceding_loc)[0:2]
            state[17] = 0
            state[18:21] = state[0:3]
    
        if right_detection['following']:
            state_update(state, ego_vehicle, right_detection['following vehicle'], 6)

        else:
            state[33:35] = wrt_ego(ego_vehicle.get_transform(), right_following_loc)[0:2]
            state[35] = 0
            state[36:39] = state[0:3]

    
    return state