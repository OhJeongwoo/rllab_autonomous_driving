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
from state import get_state


from model import MLP


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ngsim", "opendd"], default="opendd")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--num-episodes", default=10, type=int)
    args = parser.parse_args()
    return args


def prepare_ngsim_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ.get("NGSIM_DIR")
    assert data_dir, "Path to the directory with NGSIM dataset is required"
    ngsim_map = NGSimDatasets.list()
    ngsim_dataset = random.choice(ngsim_map)
    client.load_world(ngsim_dataset.carla_map.level_path)
    return NGSimLaneChangeScenario(
        ngsim_dataset,
        dataset_mode=DatasetMode.TRAIN,
        data_dir=data_dir,
        reward_type=RewardType.DENSE,
        client=client,
    )


def prepare_opendd_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ.get("OPENDD_DIR")
    assert data_dir, "Path to the directory with openDD dataset is required"
    maps = ["rdb1", "rdb2", "rdb3", "rdb4", "rdb5", "rdb6", "rdb7"]
    map_name = random.choice(maps)
    carla_map = getattr(CarlaMaps, map_name.upper())
    client.load_world(carla_map.level_path)
    return OpenDDScenario(
        client,
        dataset_dir=data_dir,
        dataset_mode=DatasetMode.TRAIN,
        reward_type=RewardType.DENSE,
        place_name=map_name,
    )


def prepare_ego_vehicle(world: carla.World) -> carla.Actor:
    car_blueprint = world.get_blueprint_library().find("vehicle.audi.a2")

    car_blueprint.set_attribute("role_name", "hero")

    ego_vehicle = world.spawn_actor(
        car_blueprint, carla.Transform(carla.Location(0, 0, 500), carla.Rotation())
    )

    assert ego_vehicle is not None, "Ego vehicle could not be spawned"

    # Setup any car sensors you like, collect observations and then use them as input to your model
    return ego_vehicle


if __name__ == "__main__":
    # set some paths if you need
    project_path = os.path.abspath("..")

    args = parser_args()

    # load yaml
    yaml_file = project_path + "/test.yaml"
    if os.path.exists(yaml_file):
        with open(yaml_file) as file:
            yaml_args = yaml.load(file)

    # TODO:set hyperparameters (or use yaml file)
    ###############################################


    ###############################################

    # starting carla client
    print(f"Trying to connect to CARLA server via {args.host}:{args.port}", end="...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)

    if args.dataset == "ngsim":
        scenario = prepare_ngsim_scenario(client)
    elif args.dataset == "opendd":
        scenario = prepare_opendd_scenario(client)
      
    world = client.get_world()
    spectator = world.get_spectator()
    ego_vehicle = prepare_ego_vehicle(world)
    scenario.reset(ego_vehicle)

    # TODO:load model
    # policy = MLP(state_dim_, action_dim_, hidden_layers_, learning_rate_)



    # start training
    for ep_idx in range(args.num_episodes):
        print(f"Running episode {ep_idx+1}/{args.num_episodes}")

        # Ego vehicle replaces one of the real-world vehicles
        scenario.reset(ego_vehicle)
        done = False
        total_reward = 0
        while not done:

            ngsim_record = scenario._ngsim_recording
            frame = ngsim_record.frame
            print(f"frame = {scenario._ngsim_recording.frame}")

            # get state
            max_distance = 40.0 
            state = get_state(world, scenario, ego_vehicle, max_distance)
            # state explanation (numpy, size:39)
            #
            # left lane    ego lane    right lane 
            #    1            2            3
            #    ^            ^            ^
            #    ^       ego vehicle       ^
            #    ^            ^            ^
            #    4            5            6
            #        (each number is idx)
            # 
            # state[0:3] : ego vehicle info (vel, acc, angular_vel)
            # state[6*idx-3 : 6*idx+3] : other vehicle info(x, y, theta, vel, acc angular_vel)
            

            # TODO:set action
            # action = policy(state)
            # ego_vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1]))

            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.2))  # for test

            # TODO: reward - can change on carla-real-traffic-scenario/ngsim/scenario.py
            chauffeur_cmd, reward, done, info = scenario.step(ego_vehicle)
            total_reward += reward
            print(f"Step: command={chauffeur_cmd.name}, total_reward={total_reward}")

            world.tick()

            # Server camera follows ego
            topdown_view = carla.Transform(
                ego_vehicle.get_transform().location + carla.Vector3D(x=0, y=0, z=30),
                carla.Rotation(pitch=-90),
            )
            spectator.set_transform(topdown_view)
        
        # TODO: update policy
        # if ep_idx % update_interval == 0:
            # update policy

    scenario.close()
