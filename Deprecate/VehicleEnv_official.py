import cv2
import numpy as np
import torch
import random
import time
from typing import *

from PathConfigs import *

from srunner.scenariomanager.carla_data_provider import *
from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.statistics_manager import StatisticsManager
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner

from EnvUtils import VehicleState, VehicleAction

from Agents.HumanAgent import HumanAgent

# Reference blog.wuhanstudio.uk

class VehicleEnv:
    STEP_TICKS = 5
    MAX_N_STEPS = 1000
    front_camera = None
    radar_numpy = None

    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs
        self.client = carla.Client(configs["host"], configs["port"])
        self.client.set_timeout(configs["timeout"])
        self.traffic_manager = self.client.get_trafficmanager(configs["traffic_manager_port"])
        self.world = self.client.get_world()

        self.route_planner = GlobalRoutePlanner(self.world.get_map(), 1.0)

        self.statistics_manager = StatisticsManager('./simulation_results.json', './live_results.txt')
        self.manager = ScenarioManager(configs["timeout"], self.statistics_manager, 0)
        # self.world = self.client.load_world(configs["world"])
        # Carla world documentation: https://carla.readthedocs.io/en/latest/python_api/#carla.World

        self.routes_indexer = RouteIndexer(configs["routes"], configs["repetitions"], "")
        self.switchRoute()

        self.blueprint_lib = self.world.get_blueprint_library()


    def loadWorld(self, route_config):
        print(f"Loading world: {route_config.town}")
        if route_config.town != self.world.get_map().name:
            self.world = self.client.load_world(route_config.town)
            self.route_planner = GlobalRoutePlanner(self.world.get_map(), 1.0)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / 20.0
        settings.synchronous_mode = True
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        # settings.weather = weather
        self.world.apply_settings(settings)

        # self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.configs["traffic_manager_port"])

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_random_device_seed(self.configs["traffic_manager_seed"])

        # Wait for the world to be ready
        self.world.tick()
        print("World Ready!")


    def switchRoute(self):
        if self.routes_indexer.peek():
            route_config = self.routes_indexer.get_next_config()

            # scenario_name = f"{route_config.name}_rep{route_config.repetition_index}"

            # Load the scenario
            self.loadWorld(route_config)

            self.spectator = CarlaDataProvider.get_world().get_spectator()

            self.scenario = RouteScenario(world=self.world, config=route_config, debug_mode=0)
            # The several lines below will cause scenario to run itself
            # Only with official agent class, this canbe done
            # self.statistics_manager.set_scenario(route_config.scenario_configs[0], self.scenario)
            # self.manager.load_scenario(route_config, self.scenario, AutonomousAgent(self.configs["host"], self.configs["port"]),
            #                            route_config.repetition_index)
            # self.manager._tick_scenario()
            # self.manager.run_scenario()

            # All waypoints in the route
            self.keypoints: List[carla.Location] = route_config.keypoints

            # All instructions during the route
            # What the agent wants to follow is self.scenario_configs[i].route
            # What self.scenario_configs[i].trigger_points is reached, the agent should follow self.scenario_configs[i].route
            self.scenario_configs = route_config.scenario_configs
            # self.scenario_configs[0].route: List[Tuple[carla.Transform, carla.RoadOption]] # 8k points
            # self.scenario_configs[0].trigger_points: List[carla.Transform] # usually 1 point
            # all scenario routes are the same
            self.route = self.scenario_configs[0].route[:]

            # self.manager.load_scenario(route_config, scenario, AutonomousAgent(), route_config.repetition_index)
            # self.manager.run_scenario()

    def reset(self):
        self.step_count = 0
        self.collision_hist = []
        self.actors = []
        self.destroy()
        self.route = self.scenario_configs[0].route[:]

        self.setupVehicle()

        sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.setupCamera(sensor_transform)
        self.setupRadar(sensor_transform)
        self.setupGnss(sensor_transform)
        self.setupCollision(sensor_transform)

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.car_prev_dist = 0.0

        self.episode_begin_time = time.time()

        while self.front_camera is None or self.radar_numpy is None or self.gnss_xyz is None:
            self.world.tick()
        self.world.tick()


    def setupVehicle(self):
        # This creates vehicle
        try:
            self.ego_vehicle = self.scenario.ego_vehicles[0]

            # spawn vehicles and set them to autopilot
            for i in range(10):
                random_spawn_point = random.choice(self.world.get_map().get_spawn_points())
                vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', random_spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.actors.append(vehicle)

            # self.vehicle = self.world.get_actors()[0]
            # while not isinstance(self.vehicle, carla.Vehicle):
            #     print("Waiting for vehicle to spawn")
            #     self.world.tick()
            #     self.vehicle = self.world.get_actors().filter("vehicle.*")[0]
            self.ego_vehicle.set_transform(self.route[0][0])
        except Exception as e:
            print(e)
            vehicle_model = self.blueprint_lib.filter(self.configs["vehicle_model"])[0]
            spawn_transform = random.choice(self.world.get_map().get_spawn_points())
            self.ego_vehicle = self.world.spawn_actor(vehicle_model, spawn_transform)
        self.actors.append(self.ego_vehicle)


    def setupCamera(self, sensor_transform: carla.Transform):
        # This is the camera sensor
        if not hasattr(self, 'rgb_cam'):
            self.rgb_cam = self.blueprint_lib.find('sensor.camera.rgb')
            self.rgb_cam.set_attribute("image_size_x", f"{self.configs['cam_w']}")
            self.rgb_cam.set_attribute("image_size_y", f"{self.configs['cam_h']}")
            self.rgb_cam.set_attribute("fov", self.configs["cam_fov"])
            self.cam_sensor = self.world.spawn_actor(self.rgb_cam, sensor_transform, attach_to=self.ego_vehicle)
            self.actors.append(self.cam_sensor)
            self.cam_sensor.listen(self.processRawImage)


    def setupRadar(self, sensor_transform: carla.Transform):
        # This is the radar sensor
        if not hasattr(self, 'radar'):
            self.radar = self.blueprint_lib.find('sensor.other.radar')
            self.radar.set_attribute("horizontal_fov", self.configs["radar_h_fov"])
            self.radar.set_attribute("vertical_fov", self.configs["radar_v_fov"])
            self.radar_sensor = self.world.spawn_actor(self.radar, sensor_transform, attach_to=self.ego_vehicle)
            self.actors.append(self.radar_sensor)
            self.radar_raw = None
            self.radar_sensor.listen(self.processRawRadar)


    def setupGnss(self, sensor_transform: carla.Transform):
        # This is the gnss sensor
        if not hasattr(self, 'gnss'):
            self.gnss = self.blueprint_lib.find('sensor.other.gnss')
            self.gnss_sensor = self.world.spawn_actor(self.gnss, sensor_transform, attach_to=self.ego_vehicle)
            self.actors.append(self.gnss_sensor)
            self.gnss_latlonalt = None
            self.gnss_xyz = None
            self.gnss_sensor.listen(self.processRawGnss)


    def setupCollision(self, sensor_transform: carla.Transform):
        # This is the collision sensor
        if not hasattr(self, 'collision_sensor'):
            collision_sensor = self.blueprint_lib.find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(collision_sensor, sensor_transform, attach_to=self.ego_vehicle)
            self.actors.append(self.collision_sensor)
            self.collision_sensor.listen(self.processCollision)
            self.collide_detected = False
            self.has_collided = False


    def processCollision(self, event):
        self.collision_hist.append(event)
        self.collide_detected = True
        self.has_collided = True

    def processRawImage(self, sensor_data):
        img = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
        img = img.reshape((self.configs["cam_h"], self.configs["cam_w"], 4))[..., :3]
        self.front_camera = img


    def processRawRadar(self, sensor_data):
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(sensor_data), 4))
        self.radar_numpy = np.flip(points, 1)
        self.radar_raw = sensor_data


    def processRawGnss(self, sensor_data):
        self.gnss_latlonalt  = np.array([sensor_data.latitude, sensor_data.longitude, sensor_data.altitude], dtype=np.float32)
        self.gnss_xyz = np.array([sensor_data.transform.location.x,
                                  sensor_data.transform.location.y,
                                  sensor_data.transform.location.z], dtype=np.float32)


    def getDistance(self, waypoint: carla.Location) -> float:
        """
        Get the distance between the vehicle and the waypoint
        :param waypoint: a carla.Location object
        :return: the distance between the vehicle and the waypoint
        """
        vehicle_loc = self.ego_vehicle.get_location()
        distance = math.sqrt((vehicle_loc.x - waypoint.x) ** 2 + (vehicle_loc.y - waypoint.y) ** 2 + (vehicle_loc.z - waypoint.z) ** 2)
        return distance


    def checkReached(self, waypoint: carla.Location) -> Tuple[float, int]:
        """
        Check if the vehicle has reached the waypoint
        :param waypoint: a carla.Location object
        :return: distance to the waypoint, and a boolean indicating if the vehicle has reached the waypoint
        """
        distance = self.getDistance(waypoint)
        return distance, distance < self.configs["reach_threshold"]



    def generateRoute(self, destination: Tuple[float, float, float]):
        """
        Generate a route from current vehicle position to end
        :param destination: a tuple of (x, y, z) representing the destination
        :return: None
        """
        start_location = self.ego_vehicle.get_location()
        end_location = carla.Location(x=destination[0], y=destination[1], z=destination[2])
        self.route = self.route_planner.trace_route(start_location, end_location)


    def visualize(self):
        if self.radar_numpy is None:
            print("Empty radar data")
            return

        if self.front_camera is None:
            print("Empty camera data")
            return

        if self.gnss_latlonalt is None:
            print("Empty gnss data")
            return

        # region Copied from https://github.com/carla-simulator/carla/issues/5395
        velocity_range = 7.5  # m/s
        current_rot = self.radar_raw.transform.rotation
        for detect in self.radar_raw:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.world.debug.draw_point(
                self.ego_vehicle.get_location() + fw_vec,
                size=0.1,
                life_time=0.1,
                persistent_lines=False,
                color=carla.Color(r, g, b))
        # endregion

        # if attribute self.route exists, and it is not None:
        # Display the next 10 waypoints on the map
        if hasattr(self, "route") and self.route is not None:
            # draw route on the map
            for route_order in self.route[:10]:
                location = route_order[0].location
                self.world.debug.draw_point(location, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)

        # display current x y z on the screen
        temp = self.front_camera.copy()
        cv2.putText(temp, f"X: {self.gnss_xyz[0]:.2f}, Y: {self.gnss_xyz[1]:.2f}, Z: {self.gnss_xyz[2]:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)


        cv2.imshow("Display", temp)


    def getNextTargetPoint(self) -> np.ndarray:
        location = self.route[0][0].location
        return np.array([location.x, location.y, location.z], dtype=np.float32)


    def computeReward(self, car_now_distance: float, is_done: bool) -> float:
        """
        Compute the reward for the current step

        Officially, the total driving score is R * P
        R is the percentage of the route that is completed
        P is initialized to 1, and is reduced every time the vehicle did something wrong
        The decay is defined as following:
            Collision with pedestrians: 0.5
            Collision with vehicles: 0.6
            Collision with static elements: 0.7
            Running a red light: 0.7
            Driving a stop sign: 0.8
            Vehicle blocked for 4 min: 0.7
            Failure to maintain minimum speed: 0.7
            Failure to yield to emergency vehicle: 0.7

        I will split reward design into different stages:
            stage 1: Learn just to drive and follow the route and not collide with anything
            stage 2: Learn to avoid moving objects

        :return: the reward
        """

        if is_done:
            return 200

        reward = 0

        # If the vehicle is closer to the waypoint, displacement will be positive, result in positive reward
        # If the vehicle is further from the waypoint, displacement will be negative, result in negative reward
        displacement = self.car_prev_dist - car_now_distance
        reward += 0.1 * displacement

        if self.collide_detected:
            reward -= 10
            self.collide_detected = False

        return reward


    def step(self, action: VehicleAction) -> Tuple[VehicleState, float, bool, int]:
        """
        Take a step in the environment
        :param action: the action to take
        :return: a tuple of (state, reward, done, user_input), user_input is cv2.waitKey() result, None if not display
        """
        self.step_count += 1

        k = None
        if self.configs["display"]:
            self.visualize()
            k = cv2.waitKey(1)


        for i in range(self.STEP_TICKS):
            action.applyTo(self.ego_vehicle)
            self.world.tick()
            ego_trans = self.ego_vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=40),
                                                         carla.Rotation(pitch=-90, yaw=90)))

        done = False

        distance_to_next_point, reached = self.checkReached(self.route[0][0].location)
        if reached:
            self.car_prev_dist = distance_to_next_point
            self.route.pop(0)
            if len(self.route) == 0:
                done = True

        if self.step_count == self.MAX_N_STEPS:
            done = True

        if self.has_collided:
            done = True

        reward = self.computeReward(distance_to_next_point, done)

        self.car_prev_dist = distance_to_next_point

        return VehicleState(self.front_camera, self.radar_numpy, self.gnss_xyz, self.getNextTargetPoint()), reward, done, k

    def getSpeed(self):
        v = self.ego_vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return kmh

    def destroy(self):
        for actor in self.actors:
            actor.destroy()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    agent = HumanAgent()

    import yaml
    with open("../config_sparse_lidar.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)
    configs["display"] = False

    env = VehicleEnv(configs)
    env.reset()
    while True:
        action, control_signal, key_pressed = agent.getAction()
        if key_pressed == ord('q'):
            break
        if not control_signal:
            continue
        env.step(action)
