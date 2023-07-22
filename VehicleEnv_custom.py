import cv2
import numpy as np
import torch
import random
import time
from typing import *
import math

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


class VehicleEnv:
    POINT_DIST = 10.0
    STEP_TICKS = 1
    MAX_N_STEPS = 5000
    lidar_numpy = None


    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs
        self.client = carla.Client(configs["host"], configs["port"])
        self.client.set_timeout(configs["timeout"])
        self.traffic_manager = self.client.get_trafficmanager(configs["traffic_manager_port"])
        self.world = self.client.get_world()
        self.loadWorld(configs["world"])

        self.route_planner = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=self.POINT_DIST)

        self.blueprint_lib = self.world.get_blueprint_library()


    def loadWorld(self, world_name: str):
        print(f"Loading world: {world_name}")
        self.client.load_world(world_name)
        self.world = self.client.get_world()

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

        self.spectator = CarlaDataProvider.get_world().get_spectator()

        # Wait for the world to be ready
        self.world.tick()
        print("World Ready!")


    def reset(self) -> None:
        """
        Reset the environment, clear and reset vehicle, all sensors and route
        :return: None
        """
        self.reach_count = 0
        self.step_count = 0
        self.smooth_speed = 0.0
        self.destroy()

        self.setupRouteVehicle()

        gps_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))

        self.setupLidar(lidar_transform)
        self.setupGnss(gps_transform)
        self.setupCollision(gps_transform)
        self.setupIMU(gps_transform)

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while self.lidar_numpy is None or self.gnss_xyz is None or self.compass is None:
            self.world.tick()
        self.world.tick()

        self.ego_prev_dist = self.getDistanceTo(self.route[0])

        self.lidar_cache = [np.zeros((63, 63, 3), dtype=np.uint8) for _ in range(4)]
        self.lidar_map
        self.lidar_map
        self.lidar_map
        self.lidar_map


    def setupRouteVehicle(self) -> None:
        """
        Generate a new route, spawn a new vehicle at the start point, spawn other vehicles
        :return: None
        """
        # get random points, n_other_actors points for other vehicles and pedestrians
        # 2 points for route start and end
        random_points = np.random.choice(self.world.get_map().get_spawn_points(), self.configs["n_other_actors"] + 2)
        self.route = self.generateRoute(random_points[-2], random_points[-1])
        # Occasionally the route is invalid, regenerate until it is valid
        while len(self.route) == 0:
            random_points = np.random.choice(self.world.get_map().get_spawn_points(),
                                             self.configs["n_other_actors"] + 2)
            self.route = self.generateRoute(random_points[-2], random_points[-1])

        # spawn ego vehicle at the start point
        vehicle_model = random.choice(self.blueprint_lib.filter(self.configs["vehicle_model"]))
        self.ego_vehicle = self.world.spawn_actor(vehicle_model, random_points[-2])
        self.actors.append(self.ego_vehicle)

        # spawn vehicles and set them to autopilot
        for i in range(self.configs["n_other_actors"]):
            random_model = random.choice(self.blueprint_lib.filter("vehicle"))
            try:
                vehicle = self.world.spawn_actor(random_model, random_points[i])
            except:
                continue
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.actors.append(vehicle)


    def setupLidar(self, sensor_transform: carla.Transform) -> None:
        """
        Setup the lidar sensor
        :param sensor_transform:  the transform of the sensor
        """
        self.lidar = self.blueprint_lib.find('sensor.lidar.ray_cast')
        self.lidar.set_attribute('noise_stddev', self.configs['noise_stddev'])
        self.lidar.set_attribute('channels', self.configs['channels'])
        self.lidar.set_attribute('range', self.configs['range'])
        self.lidar.set_attribute('rotation_frequency', self.configs['rotation_frequency'])
        self.lidar.set_attribute('points_per_second', self.configs['lidar_pps'])
        self.lidar.set_attribute('upper_fov', self.configs['lidar_fov_upper'])
        self.lidar.set_attribute('lower_fov', self.configs['lidar_fov_lower'])

        self.lidar_sensor = self.world.spawn_actor(self.lidar, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.lidar_sensor)
        self.lidar_raw = None
        self.lidar_numpy = None
        self.lidar_sensor.listen(self.processRawLidar)


    def setupGnss(self, sensor_transform: carla.Transform) -> None:
        """
        Setup the gnss (GPS) sensor
        :param sensor_transform: the transform of the sensor
        """
        self.gnss = self.blueprint_lib.find('sensor.other.gnss')
        self.gnss_sensor = self.world.spawn_actor(self.gnss, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.gnss_sensor)
        self.gnss_xyz = None
        self.gnss_sensor.listen(self.processRawGnss)


    def setupIMU(self, sensor_transform: carla.Transform) -> None:
        """
        Setup the IMU sensor
        :param sensor_transform: the transform of the sensor
        """
        self.imu = self.blueprint_lib.find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(self.imu, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.imu_sensor)

        self.compass = None
        self.imu_sensor.listen(self.processRawIMU)


    def setupCollision(self, sensor_transform: carla.Transform) -> None:
        """
        Set up the collision sensor
        :param sensor_transform: the transform of the sensor
        """
        collision_sensor = self.blueprint_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(self.processCollision)
        self.collide_detected = False


    def processCollision(self, event) -> None:
        """
        Process the collision event
        """
        self.collide_detected = True


    def processRawLidar(self, sensor_data) -> None:
        """
        Process raw lidar data
        :param sensor_data: raw data from the radar sensor
        """
        self.lidar_raw = sensor_data
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        n_points = int(points.shape[0] / 4)
        self.lidar_numpy = np.reshape(points, (n_points, 4))


    @property
    def lidar_map(self):
        temp_map = np.zeros((63, 63, 3), dtype=np.uint8)

        # every pixel is 0.8m
        pixel_resolution = 0.3  # meter
        # temp_map[31, 31] = 255
        next_point = self.getNextTargetPoint()
        relative_target_x = (next_point[0] - self.gnss_xyz[0]) / pixel_resolution
        relative_target_y = (next_point[1] - self.gnss_xyz[1]) / pixel_resolution

        theta = - self.compass  # compass in radians, North = 0.0 = 2pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        target_col = cos * relative_target_x - sin * relative_target_y
        target_row = sin * relative_target_x + cos * relative_target_y
        try:
            target_col = int(np.clip(target_col + 31, 1, 61))
            target_row = int(np.clip(target_row + 31, 1, 61))
        except ValueError:
            # Sometimes NaN, just return previous 
            print("NaN detected obtaining local lidar map")
            return np.concatenate(self.lidar_cache, axis=-1)

        # eliminate rows contain NaN
        nan_rows = np.isnan(self.lidar_numpy[:, 0])
        self.lidar_numpy = self.lidar_numpy[~nan_rows]

        x = self.lidar_numpy[:, 0] / pixel_resolution
        y = self.lidar_numpy[:, 1] / pixel_resolution

        obj_height = np.clip((self.lidar_numpy[:, 2] + 2.5) * 255, 0, 255)

        col = np.int32(np.clip(y + 31, 0, 62))
        row = np.int32(np.clip(31 - x, 0, 62))

        # Draw target point as blue
        temp_map[target_row - 1:target_row + 2, target_col - 1:target_col + 2, 0] = 255

        # draw lidar points
        temp_map[row, col, 2] = obj_height
        temp_map[row, col, 1] = 255 - obj_height

        # Draw a representation of the ego vehicle
        vehicle_length = 6
        vehicle_width = 2
        temp_map[31 - vehicle_length:32 + vehicle_length, 31 - vehicle_width:32 + vehicle_width] = 255

        blur_map = cv2.GaussianBlur(temp_map, (5, 5), 0)
        zero_mask = np.all(temp_map == 0, axis=-1)
        temp_map[zero_mask] = blur_map[zero_mask]

        self.lidar_cache.append(temp_map)
        self.lidar_cache.pop(0)

        # 4 * [63, 63, 3] -> [63, 63, 12]
        return np.concatenate(self.lidar_cache, axis=-1)


    def processRawGnss(self, sensor_data) -> None:
        """
        Process raw GPS data
        :param sensor_data: raw data from the GPS sensor
        """
        self.gnss_xyz = np.array([sensor_data.transform.location.x,
                                  sensor_data.transform.location.y,
                                  sensor_data.transform.location.z], dtype=np.float32)


    def processRawIMU(self, sensor_data) -> None:
        """
        Process raw IMU data
        :param sensor_data: raw data from the IMU sensor
        """
        self.compass = float(sensor_data.compass)


    def getDistanceTo(self, waypoint: carla.Location) -> float:
        """
        Get the 2D distance between the vehicle and the waypoint
        :param waypoint: a carla.Location object
        :return: the distance between the vehicle and the waypoint
        """
        vehicle_loc = self.ego_vehicle.get_location()
        distance = math.sqrt((vehicle_loc.x - waypoint.x) ** 2 + (vehicle_loc.y - waypoint.y) ** 2)
        return distance


    def checkReached(self, waypoint: carla.Location) -> Tuple[float, bool]:
        """
        Check if the vehicle has reached the waypoint
        :param waypoint: a carla.Location object
        :return: distance to the waypoint, and a boolean indicating if the vehicle has reached the waypoint
        """
        distance = self.getDistanceTo(waypoint)
        return distance, distance < self.configs["reach_threshold"]


    def generateRoute(self, source: carla.Transform, destination: carla.Transform) -> List[carla.Location]:
        """
        Generate a route from current vehicle position to end
        :param destination: a tuple of (x, y, z) representing the destination
        :return: None
        """
        start_location = source.location
        end_location = destination.location
        route = self.route_planner.trace_route(start_location, end_location)
        if len(route) == 0:
            return route

        locations = [route[0][0].transform.location]
        for waypoint in route[1:]:
            # if this waypoint has different location from the previous point
            if waypoint[0].transform.location != locations[-1]:
                locations.append(waypoint[0].transform.location)
        return locations


    def visualize(self, reward) -> None:
        """
        Visualize the sensors, including display XYZ from GPS, point cloud from radar, and RGB image from camera
        """
        if self.lidar_numpy is None:
            print("Empty radar data")
            return

        if self.gnss_xyz is None:
            print("Empty gnss data")
            return

        # Visualize Lidar data
        if self.configs["show_lidar"]:
            lidar_loc = self.lidar_raw.transform.location
            lidar_rot = self.lidar_raw.transform.rotation
            for point in self.lidar_numpy:
                x, y, z, intensity = point

                fw_vec = carla.Vector3D(x=float(x), y=float(y), z=float(z))
                carla.Transform(
                    carla.Location(),
                    carla.Rotation(
                        pitch=lidar_rot.pitch,
                        yaw=lidar_rot.yaw,
                        roll=lidar_rot.roll)).transform(fw_vec)

                point_trans = fw_vec + lidar_loc
                point_trans.z += 0.1

                self.world.debug.draw_point(
                    point_trans,
                    size=0.1,
                    life_time=0.1,
                    persistent_lines=False,
                    color=carla.Color(r=255, g=0, b=0)
                )

        # show reward
        ego_location = self.ego_vehicle.get_location()
        reward_location = carla.Location(ego_location.x, ego_location.y, ego_location.z+3)
        if reward > 0.5:
            self.world.debug.draw_string(reward_location, f"{reward:.2f}", draw_shadow=False,
                                         color=carla.Color(0, 255, 0), life_time=reward/5, persistent_lines=False)
        elif reward >= 0:
            self.world.debug.draw_string(reward_location, "+", draw_shadow=False,
                                         color=carla.Color(0, 255, 0), life_time=0.1, persistent_lines=False)
        elif reward > -0.5:
            self.world.debug.draw_string(reward_location, "-", draw_shadow=False,
                                         color=carla.Color(255, 0, 0), life_time=0.1, persistent_lines=False)
        else:
            self.world.debug.draw_string(reward_location, f"{reward:.2f}", draw_shadow=False,
                                         color=carla.Color(255, 0, 0), life_time=-reward/5, persistent_lines=False)
        # show reward as text

        # if attribute self.route exists, and it is not None:
        # Display the next 10 waypoints on the map
        if hasattr(self, "route") and self.route is not None:
            # draw route on the map
            for location in self.route[:10]:
                point_location = carla.Location(location.x, location.y, ego_location.z)
                self.world.debug.draw_point(point_location, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)

        if self.configs["show_lidar_map"]:
            self.lidar_map
            temp = cv2.resize(self.lidar_cache[-1], dsize=(255, 255), interpolation=cv2.INTER_NEAREST)
            cv2.line(temp, (127, 127), (127, 0), (255, 255, 255), 1)

            cv2.imshow("lidar map", temp)


    def getNextTargetPoint(self) -> np.ndarray:
        """
        Get the next target point from the route
        :return: a numpy array of shape (3, ) representing the next target point
        """
        if len(self.route) == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        location = self.route[0]
        return np.array([location.x, location.y, location.z], dtype=np.float32)


    def computeReward(self, action: VehicleAction, reached: bool, distance_to_next_point: float) -> float:
        """
        Compute the reward for the current step

        Current design:
        Move towards the next point: +distance moved
        Move away from the next point: -distance moved
        Reach the next point: +1
        Collide with Any object: -1 * speed_factor
        speed = 0: -0.01


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

        :return: the reward
        """
        reward = 0

        # +1 if reached, and encourage forward reach,
        # backward reach is also ok, but not as good
        if reached:
            if action.speed >= 0:
                reward += self.configs["point_reach_reward"]
            else:
                reward += self.configs["point_reach_reward"] * 0.75
            self.reach_count += 1
        else:
            # positive if further
            # negative if closer
            reward += (self.ego_prev_dist - distance_to_next_point) / self.POINT_DIST

        if self.collide_detected:
            # Collision speed < 20 km/h: speed_factor = 1
            # Collision speed = 40 km/h: speed_factor = 5.578
            # Collision speed = 60 km/h: speed_factor = 14.815
            # Collision speed = 80 km/h: speed_factor = 19.096
            # Collision speed = inf km/h: speed_factor = 20
            # Agent, please brake when you drive too fast!
            speed_factor = max(20 / (1 + math.exp(4.95 - 0.1 * self.smooth_speed)), 1)     # speed is in km/h
            # If during collision, the the speed is still increasing, then double penalty
            if abs(action.speed) > 0.5:
                speed_factor *= 2
            reward += self.configs["collide_penalty"] * speed_factor
            self.collide_detected = False

        if self.smooth_speed < 0.01:
            reward += self.configs["stop_penalty"]

        return reward


    def step(self, action: VehicleAction) -> Tuple[VehicleState, float, bool, int]:
        """
        Take a step in the environment
        :param action: the action to take
        :return: a tuple of (state, reward, done, user_input), user_input is cv2.waitKey() result, None if not display
        """

        self.step_count += 1

        if action is not None:
            action.applyTo(self.ego_vehicle)
        self.world.tick()
        ego_trans = self.ego_vehicle.get_transform()
        location = ego_trans.location
        location.z = 15
        rotation = ego_trans.rotation
        rotation.pitch = -60
        # rotation.yaw = -90   # face to north
        rotation.roll = 0

        # The spectator should behind the vehicle, no matter what orientation the vehicle is facing
        fw_vec = carla.Location(
            x=math.cos(math.radians(rotation.yaw)) * 3,
            y=math.sin(math.radians(rotation.yaw)) * 3,
            z=0
        )
        location = location - fw_vec

        self.spectator.set_transform(carla.Transform(location, rotation))

        self.updateSpeed()

        done = False

        distance_to_next_point, reached = self.checkReached(self.route[0])
        reward = self.computeReward(action, reached, distance_to_next_point)

        if reached:
            self.route.pop(0)
            if len(self.route) == 0:
                done = True
            else:
                distance_to_next_point, reached = self.checkReached(self.route[0])
                self.ego_prev_dist = distance_to_next_point

        if self.step_count == self.MAX_N_STEPS:
            done = True

        k = None
        if self.configs["display"]:
            self.visualize(reward)
            k = cv2.waitKey(1)

        self.ego_prev_dist = distance_to_next_point

        return VehicleState(self.lidar_map, self.gnss_xyz, self.getNextTargetPoint(), self.compass, self.smooth_speed), reward, done, k


    def updateSpeed(self) -> None:
        """
        Get the speed of the ego vehicle
        After several tests, speed of 80 - 90 is the max speed within the map
        speed of 40 is already fast
        :return: the speed in km/h
        """
        v = self.ego_vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self.smooth_speed = self.smooth_speed * 0.2 + kmh * 0.8


    def destroy(self) -> None:
        """
        Destroy eho_vehicle, other vehicles and sensors
        """
        if hasattr(self, "actors"):
            for actor in self.actors:
                try:
                    actor.destroy()
                except:
                    print("Actor destroy failed, it's probably already gone")
        self.actors = []
        try:
            cv2.destroyWindow("Display")
        except:
            pass
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    agent = HumanAgent()

    import yaml


    with open("config_1.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)

    env = VehicleEnv(configs)
    env.reset()
    while True:
        action, control_signal, key_pressed = agent.getAction()
        if key_pressed == ord('q'):
            break
        if key_pressed == ord('r'):
            env.reset()
        if not control_signal:
            env.step(VehicleAction.getStopAction())
        else:
            env.step(action)
