import pickle

import cv2
import numpy as np
import torch
import random
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


class PracticeFieldEnv:
    MAX_N_STEPS = 20000
    lidar_numpy = None
    lidar_map_size = 127

    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs
        self.client = carla.Client(configs["host"], configs["port"])
        self.client.set_timeout(configs["timeout"])
        self.traffic_manager = self.client.get_trafficmanager(configs["traffic_manager_port"])
        self.world = self.client.get_world()
        self.loadWorld(configs["world"])

        self.practice_fields = []
        polygons = np.array(configs["practice_fields"])  # (N, 4, 2)
        for polygon in polygons:
            sorted_x = np.sort(polygon[:, 0])
            sorted_y = np.sort(polygon[:, 1])
            # (min_x, min_y, max_x, max_y)
            field_box = np.array([sorted_x[1], sorted_y[1], sorted_x[-2], sorted_y[-2]])
            self.practice_fields.append(field_box)

        # vehicle is initialized at the center of the field
        self.init_loc = carla.Location(x=(self.practice_fields[0][0] + self.practice_fields[0][2]) / 2,
                                       y=(self.practice_fields[0][1] + self.practice_fields[0][3]) / 2,
                                       z=2.0)

        self.point_sparsity = configs["point_sparsity"]
        self.route_planner = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=self.point_sparsity)

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
        self.collision_count = 0
        self.reach_count = 0
        self.total_reward = 0
        self.step_count = 0
        self.smooth_speed = 0.0
        self.destroy()

        self.setupRouteVehicle()

        gps_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=self.configs["lidar_height"]))

        self.setupLidar(lidar_transform)
        self.setupGnss(gps_transform)
        self.setupCollision(gps_transform)
        self.setupIMU(gps_transform)

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while self.lidar_numpy is None or self.gnss_xyz is None or self.compass is None:
            self.world.tick()
        self.world.tick()

        self.ego_prev_dist = self.getDistanceTo(self.route[0])

        ego_loc = self.ego_vehicle.get_location()
        self.location_record = [(-999, -999)] * 999
        self.location_record.append((ego_loc.x, ego_loc.y))

        self.lidar_cache = [np.zeros((self.lidar_map_size, self.lidar_map_size, 3), dtype=np.uint8) for _ in range(8)]
        self.lidar_map
        self.lidar_map
        self.lidar_map
        self.lidar_map
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
        random_points = np.random.choice(self.world.get_map().get_spawn_points(), self.configs["n_other_actors"])

        # Generate route
        self.route = self.generateRoute()
        self.route_size = len(self.route)

        # spawn ego vehicle at the start point
        vehicle_model = random.choice(self.blueprint_lib.filter(self.configs["vehicle_model"]))
        ego_rot = carla.Rotation(pitch=0, yaw=-90, roll=0)
        ego_transform = carla.Transform(self.init_loc, ego_rot)
        self.ego_vehicle = self.world.spawn_actor(vehicle_model, ego_transform)

        self.actors.append(self.ego_vehicle)

        # spawn vehicles and set them to autopilot
        for i in range(self.configs["n_other_actors"]):
            vehicle_model = random.choice(self.blueprint_lib.filter("vehicle"))
            # vehicle_model = self.blueprint_lib.find("vehicle.bh.crossbike")
            try:
                vehicle = self.world.spawn_actor(vehicle_model, random_points[i])
            except:
                continue
            # set autopilot
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.traffic_manager.ignore_lights_percentage(vehicle, 0.8)
            self.actors.append(vehicle)


    def setupLidar(self, sensor_transform: carla.Transform) -> None:
        """
        Set up the lidar sensor
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
        current_lidar = self.lidar_numpy
        temp_map = np.zeros((self.lidar_map_size, self.lidar_map_size, 3), dtype=np.int32)

        half_size = self.lidar_map_size // 2

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
            target_col = int(np.clip(target_col + half_size, 1, self.lidar_map_size - 2))
            target_row = int(np.clip(target_row + half_size, 1, self.lidar_map_size - 2))
        except ValueError:
            # Sometimes NaN, just return previous 
            print("NaN detected obtaining local lidar map")
            return np.concatenate(self.lidar_cache, axis=-1)

        # eliminate rows contain NaN
        nan_rows = np.isnan(current_lidar[:, 0])
        current_lidar = current_lidar[~nan_rows]

        x = current_lidar[:, 0] / pixel_resolution
        y = current_lidar[:, 1] / pixel_resolution

        obj_height = np.clip((current_lidar[:, 2] + self.configs["lidar_height"]) * 255, 0, 1000)

        col = np.int32(np.clip(y + half_size, 0, self.lidar_map_size - 1))
        row = np.int32(np.clip(half_size - x, 0, self.lidar_map_size - 1))
        # yaw: in place rotation
        # pitch: head up and down
        # roll: scroll left and right
        # Lidar points are affected by pitch and roll, we need to rotate them back to the horizontal plane
        vehicle_rotation = self.ego_vehicle.get_transform().rotation
        pitch = math.radians(vehicle_rotation.pitch)
        roll = math.radians(vehicle_rotation.roll)
        roll_adjust = np.clip(- np.tan(roll) * pixel_resolution * (half_size - col) * 255, 0, 1000)
        pitch_adjust = np.clip(- np.tan(pitch) * pixel_resolution * (half_size - row) * 255, 0, 1000)

        obj_height = obj_height - roll_adjust - pitch_adjust

        # Draw target point as blue
        temp_map[target_row, target_col, 0] = 255

        # draw lidar points
        temp_map[row, col, 2] = obj_height

        # Draw a representation of the ego vehicle
        temp_map[half_size - 7:half_size + 7, half_size - 3:half_size + 3, 1] = 255

        blue = temp_map[:, :, 0]
        blue = cv2.GaussianBlur(np.float32(blue), ksize=(41, 41), sigmaX=10.0)
        temp_map[:, :, 0] = np.int32(blue / blue.max() * 255)

        self.lidar_cache.append(temp_map)
        self.lidar_cache.pop(0)

        # 8 * [63, 63, 3] -> max -> [63, 63, 3]
        return np.max(self.lidar_cache, axis=0)


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


    def generateRoute(self, route_length: int = 100) -> List[carla.Location]:
        """
        Generate a route by randomly select points within practice field
        field_box = (min_x, min_y, max_x, max_y)
        and ensure that the distance between adjacent waypoints is larger than reaching threshold
        """
        reach_distance = self.configs["reach_threshold"]
        route = [self.init_loc]
        for _ in range(route_length - 1):

            field_box = self.practice_fields[random.randint(len(self.practice_fields))]
            x = np.random.uniform(field_box[0], field_box[2])
            y = np.random.uniform(field_box[1], field_box[3])
            distance = math.sqrt((route[-1].x - x) ** 2 + (route[-1].y - y) ** 2)
            while distance < reach_distance:
                field_box = self.practice_fields[random.randint(len(self.practice_fields))]
                x = np.random.uniform(field_box[0], field_box[2])
                y = np.random.uniform(field_box[1], field_box[3])
                distance = math.sqrt((route[-1].x - x) ** 2 + (route[-1].y - y) ** 2)
            route.append(carla.Location(x, y, 0))

        return route[1:]



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
                point_trans.z += 1.0

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
        if reward >= 0.5:
            self.world.debug.draw_string(reward_location, f"{reward:.2f}", draw_shadow=False,
                                         color=carla.Color(0, 255, 0), life_time=reward/5, persistent_lines=False)
        elif reward > 0:
            self.world.debug.draw_string(reward_location, "+", draw_shadow=False,
                                         color=carla.Color(0, 255, 0), life_time=0.1, persistent_lines=False)
        elif reward == 0:
            pass
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
            point_location = carla.Location(self.route[0].x, self.route[0].y, ego_location.z)
            self.world.debug.draw_point(point_location, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)

        if self.configs["show_lidar_map"]:
            self.lidar_map
            img = np.uint8(np.clip(np.max(self.lidar_cache, axis=0), 0, 510) / 2)
            # img = np.uint8(np.clip(self.lidar_cache[-1], 0, 510) / 2)
            temp = cv2.resize(img, dsize=(255, 255), interpolation=cv2.INTER_NEAREST)
            cv2.line(temp, (127, 127), (127, 0), (255, 255, 255), 1)

            cv2.imshow("lidar map", temp)

        # Draw fields
        # (min_x, min_y, max_x, max_y)
        for field_box in self.practice_fields:
            loc_0 = carla.Location(field_box[0], field_box[1], ego_location.z + 0.5)
            loc_1 = carla.Location(field_box[0], field_box[3], ego_location.z + 0.5)
            loc_2 = carla.Location(field_box[2], field_box[3], ego_location.z + 0.5)
            loc_3 = carla.Location(field_box[2], field_box[1], ego_location.z + 0.5)

            self.world.debug.draw_line(loc_0, loc_1, thickness=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
            self.world.debug.draw_line(loc_1, loc_2, thickness=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
            self.world.debug.draw_line(loc_2, loc_3, thickness=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
            self.world.debug.draw_line(loc_3, loc_0, thickness=0.1, color=carla.Color(255, 0, 0), life_time=0.1)


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

        ego_loc = self.ego_vehicle.get_location()
        self.location_record.append((ego_loc.x, ego_loc.y))
        self.location_record.pop(0)

        # If the ego vehicle does not move at least 2 meters in 50 steps, then apply penalty
        displacement_1000_steps = math.sqrt((self.location_record[-1][0] - self.location_record[0][0]) ** 2 + \
                                          (self.location_record[-1][1] - self.location_record[0][1]) ** 2)
        if displacement_1000_steps < 2:
            reward -= 0.5

        # +1 if reached, and encourage forward reach,
        # backward reach is also ok, but not as good
        if reached:
            if action.speed >= 0:
                reward += self.configs["point_reach_reward"]
            else:
                reward += self.configs["point_reach_reward"] * 0.75
            # if a target is reached, then all locations records are reset
            # in this way, we allow the vehicle to revisit past positions
            self.location_record[:999] = [(-999, -999)] * 999
            self.reach_count += 1
        else:
            # positive if further
            # negative if closer
            reward += (self.ego_prev_dist - distance_to_next_point) / self.point_sparsity

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
            self.collision_count += 1

        # if self.smooth_speed < 0.01:
        #     reward += self.configs["stop_penalty"]

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
        location.z = 15 + max(int(location.z), 0)
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

        self.total_reward += reward

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


    def saveRoute(self, path: str) -> None:
        # Save the route to a file
        route = self.route[:]
        # The first point is the current location of the ego vehicle
        # This enables the user to customize the start point of the route

        ego_trans = self.ego_vehicle.get_transform()
        location = ego_trans.location
        rotation = ego_trans.rotation
        info_dict = {
            "start_location": (location.x, location.y, location.z),
            "start_rotation": (rotation.pitch, rotation.roll, rotation.yaw),
            "xyz_list": [(location.x, location.y, location.z) for location in route]
        }
        with open(path, "wb") as out_file:
            pickle.dump(info_dict, out_file)

    @property
    def completion_percentage(self) -> float:
        return self.reach_count / self.route_size

    @property
    def normalized_score(self) -> float:
        # every waypoint reaching gives 1 reward
        # So the total reward is proportional to the number of waypoints
        # The absolute reward may differ due to route length
        return self.total_reward / self.route_size


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
    agent = HumanAgent(demonstrate_mode=False)

    import yaml

    with open("config_practice_1.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)

    env = PracticeFieldEnv(configs)
    env.reset()
    while True:
        action, control_signal, key_pressed = agent.getAction()
        if key_pressed == ord("p"):
            print(env.ego_vehicle.get_transform())
        if key_pressed == ord('s'):
            env.saveRoute("route.pkl")
        if key_pressed == ord('q'):
            break
        if key_pressed == ord('r'):
            env.reset()
        if not control_signal:
            s, r, done, k = env.step(VehicleAction.getStopAction())
        else:
            s, r, done, k = env.step(action)

        if done:
            env.reset()
