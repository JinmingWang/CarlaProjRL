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
    POINT_DIST = 5.0
    STEP_TICKS = 1
    MAX_N_STEPS = 5000
    front_camera = None
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
        self.step_count = 0
        self.collision_hist = []
        self.destroy()

        self.setupRouteVehicle()

        sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))

        self.setupCamera(sensor_transform)
        self.setupLidar(lidar_transform)
        self.setupGnss(sensor_transform)
        self.setupCollision(sensor_transform)
        self.setupIMU(sensor_transform)

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.car_prev_dist = 0.0

        self.episode_begin_time = time.time()

        while self.front_camera is None or self.lidar_numpy is None or self.gnss_xyz is None or self.imu_numpy is None:
            self.world.tick()
        self.world.tick()


    def setupRouteVehicle(self) -> None:
        """
        Generate a new route, spawn a new vehicle at the start point, spawn other vehicles
        :return: None
        """
        random_points = np.random.choice(self.world.get_map().get_spawn_points(), self.configs["n_other_actors"] + 2)
        self.route = self.generateRoute(random_points[-2], random_points[-1])

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

        self.car_prev_dist = self.getDistance(self.route[0][0].transform.location)


    def setupCamera(self, sensor_transform: carla.Transform) -> None:
        """
        Setup the camera sensor
        :param sensor_transform: the transform of the sensor
        """
        self.rgb_cam = self.blueprint_lib.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.configs['cam_w']}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.configs['cam_h']}")
        self.rgb_cam.set_attribute("fov", self.configs["cam_fov"])
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.cam_sensor)
        self.cam_sensor.listen(self.processRawImage)


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
        self.lidar_numpy = np.zeros((self.configs["max_points_stored"], 4))
        self.lidar_np_head_index = 0
        self.lidar_sensor.listen(self.processRawLidar)


    def setupGnss(self, sensor_transform: carla.Transform) -> None:
        """
        Setup the gnss (GPS) sensor
        :param sensor_transform: the transform of the sensor
        """
        self.gnss = self.blueprint_lib.find('sensor.other.gnss')
        self.gnss_sensor = self.world.spawn_actor(self.gnss, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.gnss_sensor)
        self.gnss_latlonalt = None
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

        self.imu_numpy = np.zeros(7, np.float32)
        self.imu_sensor.listen(self.processRawIMU)


    def setupCollision(self, sensor_transform: carla.Transform) -> None:
        """
        Setup the collision sensor
        :param sensor_transform: the transform of the sensor
        """
        collision_sensor = self.blueprint_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, sensor_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(self.processCollision)
        self.collide_detected = False
        self.has_collided = False


    def processCollision(self, event) -> None:
        """
        Process the collision event
        :param event: collision event
        """
        self.collision_hist.append(event)
        self.collide_detected = True
        self.has_collided = True

    def processRawImage(self, sensor_data) -> None:
        """
        Process RGB camera image
        :param sensor_data: raw data from the camera sensor
        """
        img = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
        img = img.reshape((self.configs["cam_h"], self.configs["cam_w"], 4))[..., :3]
        self.front_camera = img


    def processRawLidar(self, sensor_data) -> None:
        """
        Process raw lidar data
        :param sensor_data: raw data from the radar sensor
        """
        self.lidar_raw = sensor_data

        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        n_points = int(points.shape[0] / 4)
        temp_lidar_numpy = np.reshape(points, (n_points, 4))[:self.configs["max_points_each_time"]]
        n_points = min(n_points, self.configs["max_points_each_time"])

        start = self.lidar_np_head_index
        end = min(self.lidar_np_head_index + n_points, self.configs["max_points_stored"])
        self.lidar_numpy[start: end] = temp_lidar_numpy[: end - start]
        self.lidar_np_head_index = end % self.configs["max_points_stored"]



    def processRawGnss(self, sensor_data) -> None:
        """
        Process raw GPS data
        :param sensor_data: raw data from the GPS sensor
        """
        self.gnss_latlonalt  = np.array([sensor_data.latitude, sensor_data.longitude, sensor_data.altitude], dtype=np.float32)
        self.gnss_xyz = np.array([sensor_data.transform.location.x,
                                  sensor_data.transform.location.y,
                                  sensor_data.transform.location.z], dtype=np.float32)


    def processRawIMU(self, sensor_data) -> None:
        """
        Process raw IMU data
        :param sensor_data: raw data from the IMU sensor
        """
        self.imu_numpy = np.array([sensor_data.accelerometer.x,
                                      sensor_data.accelerometer.y,
                                      sensor_data.accelerometer.z,
                                      sensor_data.gyroscope.x,
                                      sensor_data.gyroscope.y,
                                      sensor_data.gyroscope.z,
                                      sensor_data.compass], dtype=np.float32)


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



    def generateRoute(self, source: carla.Transform, destination: carla.Transform):
        """
        Generate a route from current vehicle position to end
        :param destination: a tuple of (x, y, z) representing the destination
        :return: None
        """
        start_location = source.location
        end_location = destination.location
        return self.route_planner.trace_route(start_location, end_location)


    def visualize(self) -> None:
        """
        Visualize the sensors, including display XYZ from GPS, point cloud from radar, and RGB image from camera
        """
        if self.lidar_numpy is None:
            print("Empty radar data")
            return

        if self.front_camera is None:
            print("Empty camera data")
            return

        if self.gnss_latlonalt is None:
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

        # if attribute self.route exists, and it is not None:
        # Display the next 10 waypoints on the map
        if hasattr(self, "route") and self.route is not None:
            # draw route on the map
            for route_order in self.route[:10]:
                location = route_order[0].transform.location
                location.z += 3
                self.world.debug.draw_point(location, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)

        # display current x y z on the screen
        if self.configs["show_window"]:
            temp = self.front_camera.copy()
            if self.configs["show_xyz"]:
                cv2.putText(temp, f"X: {self.gnss_xyz[0]:.2f}, Y: {self.gnss_xyz[1]:.2f}, Z: {self.gnss_xyz[2]:.2f}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

                target_location = self.route[0][0].transform.location
                cv2.putText(temp, f"TX: {target_location.x:.2f}, TY: {target_location.y:.2f}, TZ: {target_location.z:.2f}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)


            cv2.imshow("Display", temp)


    def getNextTargetPoint(self) -> np.ndarray:
        """
        Get the next target point from the route
        :return: a numpy array of shape (3, ) representing the next target point
        """
        if len(self.route) == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        location = self.route[0][0].transform.location
        return np.array([location.x, location.y, location.z], dtype=np.float32)


    def computeReward(self, reached: bool, distance_to_next_point: float) -> float:
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
        reward = 0

        # positive if further
        # negative if closer
        reward += (self.car_prev_dist - distance_to_next_point) / self.POINT_DIST

        # +1 if reached
        if reached:
            reward += self.configs["point_reach_reward"]

        # -3 if collision
        if self.collide_detected:
            reward -= self.configs["collide_penalty"]
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
            if action is not None:
                action.applyTo(self.ego_vehicle)
            self.world.tick()
            ego_trans = self.ego_vehicle.get_transform()
            location = ego_trans.location
            location.z = 15
            rotation = ego_trans.rotation
            rotation.pitch = -60
            # rotation.yaw = 60
            rotation.roll = 0

            # The spectator should behind the vehicle, no matter what orientation the vehicle is facing
            fw_vec = carla.Location(
                x=math.cos(math.radians(rotation.yaw)) * 3,
                y=math.sin(math.radians(rotation.yaw)) * 3,
                z=0
            )
            location = location - fw_vec

            self.spectator.set_transform(carla.Transform(location, rotation))

        done = False

        distance_to_next_point, reached = self.checkReached(self.route[0][0].transform.location)
        if reached:
            self.car_prev_dist = distance_to_next_point
            self.route.pop(0)
            if len(self.route) == 0:
                done = True

        if self.step_count == self.MAX_N_STEPS:
            done = True

        # if self.has_collided:
        #     done = True

        reward = self.computeReward(reached, distance_to_next_point)


        self.car_prev_dist = distance_to_next_point

        return VehicleState(self.front_camera, self.lidar_numpy, self.gnss_xyz, self.getNextTargetPoint(), self.imu_numpy), reward, done, k


    def getSpeed(self) -> int:
        """
        Get the speed of the ego vehicle
        :return: the speed in km/h
        """
        v = self.ego_vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return kmh


    def destroy(self) -> None:
        """
        Destroy eho_vehicle, other vehicles and sensors
        """
        if hasattr(self, "actors"):
            for actor in self.actors:
                actor.destroy()
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
            env.step(None)
        else:
            env.step(action)
