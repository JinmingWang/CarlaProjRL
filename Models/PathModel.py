import torch

from Models.ModelUtils import *
from Models.RLPathModel import LocalPathPlanner
import numpy as np
import cv2

"""
This agent just greedy follow the next point in the path.
Does not care about collision (Assumes no obstacle)
"""

class PathModel():
    def __init__(self, *args):
        super().__init__()
        self.grid_world = None

    @staticmethod
    def getSteer(dx, dy):
        # Step 1. Compute steer to let the vehicle directly point to the next point in the path
        steer_rad = torch.atan2(dx, - dy)
        steer_angle = steer_rad * 180 / torch.pi
        greedy_steer = func.hardtanh(steer_angle / 70, -1, 1)
        return greedy_steer.detach(), -torch.sign(dy)

    @staticmethod
    def getSaftyMap(lidar_map):
        blur_map = lidar_map[:, 2, ...].unsqueeze(1)  # (B, 1, 127, 127)
        blur_map = blur_map * (blur_map > 0.2)
        blur_map = func.max_pool2d(blur_map, kernel_size=3, stride=1, padding=1)
        blur_map = func.max_pool2d(blur_map, kernel_size=3, stride=1, padding=1).squeeze(1)
        safety_map = torch.ones(blur_map.shape[0], 5, 127, 127, dtype=torch.float32, device=blur_map.device)
        safety_map[:, 1, :, :] = safety_map[:, 1, :, :] * (blur_map < 0.4)  # green means absolutely safe
        # safety_map[:, 0, :, :] = safety_map[:, 0, :, :] * (blur_map > 0.1) * (blur_map < 0.4)  # blue means not suggested
        safety_map[:, 2, :, :] = safety_map[:, 2, :, :] * (blur_map > 0.4)  # red means dangerous
        safety_map[:, 3, :, :] = lidar_map[:, 0, :, :]  # channel 3 means the target
        # safety_map[:, 4, :, :] = lidar_map[:, 1, :, :]  # channel 4 means the start

        return safety_map

    @staticmethod
    def getGridWorld(safety_map):
        # grid_world, blue: wall, green: target, red: start
        grid_world = func.max_pool2d(safety_map[:, 2], kernel_size=4, stride=4, padding=2)

        target_loc = torch.argmax(safety_map[:, 3].flatten(1), dim=1, keepdim=True)
        target_rows = target_loc // 127 // 4
        target_cols = target_loc % 127 // 4
        # compute the distance from every pixel to the target
        h_dist_map = torch.abs(torch.arange(32, dtype=torch.float32, device=grid_world.device).unsqueeze(0).repeat(32, 1) - target_cols)
        v_dist_map = torch.abs(torch.arange(32, dtype=torch.float32, device=grid_world.device).unsqueeze(1).repeat(1, 32) - target_rows)
        dist_map = torch.sqrt(h_dist_map ** 2 + v_dist_map ** 2).unsqueeze(0).repeat(grid_world.shape[0], 1, 1)
        end = 1 - dist_map / torch.max(dist_map.flatten(0), dim=0).values.view(-1, 1, 1)

        start = torch.zeros((grid_world.shape[0], 32, 32), dtype=torch.float32, device=grid_world.device)
        start[:, 16, 16] = 1

        grid_world = torch.stack([end, start, grid_world], dim=1)

        return grid_world

    @staticmethod
    def getPathAStar(obstacle_map, start_pos, end_pos, value_map):
        # run A* algorithm
        path = [start_pos]
        path_map = torch.zeros_like(obstacle_map)
        path_map[end_pos[0], end_pos[1]] = 0.2
        path_map[start_pos[0], start_pos[1]] = 1

        while path[-1] != end_pos:
            current_r, current_c = path[-1]
            best_neighbor = path[-1]
            best_value = -1
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_r = current_r + dr
                neighbor_c = current_c + dc
                if 0 <= neighbor_r < 34 and 0 <= neighbor_c < 34 and obstacle_map[neighbor_r, neighbor_c] == 0:
                    value = value_map[neighbor_r, neighbor_c]
                    if value > best_value:
                        best_value = value
                        best_neighbor = [neighbor_r, neighbor_c]
            path.append(best_neighbor)
            path_map[current_r, current_c] = 0.5
            path_map[best_neighbor[0], best_neighbor[1]] = 1
            # update value map
            value_map[current_r, current_c] -= 1 / 16

        return path

    @staticmethod
    def smoothPath(path, obstacle_map):
        # shorten path, if two non-adjacent path points are connected by a straight line,
        # replace all middle points by a straight line
        if len(path) < 3:
            return path

        i = 0
        while (i < len(path) - 2):
            # find the last connectable point from path[i]
            should_replace = False
            replace_id = -1
            longest_linkage = []
            for j in range(i + 2, len(path)):
                ri, ci = path[i]
                rj, cj = path[j]
                dr = rj - ri
                dc = cj - ci
                l1_dist = abs(dr) + abs(dc)

                linkage = [[ri, ci]]
                linkage_exist = True
                while linkage[-1] != [rj, cj]:
                    moveable = False
                    if linkage[-1][0] != rj and obstacle_map[linkage[-1][0] + np.sign(dr), linkage[-1][1]] == 0:
                        linkage.append([linkage[-1][0] + np.sign(dr), linkage[-1][1]])
                        moveable = True
                    if linkage[-1][1] != cj and obstacle_map[linkage[-1][0], linkage[-1][1] + np.sign(dc)] == 0:
                        linkage.append([linkage[-1][0], linkage[-1][1] + np.sign(dc)])
                        moveable = True
                    if not moveable:
                        linkage_exist = False
                        break

                if j - i > l1_dist and linkage_exist:
                    longest_linkage = linkage[:]
                    should_replace = True
                    replace_id = j

            if should_replace:
                # now, replace path[i+1] to path[replace_id-1] by a straight line connecting path[i] and path[replace_id]
                path = path[:i + 1] + longest_linkage + path[replace_id:]
                i = replace_id
            else:
                i += 1

        return path


    def findPathTraditional(self, grid_world):
        # grid_world: (1, 3, 32, 32)
        # channel 0: end
        # channel 1: start
        # channel 2: obstacle

        grid_world = func.pad(grid_world, (1, 1, 1, 1), mode="constant", value=0)

        # run path finding algorithm
        obstacle_map = grid_world[0, 2]  # 32x32
        value_map = grid_world[0, 0]  # 32x32

        start_pos = [17, 17]
        end_pos = [int(each) for each in torch.where(grid_world[0, 0] == 1)]

        obstacle_map[start_pos[0], start_pos[1]] = 0

        if obstacle_map[end_pos[0], end_pos[1]] == 1:
            grid_world[0, 1, 17, 17] = 1
            return [], grid_world

        path = PathModel.getPathAStar(obstacle_map, start_pos, end_pos, value_map)

        path = PathModel.smoothPath(path, obstacle_map)


        for r, c in path:
            grid_world[0, 1, r, c] = 1
        # cv2.imshow("grid_world", cv2.resize(grid_world_np, (256, 256), interpolation=cv2.INTER_NEAREST))

        return path[1:], grid_world


    def forward(self, lidar_map, spacial_features):
        B = lidar_map.shape[0]
        self.grid_world = torch.zeros((B, 3, 34, 34)).cuda()
        batch_steer = torch.zeros(B).cuda()
        batch_speed = torch.zeros(B).cuda()
        for bi in range(B):
            safety_map = self.getSaftyMap(lidar_map[bi:bi+1])
            grid_world = self.getGridWorld(safety_map)
            path, grid_world = self.findPathTraditional(grid_world)
            self.grid_world[bi] = grid_world[0]

            for i in range(min(len(path), 5)):
                target_y = torch.tensor(path[i][0] - 17, dtype=torch.float32, device="cuda")
                target_x = torch.tensor(path[i][1] - 17, dtype=torch.float32, device="cuda")
                if target_y == 0:
                    target_y -= 1

                steer, speed_direction = self.getSteer(target_x, target_y)
                batch_steer[bi] += steer
                batch_speed[bi] += speed_direction

        return torch.clip(batch_speed, -1, 1), batch_steer / B


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


