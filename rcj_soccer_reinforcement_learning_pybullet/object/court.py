import abc
import logging
import random
import time
import os

import numpy as np
import pybullet as p
import pybullet_data


script_dir = os.path.dirname(os.path.abspath(__file__))

stl_dir = os.path.join(script_dir, 'stl')

goal_path = os.path.join(stl_dir, 'goal.stl')
wall_path = os.path.join(stl_dir, 'wall.stl')
line_path = os.path.join(stl_dir, 'line.stl')


class Court(object):

    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [4.0, 0.0, 0.0]
        self.cp = create_position
        self.num_lines = 8
        self.line_ids = []
        self.line_default_color = [0.0, 1.0, 0.0]
        self.line_hit_color = [1.0, 0.0, 0.0]
        self.line_height = 0.01
        self.line_from = [[0.12, 0.12, self.line_height],
                         [1.76, 0.12, self.line_height],
                         [0.12, 0.12, self.line_height],
                         [1.24, 0.12, self.line_height],
                         [0.12, 2.35, self.line_height],
                         [1.24, 2.35, self.line_height],
                         [0.64, 0.13, self.line_height],
                         [0.64, 2.34, self.line_height]]
        self.line_to = [[0.12, 2.35, self.line_height],
                       [1.76, 2.35, self.line_height],
                       [0.62, 0.12, self.line_height],
                       [1.76, 0.12, self.line_height],
                       [0.62, 2.35, self.line_height],
                       [1.76, 2.35, self.line_height],
                       [1.22, 0.13, self.line_height],
                       [1.22, 2.34, self.line_height]]
        self.from_positions = []
        self.to_positions = []
        self.__my_goal_line_idx = 6
        self.__enemy_goal_line_idx = 7
        self.hit_ids = []
        self.my_goal_position = [0.93+self.cp[0], 0.11+self.cp[1], 0.05]
        self.enemy_goal_position = [0.93+self.cp[0], 2.32+self.cp[1], 0.05]
        self.ball_start_position = [0.915+self.cp[0], 1.8+self.cp[1], 0.1+self.cp[2]]
        self.ball_position = self.ball_start_position

    @property
    def my_goal_line_idx(self):
        return self.__my_goal_line_idx

    @property
    def enemy_goal_line_idx(self):
        return self.__enemy_goal_line_idx

    def create_court(self):

        wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=wall_path,
            meshScale=[0.001, 0.001, 0.001],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )

        wall_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=wall_path,
            meshScale=[0.001, 0.001, 0.001],
            rgbaColor=[0.22, 0.22, 0.22, 1]
        )

        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=self.cp,
            baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0.0, -np.pi/2.0])
        )

        goal_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=goal_path,
            meshScale=[0.001, 0.001, 0.001],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )

        blue_goal_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=goal_path,
            meshScale=[0.001, 0.001, 0.001],
            rgbaColor=[0.2, 0.3, 0.8, 1]
        )

        yellow_goal_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=goal_path,
            meshScale=[0.001, 0.001, 0.001],
            rgbaColor=[0.8, 0.7, 0.2, 1]
        )

        blue_goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=goal_collision,
            baseVisualShapeIndex=blue_goal_visual,
            basePosition=[self.cp[0]+0.62, self.cp[1]+0.02, self.cp[2]+0.0],
            baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0.0, -np.pi/2.0])
        )

        yellow_goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=goal_collision,
            baseVisualShapeIndex=yellow_goal_visual,
            basePosition=[self.cp[0]+1.24, self.cp[1]+2.45, self.cp[2]+0.0],
            baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0.0, np.pi/2.0])
        )

        line_collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=1e-10
        )

        line_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=line_path,
            meshScale=[0.001, 0.001, 0.001],
            rgbaColor=[1.0, 1.0, 1.0, 1]
        )

        line_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=line_collision,
            baseVisualShapeIndex=line_visual,
            basePosition=[self.cp[0]+0.14, self.cp[1]+0.14, self.cp[2]+0.0],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        )

        return wall_id, blue_goal_id, yellow_goal_id

    def create_debug_line(self):
        """コートのデバックラインを作成"""
        for i in range(self.num_lines):
            self.line_ids.append(p.addUserDebugLine(
                [self.line_from[i][0] + self.cp[0],
                 self.line_from[i][1] + self.cp[1],
                 self.line_from[i][2] + self.cp[2]],
                [self.line_to[i][0] + self.cp[0],
                 self.line_to[i][1] + self.cp[1],
                 self.line_to[i][2] + self.cp[2]],
                lineColorRGB=self.line_default_color,
                lineWidth=2.0,
                lifeTime=0
            ))

    def detection_debug_line(self):
        """ラインの当たり判定を検出し、ヒット時はラインを上書き更新"""
        self.from_positions = []
        self.to_positions = []
        self.hit_ids = []

        for i in range(self.num_lines):
            self.from_positions.append([
                self.line_from[i][0] + self.cp[0],
                self.line_from[i][1] + self.cp[1],
                self.line_from[i][2] + self.cp[2]
            ])
            self.to_positions.append([
                self.line_to[i][0] + self.cp[0],
                self.line_to[i][1] + self.cp[1],
                self.line_to[i][2] + self.cp[2]
            ])

        results = p.rayTestBatch(self.from_positions, self.to_positions)

        for i in range(self.num_lines):
            hit_id = results[i][0]
            self.hit_ids.append(hit_id)
            # ここで、replaceItemUniqueIdを使用して同じラインIDを書き換える
            if hit_id == -1:
                # ヒットがない場合は元の色で上書き
                p.addUserDebugLine(
                    self.from_positions[i],
                    self.to_positions[i],
                    lineColorRGB=self.line_default_color,
                    lineWidth=1.0,
                    lifeTime=0,
                    replaceItemUniqueId=self.line_ids[i]
                )
            else:
                # ヒットがある場合はヒット色で上書き
                p.addUserDebugLine(
                    self.from_positions[i],
                    self.to_positions[i],
                    lineColorRGB=self.line_hit_color,
                    lineWidth=1.0,
                    lifeTime=0,
                    replaceItemUniqueId=self.line_ids[i]
                )

        return self.hit_ids

    def create_ball(self, position=None):

        self.ball_position = position

        if position is None:
            self.ball_position = self.ball_start_position

        collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE,
                                                    radius=0.037)

        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE,
                                              radius=0.037,
                                              rgbaColor=[0.15, 0.15, 0.15, 0.95])

        ball_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.ball_position
        )
        return ball_id

if __name__ == '__main__':
    import doctest
    doctest.testmod()