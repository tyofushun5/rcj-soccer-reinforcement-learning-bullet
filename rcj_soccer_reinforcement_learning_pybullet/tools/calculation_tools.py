import math

import numpy as np
import pybullet as p

class CalculationTool(object):
    def __init__(self):
        pass

    @staticmethod
    def angle_calculation_id(a_id, b_id):
        a_pos, _ = p.getBasePositionAndOrientation(a_id)
        b_pos, _ = p.getBasePositionAndOrientation(b_id)
        angle_deg = CalculationTool.angle_calculation_pos(a_pos, b_pos)
        return angle_deg

    @staticmethod
    def normalization(value):
        normalized = (value / 180) - 1
        return normalized

    @staticmethod
    def denormalization(value):
        denormalized = (value + 1) * 180
        return denormalized

    @staticmethod
    def angle_calculation_pos(a_pos, b_pos):
        dx = b_pos[0] - a_pos[0]
        dy = b_pos[1] - a_pos[1]
        angle_radians = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_radians)
        angle_deg = angle_deg % 360
        angle_deg = round(angle_deg, 1)
        return angle_deg

    @staticmethod
    def euclidean_distance_id(a_id, b_id):
        a_pos, _ = p.getBasePositionAndOrientation(a_id)
        b_pos, _ = p.getBasePositionAndOrientation(b_id)
        if len(a_pos) != len(b_pos):
            raise ValueError
        x1, y1 = a_pos[:2]
        x2, y2 = b_pos[:2]
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def euclidean_distance_pos(a_pos, b_pos):
        if len(a_pos) != len(b_pos):
            raise ValueError
        x1, y1 = a_pos[:2]
        x2, y2 = b_pos[:2]
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def vector_calculations(angle_deg, magnitude):
        angle_rad = math.radians(angle_deg)
        vector_x = magnitude * math.sin(angle_rad)
        vector_y = magnitude * math.cos(angle_rad)
        return vector_x, vector_y

    @staticmethod
    def angular_velocity_calculation(number=0):
        angular_velocity = 0.0
        match number:
            case -1.0:
                angular_velocity=0.0
            case 0.0:
                angular_velocity=0.5
            case 1.0:
                angular_velocity=-0.5
        return angular_velocity

    @staticmethod
    def angular_vector_calculation(number=0):
        angular_vector = number * 0.5
        return angular_vector

    @staticmethod
    def movement_reward_calculation(pos,
                                    previous_pos,
                                    past_distance,
                                    fine=0.0,
                                    penalty=0.0
                                    ):
        distance = CalculationTool.euclidean_distance_pos(pos, previous_pos)
        if distance < past_distance:
            result = fine
        else:
            result = penalty
        return result

    @staticmethod
    def distance_reward_calculation(agent_pos,
                                    ball_pos,
                                    ball_past_distance,
                                    fine=0.0,
                                    penalty=0.0
                                    ):
        distance = CalculationTool.euclidean_distance_pos(agent_pos, ball_pos)
        if distance < ball_past_distance:
            reward = fine
        else:
            reward = penalty
        return reward

