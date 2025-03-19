import math

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
        vector_x = magnitude * math.cos(angle_rad)
        vector_y = magnitude * math.sin(angle_rad)
        return vector_x, vector_y

    @staticmethod
    def movement_reward_calculation(reward, pos, previous_pos, past_distance):
        distance = CalculationTool.euclidean_distance_pos(pos, previous_pos)
        if distance < past_distance:
            reward += 0.5
        else:
            reward -= 0.5
        return reward

    @staticmethod
    def distance_reward_calculation(reward, agent_pos, ball_pos, ball_past_distance):
        distance = CalculationTool.euclidean_distance_pos(agent_pos, ball_pos)
        if distance < ball_past_distance:
            reward += 0.6
        else:
            reward -= 0.6
        return reward
