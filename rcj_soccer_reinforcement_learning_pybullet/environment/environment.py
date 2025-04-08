import random
import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rcj_soccer_reinforcement_learning_pybullet.object.unit import Unit
from rcj_soccer_reinforcement_learning_pybullet.tools.calculation_tools import CalculationTool
from rcj_soccer_reinforcement_learning_pybullet.reward_function.reward_function import RewardFunction

class Environment(gym.Env):
    def __init__(self, create_position, max_steps, magnitude, gui=False):
        super().__init__()
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=50,
                                     cameraPitch=-35,
                                     cameraTargetPosition=[0, 0, 0])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF('stadium.sdf')

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(4,),
                                            dtype=np.float32)

        self.unit = Unit()
        self.cal = CalculationTool()
        self.reward_cal = RewardFunction()
        self.cp = create_position
        self.ball_random_pos = [0.915+self.cp[0], 1.8+self.cp[1], 0.1+self.cp[2]]
        self.agent_random_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.unit.create_unit(self.cp, self.agent_random_pos, self.ball_random_pos)

        self.hit_ids = []
        self.max_steps = max_steps
        self.magnitude = magnitude
        self.step_count = 0

        self.is_online_obs = -1

        self.reset()

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        self.step_count += 1

        self.unit.action(robot_id=self.unit.agent_id,
                         angle=action[0],
                         rotate=action[1],
                         magnitude=self.magnitude)

        for i in range(10):
            p.stepSimulation()

        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.unit.agent_id)
        euler = p.getEulerFromQuaternion(agent_ori)
        yaw_deg = math.degrees(euler[2])
        yaw_deg = (yaw_deg + 360) % 360
        yaw_deg_from_y_axis = ((90-yaw_deg)-90) % 360
        yaw_deg_from_y_axis = round(yaw_deg_from_y_axis, 2)

        ball_angle = self.cal.angle_calculation_id(self.unit.agent_id,
                                                   self.unit.ball_id)

        enemy_goal_angle = self.cal.angle_calculation_pos(agent_pos,
                                                          self.unit.court.enemy_goal_position)

        my_goal_angle = self.cal.angle_calculation_pos(agent_pos,
                                                       self.unit.court.my_goal_position)

        ball_angle = ball_angle - yaw_deg_from_y_axis
        my_goal_angle = my_goal_angle - yaw_deg_from_y_axis
        enemy_goal_angle = enemy_goal_angle - yaw_deg_from_y_axis

        if ball_angle < 0:
            ball_angle = ball_angle + 360

        if my_goal_angle < 0:
            my_goal_angle = my_goal_angle + 360

        if enemy_goal_angle < 0:
            enemy_goal_angle = enemy_goal_angle + 360


        ball_angle = round(ball_angle, 2)
        my_goal_angle = round(my_goal_angle, 2)
        enemy_goal_angle = round(enemy_goal_angle, 2)

        self.hit_ids = self.unit.detection_line()
        reward = self.reward_cal.reward_calculation(self.hit_ids,
                                                    self.unit.agent_id,
                                                    self.unit.ball_id,
                                                    self.unit.wall_id,
                                                    self.unit.blue_goal_id,
                                                    self.unit.yellow_goal_id,
                                                    self.step_count,
                                                    self.max_steps)
        if self.reward_cal.is_online:
            self.is_online_obs = 1.0
        else:
            self.is_online_obs = -1.0

        normalized_ball_angle = self.cal.normalization(ball_angle)
        normalized_enemy_goal_angle = self.cal.normalization(enemy_goal_angle)
        normalized_my_goal_angle = self.cal.normalization(my_goal_angle)

        observation = np.array([normalized_ball_angle,
                                normalized_enemy_goal_angle,
                                normalized_my_goal_angle,
                                self.is_online_obs],
                                dtype=np.float32)

        if self.reward_cal.is_goal:
            terminated = True

        if  self.reward_cal.is_out:
            truncated = True

        #print(my_goal_angle, enemy_goal_angle, reward_function)
        #print("ball",observation)
        #print(reward_function)
        #print("action",action)
        #print(ball_angle,yaw_deg_from_y_axis)
        #print(enemy_goal_angle)
        #print(self.is_online_obs)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF('stadium.sdf')

        if seed is not None:
            np.random.seed(seed)

        min_distance = 0.18

        while True:
            self.agent_random_pos[0] = random.uniform(0.4, 1.5) + self.cp[0]
            self.agent_random_pos[1] = random.uniform(0.4, 1.8) + self.cp[1]

            self.ball_random_pos[0] = random.uniform(0.4, 1.5) + self.cp[0]
            self.ball_random_pos[1] = random.uniform(0.4, 1.8) + self.cp[1]

            dx = self.agent_random_pos[0] - self.ball_random_pos[0]
            dy = self.agent_random_pos[1] - self.ball_random_pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            if distance >= min_distance:
                break

        self.unit = Unit()
        self.unit.create_unit(self.cp, self.agent_random_pos, self.ball_random_pos)

        self.step_count = 0
        initial_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        info = {}
        return initial_obs, info

    def render(self):
        pass

    def close(self):
        p.disconnect()

if __name__ == '__main__':
    import doctest
    doctest.testmod()