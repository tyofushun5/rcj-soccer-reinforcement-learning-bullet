import random

import cv2 as cv
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rcj_soccer_reinforcement_learning_pybullet.object.unit import Unit
from rcj_soccer_reinforcement_learning_pybullet.tools.calculation_tools import CalculationTool
from rcj_soccer_reinforcement_learning_pybullet.reward.image_reward import ImageRewardCalculation

class ImageEnvironment(gym.Env):
    def __init__(self, create_position, max_steps, magnitude, gui=False):
        super().__init__()
        # PyBulletの初期化
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF("stadium.sdf")

        self.action_space = spaces.Discrete(360)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "ball_angle": spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        })

        self.unit = Unit()
        self.cal = CalculationTool()
        self.reward_cal = ImageRewardCalculation()
        self.cp = create_position
        self.agent_random_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.unit.create_unit(self.cp, self.agent_random_pos)

        self.hit_ids = []
        self.max_steps = max_steps
        self.magnitude = magnitude
        self.step_count = 0

        self.reset()

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        self.step_count += 1

        self.unit.action(robot_id=self.unit.agent_id,
                         angle_deg=action,
                         magnitude=self.magnitude)

        for _ in range(10):
            p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(self.unit.agent_id)
        fixed_ori = p.getQuaternionFromEuler([np.pi/2.0, 0, np.pi])
        p.resetBasePositionAndOrientation(self.unit.agent_id,
                                          pos,
                                          fixed_ori)

        agent_pos, _ = p.getBasePositionAndOrientation(self.unit.agent_id)

        image_obs = self.unit.get_image()

        cv.imshow('name',image_obs)
        cv.waitKey(1)

        ball_angle = self.cal.angle_calculation_id(self.unit.agent_id,
                                                   self.unit.ball_id)

        ball_angle = np.array([round(ball_angle, 2)], dtype=np.float32)

        self.hit_ids = self.unit.detection_line()
        reward = self.reward_cal.reward_calculation(self.hit_ids,
                                                    self.unit.agent_id,
                                                    self.unit.ball_id,
                                                    self.unit.wall_id,
                                                    self.unit.blue_goal_id,
                                                    self.unit.yellow_goal_id,
                                                    self.step_count)

        observation = {
            "image": image_obs,
            "ball_angle": ball_angle
        }

        if self.reward_cal.is_goal:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        if  self.reward_cal.is_out:
            truncated = True

        if self.step_count % 10 == 0:
            print(reward,ball_angle)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF("stadium.sdf")

        if seed is not None:
            np.random.seed(seed)

        self.agent_random_pos[0] = random.uniform(0.4, 1.5) + self.cp[0]
        self.agent_random_pos[1] = random.uniform(0.4, 1.5) + self.cp[1]

        self.unit = Unit()
        self.unit.create_unit(self.cp, self.agent_random_pos)
        image_obs = self.unit.get_image()

        self.step_count = 0

        ball_angle = self.cal.angle_calculation_id(self.unit.agent_id, self.unit.ball_id)
        ball_angle = np.array([round(ball_angle, 2)], dtype=np.float32)

        initial_obs = {
            "image": image_obs,
            "ball_angle": ball_angle
        }

        info = {}
        return initial_obs, info

    def render(self):
        pass

    def close(self):
        p.disconnect()
