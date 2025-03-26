import abc
import os
import math

import cv2 as cv
import pybullet as p
import numpy as np

from rcj_soccer_reinforcement_learning_pybullet.tools.calculation_tools import CalculationTool


script_dir = os.path.dirname(os.path.abspath(__file__))

stl_dir = os.path.join(script_dir, "stl")

robot_collision_path = os.path.join(stl_dir, "robot_v2_collision.stl")
robot_visual_path = os.path.join(stl_dir, "robot_v2_visual.stl")


class Robot(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.cal = CalculationTool()

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def action(self, robot_id, angle_deg, magnitude):
        pass



class Agent(Robot):
    def __init__(self, create_position):
        super().__init__()
        self.cp = create_position
        self.start_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.position = self.start_pos

    def create(self, position=None):
         """アタッカーを生成
         オブジェクト設定"""
         self.position = position

         if position is None:
             self.position = self.start_pos

         agent_collision = p.createCollisionShape(
             shapeType=p.GEOM_MESH,
             fileName=robot_collision_path,
             meshScale=[0.001, 0.001, 0.001]
         )

         #外観設定
         agent_visual = p.createVisualShape(
             shapeType=p.GEOM_MESH,
             fileName=robot_visual_path,
             meshScale=[0.001, 0.001, 0.001],
             rgbaColor=[0.2, 0.2, 0.2, 1] #黒色
         )
         # 動的ボディとしてオブジェクトを作成
         agent_id = p.createMultiBody(
             baseMass=1.4,
             baseCollisionShapeIndex=agent_collision,
             baseVisualShapeIndex=agent_visual,
             basePosition=self.position,
             baseOrientation = p.getQuaternionFromEuler([np.pi/2.0, 0.0, np.pi])
         )

         return agent_id


    def action(self, agent_id, angle_deg=0, magnitude=7.0):
        """ロボットを動かすメソッド"""

        # Dynamics情報を取得
        dynamics_info = p.getDynamicsInfo(agent_id, -1)
        center_of_mass = dynamics_info[3]  # 重心の位置

        # 摩擦係数を調整
        p.changeDynamics(
            bodyUniqueId=agent_id,
            linkIndex=-1,
            lateralFriction=0.5,  # 摩擦係数
            spinningFriction=0.1,  # 回転摩擦
            rollingFriction=0.1,  # 転がり摩擦
            angularDamping=1.0  # 回転の減衰
        )

        x, y = self.cal.vector_calculations(angle_deg=angle_deg, magnitude=magnitude)

        #中心に力を加える
        p.applyExternalForce(
            objectUniqueId=agent_id,
            linkIndex=-1,
            forceObj=[x, y, 0],
            posObj=center_of_mass,
            flags=p.WORLD_FRAME
        )

        # 回転速度をリセット
        p.resetBaseVelocity(
            objectUniqueId=agent_id,
            angularVelocity=[0.0, 0.0, 0.0]  # 回転速度
        )

    @staticmethod
    def get_camera_image(robot_id, width=84, height=84):

        position, orientation = p.getBasePositionAndOrientation(robot_id)
        euler_orientation = p.getEulerFromQuaternion(orientation)

        yaw = euler_orientation[2] - math.pi/2
        pitch = -math.radians(30)

        forward = [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch)
        ]

        up = [
            -math.sin(pitch) * math.cos(yaw),
            -math.sin(pitch) * math.sin(yaw),
            math.cos(pitch)
        ]

        camera_position = [
            position[0],
            position[1]+0.024402,
            position[2]+0.162174
        ]

        cam_target = [
            camera_position[0]+forward[0],
            camera_position[1]+forward[1],
            camera_position[2]+forward[2]]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=cam_target,
            cameraUpVector=up
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=102,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )

        _, _, rgb, _, _ = p.getCameraImage(width,
                                           height,
                                           view_matrix,
                                           projection_matrix,
                                           renderer=p.ER_TINY_RENDERER)

        rgb_array = np.reshape(rgb, (height, width, 4))
        rgb_image = rgb_array[:, :, :3].astype(np.uint8)
        return rgb_image

if __name__ == '__main__':
    import doctest
    doctest.testmod()

