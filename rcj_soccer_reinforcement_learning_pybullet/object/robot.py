import abc
import os
import math

import pybullet as p
import numpy as np

from rcj_soccer_reinforcement_learning_pybullet.tools.calculation_tools import CalculationTool


script_dir = os.path.dirname(os.path.abspath(__file__))

stl_dir = os.path.join(script_dir, 'stl')

robot_collision_path = os.path.join(stl_dir, 'robot_v2_collision.stl')
robot_visual_path = os.path.join(stl_dir, 'robot_v2_visual.stl')


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
        self.default_ori = [0.0, 0.0, 0.0]
        self.position = self.start_pos
        self.radius = 0.11
        self.height = 0.11
        self.mass = 1.4

    def create(self, position=None):
         """アタッカーを生成
         オブジェクト設定"""
         self.position = position

         if position is None:
             self.position = self.start_pos


         agent_collision = p.createCollisionShape(
             shapeType=p.GEOM_MESH,
             fileName=robot_collision_path,
             meshScale=[0.0001, 0.0001, 0.0001]
         )

         # agent_collision = p.createCollisionShape(
         #     shapeType=p.GEOM_CYLINDER,
         #     radius=self.radius,
         #     height=self.height
         # )

         # visual_shift = [0, 0, -self.height / 2]

         agent_visual = p.createVisualShape(
             shapeType=p.GEOM_MESH,
             fileName=robot_visual_path,
             meshScale=[0.0001, 0.0001, 0.0001],
             rgbaColor=[0.2, 0.2, 0.2, 1],
             # visualFramePosition=visual_shift
         )

         agent_id = p.createMultiBody(
             baseMass=self.mass,
             baseCollisionShapeIndex=agent_collision,
             baseVisualShapeIndex=agent_visual,
             basePosition=self.position,
             baseOrientation = p.getQuaternionFromEuler(self.default_ori),
         )

         return agent_id


    def action(self, agent_id, angle_deg=0, rotate=0, magnitude=21.0):
        """ロボットを動かすメソッド"""

        dynamics_info = p.getDynamicsInfo(agent_id, -1)
        center_of_mass = dynamics_info[3]  # 重心


        p.changeDynamics(
            bodyUniqueId=agent_id,
            linkIndex=-1,
            lateralFriction=0.32,  # 摩擦係数
            spinningFriction=0.01,  # 回転摩擦
            rollingFriction=0.10,  # 転がり摩擦
            angularDamping=0.5  # 回転の減衰
        )

        x, y = self.cal.vector_calculations(angle_deg=angle_deg, magnitude=magnitude)
        angular_vector = self.cal.angular_vector_calculation(rotate)

        p.applyExternalForce(
            objectUniqueId=agent_id,
            linkIndex=-1,
            forceObj=[x, y, 0.0],
            posObj=center_of_mass,
            flags=p.LINK_FRAME
        )

        p.applyExternalTorque(
            objectUniqueId=agent_id,
            linkIndex=-1,
            torqueObj=[0.0, 0.0, angular_vector],
            flags=p.LINK_FRAME
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
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.reshape(rgb, (height, width, 4))
        rgb_image = rgb_array[:, :, :3].astype(np.uint8)
        return rgb_image


class AlgorithmRobot(Robot):
    def __init__(self, create_position, mode='ally'):
        super().__init__()
        self.cp = create_position
        self.start_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.default_ori = [0.0, 0.0, 0.0]
        self.position = self.start_pos
        self.radius = 0.11
        self.height = 0.11
        self.mass = 1.4

        if mode == 'enemy':
            self.default_ori = [0.0, 0.0, np.pi]
            self.start_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]

    def create(self, position=None):

        self.position = position

        if position is None:
            self.position = self.start_pos


        robot_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=robot_collision_path,
            meshScale=[0.0001, 0.0001, 0.0001]
        )

        # agent_collision = p.createCollisionShape(
        #     shapeType=p.GEOM_CYLINDER,
        #     radius=self.radius,
        #     height=self.height
        # )

        # visual_shift = [0, 0, -self.height / 2]

        robot_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=robot_visual_path,
            meshScale=[0.0001, 0.0001, 0.0001],
            rgbaColor=[0.2, 0.2, 0.2, 1],
            # visualFramePosition=visual_shift
        )

        robot_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=robot_collision,
            baseVisualShapeIndex=robot_visual,
            basePosition=self.position,
            baseOrientation = p.getQuaternionFromEuler(self.default_ori),
        )

        return robot_id


    def action(self, agent_id, magnitude=21.0):
        """ロボットを動かすメソッド"""

        dynamics_info = p.getDynamicsInfo(agent_id, -1)
        center_of_mass = dynamics_info[3]  # 重心


        p.changeDynamics(
            bodyUniqueId=agent_id,
            linkIndex=-1,
            lateralFriction=0.32,  # 摩擦係数
            spinningFriction=0.01,  # 回転摩擦
            rollingFriction=0.10,  # 転がり摩擦
            angularDamping=0.5  # 回転の減衰
        )

        p.applyExternalForce(
            objectUniqueId=agent_id,
            linkIndex=-1,
            forceObj=[0.0, 0.0, 0.0],
            posObj=center_of_mass,
            flags=p.LINK_FRAME
        )

        p.applyExternalTorque(
            objectUniqueId=agent_id,
            linkIndex=-1,
            torqueObj=[0.0, 0.0, 0.0],
            flags=p.LINK_FRAME
        )

if __name__ == '__main__':
    import doctest
    doctest.testmod()

