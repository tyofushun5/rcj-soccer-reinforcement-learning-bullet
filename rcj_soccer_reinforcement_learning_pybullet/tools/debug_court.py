import time

import numpy as np
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

def create():
    """アタッカーを生成
    オブジェクト設定"""

    agent_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName="../stl/robot_v2.stl",
        meshScale=[0.001, 0.001, 0.001]
    )
    #外観設定
    agent_visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName="../stl/robot_v2.stl",
        meshScale=[0.001, 0.001, 0.001],
        rgbaColor=[0.2, 0.2, 0.2, 1] #黒色
    )
    # 動的ボディとしてオブジェクトを作成
    agent_id = p.createMultiBody(
        baseMass=1.4,
        baseCollisionShapeIndex=agent_collision,
        baseVisualShapeIndex=agent_visual,
        basePosition=[0.0, 0.0, 1.0],
        baseOrientation = p.getQuaternionFromEuler([np.pi/2.0, 0.0, 0.0])
    )
    return agent_id

agent_id = create()
t = 0
while True:
    p.stepSimulation()

    # 回転速度をリセット
    p.resetBaseVelocity(
        objectUniqueId=agent_id,
        angularVelocity=[np.pi/2.0, 0.0, 0]  # 回転速度
    )
    time.sleep(0.01)
    t += 0.01