import time
import os

import numpy as np
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)

def create_court():
    # STL ファイルの絶対パスを使用
    stl_file_path = os.path.join(os.path.dirname(__file__), 'stl_data', 'wall.stl')

    if not os.path.isfile(stl_file_path):
        raise FileNotFoundError(f"STL ファイルが見つかりません: {stl_file_path}")

    collision_court = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=[0.001, 0.001, 0.001],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    visual_court = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=[0.001, 0.001, 0.001],
        rgbaColor=[1, 1, 1, 1]
    )

    court_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_court,
        baseVisualShapeIndex=visual_court,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0])
    )
    return court_id

create_court()

t = 0
while True:
    p.stepSimulation()
    time.sleep(0.01)
    t += 0.01