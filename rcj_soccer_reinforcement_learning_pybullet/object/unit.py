import pybullet as p

from rcj_soccer_reinforcement_learning_pybullet.object.court import Court
from rcj_soccer_reinforcement_learning_pybullet.object.robot import Agent
from rcj_soccer_reinforcement_learning_pybullet.object.robot import AlgorithmRobot
from rcj_soccer_reinforcement_learning_pybullet.tools.calculation_tools import CalculationTool

class Unit(object):

    def __init__(self):
        self.cp = None
        self.agent = None
        self.agent_id = None
        self.previous_agent_pos = [0, 0, 0]
        self.wall_id = None
        self.blue_goal_id = None
        self.yellow_goal_id = None
        self.ball_id = None
        self.court = None
        self.hit_ids = []
        self.past_distance = 0
        self.ball_past_distance = 0

        self.cal = CalculationTool()

    def create_unit(self, create_position, agent_pos, ball_pos):
        self.cp = create_position
        self.court = Court(self.cp)
        self.agent = Agent(self.cp)
        self.wall_id, self.blue_goal_id, self.yellow_goal_id = self.court.create_court()
        self.court.create_debug_line()
        self.ball_id = self.court.create_ball(ball_pos)
        self.agent_id = self.agent.create(agent_pos)
        self.previous_agent_pos = self.agent.start_pos
        Unit.set_camera(self.cp)

    def detection_line(self):
        self.hit_ids = self.court.detection_debug_line()
        return self.hit_ids

    def action(self, robot_id, angle=0, rotate=0, magnitude=21.0):
        angle_deg = self.cal.denormalization(angle)
        # print("angle_deg",angle_deg)
        self.agent.action(robot_id, angle_deg, rotate, magnitude)

    def get_image(self):
        image = self.agent.get_camera_image(self.agent_id)
        return image

    @staticmethod
    def set_camera(position):
        p.resetDebugVisualizerCamera(cameraDistance=1.70,
                                     cameraYaw=90.0,
                                     cameraPitch=-45.0,
                                     cameraTargetPosition=[position[0]+0.91, position[1]+1.25, position[2]])

if __name__ == '__main__':
    import doctest
    doctest.testmod()
