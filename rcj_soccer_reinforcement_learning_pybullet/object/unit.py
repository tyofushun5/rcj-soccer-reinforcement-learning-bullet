import pybullet as p

from rcj_soccer_reinforcement_learning_pybullet.object.court import Court
from rcj_soccer_reinforcement_learning_pybullet.object.robot import Agent


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

    def create_unit(self, create_position, agent_pos):
        self.cp = create_position
        self.court = Court(self.cp)
        self.agent = Agent(self.cp)
        self.wall_id, self.blue_goal_id, self.yellow_goal_id = self.court.create_court()
        self.court.create_court_line()
        self.ball_id = self.court.create_ball()
        self.agent_id = self.agent.create(agent_pos)
        self.previous_agent_pos = self.agent.start_pos
        Unit.set_camera(self.cp)

    def detection_line(self):
        self.hit_ids = self.court.detection_line()
        return self.hit_ids

    def action(self, robot_id, angle_deg, magnitude):
        self.agent.action(robot_id, angle_deg, magnitude)

    def get_image(self):
        image = self.agent.get_camera_image(self.agent_id)

        return image

    @staticmethod
    def set_camera(position):
        p.resetDebugVisualizerCamera(cameraDistance=1.70,
                                     cameraYaw=90,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[position[0]+0.91, position[1]+1.25, position[2]])


