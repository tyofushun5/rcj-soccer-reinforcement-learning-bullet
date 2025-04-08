import pybullet as p

from rcj_soccer_reinforcement_learning_pybullet.tools.calculation_tools import CalculationTool
from rcj_soccer_reinforcement_learning_pybullet.object.court import Court

class RewardFunction(Court):
    def __init__(self):
        super().__init__()
        self.previous_attacker_pos = [0, 0, 0]
        self.past_distance = 0
        self.ball_past_distance = 0
        self.is_goal = False
        self.is_out = False
        self.is_touch = False
        self.is_online = False
        self.cal = CalculationTool()

    def reward_calculation(self,
                           hit_ids,
                           agent_id,
                           ball_id,
                           wall_id,
                           blue_goal_id,
                           yellow_goal_id,
                           step_count,
                           max_steps,
        ):
        reward = 0
        self.is_goal = False
        self.is_out = False
        self.is_online = False

        # reward_function -= step_count*0.001

        agent_pos, _ = p.getBasePositionAndOrientation(agent_id)

        reward += self.cal.movement_reward_calculation(agent_pos,
                                                       self.previous_attacker_pos,
                                                       self.past_distance,
                                                       fine=0.2,
                                                       penalty=-0.1)
        self.previous_attacker_pos = agent_pos

        ball_pos, _ = p.getBasePositionAndOrientation(ball_id)

        reward += self.cal.distance_reward_calculation(agent_pos,
                                                       ball_pos,
                                                       self.ball_past_distance,
                                                       fine=0.2,
                                                       penalty=-0.1)

        self.ball_past_distance = self.cal.euclidean_distance_pos(agent_pos,
                                                                  ball_pos)

        if p.getContactPoints(ball_id, agent_id):
            reward += 3.0
            self.is_touch = True
        else:
            reward -= 0.1
            self.is_touch = False
        if hit_ids[self.my_goal_line_idx] == ball_id:
            reward -= 100.0
            self.is_out = True
        if hit_ids[self.enemy_goal_line_idx] == ball_id:
            reward += 100.0
            self.is_goal = True
        if hit_ids[self.my_goal_line_idx] == agent_id:
            reward -= 100.0
            self.is_out = True
        if hit_ids[self.enemy_goal_line_idx] == agent_id:
            reward -= 100.0
            self.is_out = True
        if p.getContactPoints(wall_id, agent_id):
            reward -= 100.0
            self.is_out = True
        if step_count >= max_steps:
            reward -= 100.0
            self.is_out = True
        if p.getContactPoints(blue_goal_id, agent_id):
            reward -= 0.5
        if p.getContactPoints(yellow_goal_id, agent_id):
            reward -= 0.5
        if p.getContactPoints(wall_id, ball_id):
            reward -= 0.2
        for i in range(len(hit_ids)):
            if hit_ids[i] == agent_id:
                reward -= 0.3
                self.is_online = True

        angle = self.cal.angle_calculation_id(agent_id, ball_id)

        if angle<=90 or angle>=270:
            reward += 0.3
        else:
            reward -= 0.2
        if angle<=45 or angle>=315:
            reward += 0.4
        else:
            reward -= 0.2
        return reward
if __name__ == '__main__':
    import doctest
    doctest.testmod()