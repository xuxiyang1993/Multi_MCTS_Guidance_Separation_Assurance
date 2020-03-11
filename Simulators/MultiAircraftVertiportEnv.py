import math
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
from collections import OrderedDict

from config_vertiport import Config

__author__ = "Xuxi Yang <xuxiyang@iastate.edu>"


class MultiAircraftEnv(gym.Env):
    """
    This is the airspace simulator where we can control multiple aircraft to their respective
    goal position while avoiding conflicts between each other. The aircraft will takeoff from
    different vertiports, and select a random vertiport as its destination.
    **STATE:**
    The state consists all the information needed for the aircraft to choose an optimal action:
    position, velocity, speed, heading, goal position, of each aircraft:
    (x, y, v_x, v_y, speed, \psi, goal_x, goal_y) for each aircraft.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 for the change of heading angle of each aircraft.
    More specifically, the action is a dictionary in form {id: action, id: action, ...}
    """

    def __init__(self, sd=2, debug=False):
        self.load_config()  # load parameters for the simulator
        self.load_vertiport()  # load config for the vertiports
        self.state = None
        self.viewer = None

        # build observation space and action space
        self.observation_space = self.build_observation_space()  # observation space deprecated, not in use for MCTS
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32)  # position range is the length and width of airspace
        self.action_space = spaces.Tuple((spaces.Discrete(3),) * self.num_aircraft)
        # action space deprecated, since number of aircraft is changing from time to time

        self.time_step = 0
        self.total_timesteps = 0

        self.conflicts = 0
        self.seed(sd)

        self.debug = debug

        self.route_time = {0: {1: [], 2: [], 3: []},
                           1: {1: [], 2: [], 3: []}}
        self.route_conflicts = {0: {1: 0, 2: 0, 3: 0},
                                1: {1: 0, 2: 0, 3: 0}}
        self.route_NMACs = {0: {1: 0, 2: 0, 3: 0},
                            1: {1: 0, 2: 0, 3: 0}}

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height  # dimension of the airspace
        self.num_aircraft = Config.num_aircraft
        self.EPISODES = Config.EPISODES
        # self.tick = Config.tick
        self.scale = Config.scale  # 1 meter = ? pixels, set to 60 here
        self.minimum_separation = Config.minimum_separation
        self.NMAC_dist = Config.NMAC_dist
        # self.horizon_dist = Config.horizon_dist
        self.initial_min_dist = Config.initial_min_dist  # when aircraft generated, is shouldn't be too close to others
        self.goal_radius = Config.goal_radius
        self.init_speed = Config.init_speed
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed

    def load_vertiport(self):
        self.vertiport_list = []
        # read the vertiport location from config file
        for i in range(Config.vertiport_loc.shape[0]):
            self.vertiport_list.append(VertiPort(id=i, position=Config.vertiport_loc[i]))

    def reset(self):
        # aircraft is stored in this dict
        self.aircraft_dict = AircraftDict()
        self.id_tracker = 0  # assign id to newly generated aircraft, increase by one after generating aircraft.

        # keep track of number of conflicts, goals, and NMACs.
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0

        return self._get_ob()

    # deprecated
    def pressure_reset(self):
        # aircraft is stored in this dict
        self.aircraft_dict = AircraftDict()
        self.id_tracker = 0
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0

        for id in range(self.num_aircraft):
            theta = 2 * id * math.pi / self.num_aircraft
            r = self.window_width / 2 - 10
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            position = (self.window_width / 2 + x, self.window_height / 2 + y)
            goal_pos = (self.window_width / 2 - x, self.window_height / 2 - y)

            aircraft = Aircraft(
                id=id,
                position=position,
                speed=self.init_speed,
                heading=theta + math.pi,
                goal_pos=goal_pos
            )

            self.aircraft_dict.add(aircraft)

        return self._get_ob()

    def pressure_reset1(self):
        # aircraft is stored in this dict
        self.aircraft_dict = AircraftDict()
        self.id_tracker = 0
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0

        self.conflict_flag = [False] * self.num_aircraft
        self.NMAC_flag = [False] * self.num_aircraft

        for id in range(self.num_aircraft):
            theta = np.random.uniform(0, 2 * np.pi)
            r2 = np.random.uniform((10000 / 60) ** 2, (15000 / 60) ** 2)
            x = math.sqrt(r2) * np.cos(theta)
            y = math.sqrt(r2) * np.sin(theta)
            position = (self.window_width / 2 + x, self.window_height / 2 + y)
            goal_pos = (self.window_width / 2 - x, self.window_height / 2 - y)

            aircraft = Aircraft(
                id=id,
                position=position,
                speed=self.init_speed,
                heading=theta + math.pi,
                goal_pos=goal_pos
            )

            if id > 0:
                while np.min(self.dist_to_all_aircraft(aircraft)[0]) < 2 * Config.minimum_separation:
                    theta = np.random.uniform(0, 2 * np.pi)
                    r2 = np.random.uniform((10000 / 60) ** 2, (15000 / 60) ** 2)
                    x = math.sqrt(r2) * np.cos(theta)
                    y = math.sqrt(r2) * np.sin(theta)
                    position = (self.window_width / 2 + x, self.window_height / 2 + y)
                    goal_pos = (self.window_width / 2 - x, self.window_height / 2 - y)

                    aircraft = Aircraft(
                        id=id,
                        position=position,
                        speed=self.init_speed,
                        heading=theta + math.pi,
                        goal_pos=goal_pos
                    )

            self.aircraft_dict.add(aircraft)

        return self._get_ob()

    def _get_ob(self):
        s = []
        id = []
        # loop all the aircraft
        # return the information of each aircraft and their respective id
        # s is in shape [number_aircraft, 8], id is list of length number_aircraft
        for key, aircraft in self.aircraft_dict.ac_dict.items():
            # (x, y, vx, vy, speed, heading, gx, gy)
            s.append(aircraft.position[0])
            s.append(aircraft.position[1])
            s.append(aircraft.velocity[0])
            s.append(aircraft.velocity[1])
            s.append(aircraft.speed)
            s.append(aircraft.heading)
            s.append(aircraft.goal.position[0])
            s.append(aircraft.goal.position[1])

            id.append(key)

        return np.reshape(s, (-1, 8)), id

    def step(self, a, near_end=False):
        # a is a dictionary: {id: action, id: action, ...}
        # since MCTS is used every 5 seconds, there may be new aircraft generated during the 5 time step interval, which
        # MCTS algorithm doesn't generate an action for it. In this case we let it fly straight.
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            try:
                aircraft.step(a[id])
            except KeyError:
                aircraft.step()

        # for vertiport in self.vertiport_list:
        #     vertiport.step()  # increase the clock of vertiport by 1
        #     # generate new aircraft if the clock pass the interval
        #     if vertiport.clock_counter >= vertiport.time_next_aircraft and not near_end:
        #         goal_vertiport_id = random.choice([e for e in range(len(self.vertiport_list)) if not e == vertiport.id])
        #         # generate new aircraft and prepare to add it the dict
        #         v1 = vertiport
        #         v2 = self.vertiport_list[goal_vertiport_id]
        #         route_dist = dist(v1.position[0], v1.position[1], v2.position[0], v2.position[1])
        #         if route_dist < 333:
        #             route = 1
        #         elif route_dist > 555:
        #             route = 3
        #         else:
        #             route = 2
        #
        #         aircraft = Aircraft(
        #             id=self.id_tracker,
        #             position=vertiport.position,
        #             speed=self.init_speed,
        #             heading=self.random_heading(),
        #             goal_pos=self.vertiport_list[goal_vertiport_id].position,
        #             goal_vertiport_id=goal_vertiport_id,
        #             sector_id=-1,
        #             priority=np.random.randint(0, 2),
        #             route=route,
        #             start_time=self.time_step
        #         )
        #         # calc its dist to all the other aircraft
        #         dist_array, id_array = self.dist_to_all_aircraft(aircraft)
        #         min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
        #         # add it to dict only if it's far from others
        #         if min_dist > 3 * self.minimum_separation:  # and self.aircraft_dict.num_aircraft < 10:
        #             self.aircraft_dict.add(aircraft)
        #             self.id_tracker += 1  # increase id_tracker
        #
        #             vertiport.generate_interval()  # reset clock for this vertiport and generate a new time interval

        # return the reward, done, and info
        reward, terminal, info = self._terminal_reward()

        self.total_timesteps += self.aircraft_dict.num_aircraft
        self.time_step += 1

        return self._get_ob(), reward, terminal, info

    def _terminal_reward(self):
        """
        determine the reward and terminal for the current transition, and use info. Main idea:
        1. for each aircraft:
          a. if there a conflict, return a penalty for it
          b. if there is NMAC, assign a penalty to it and prepare to remove this aircraft from dict
          b. elif it is out of map, assign its reward as Config.wall_penalty, prepare to remove it
          c. elif if it reaches goal, assign its reward to Config.goal_reward, prepare to remove it
          d. else assign its reward as Config.step_penalty.
        3. remove out-of-map aircraft and goal-aircraft

        """
        reward = 0
        # info = {'n': [], 'c': [], 'w': [], 'g': []}
        info_dist_list = []
        aircraft_to_remove = []  # add goal-aircraft and out-of-map aircraft to this list

        for id, aircraft in self.aircraft_dict.ac_dict.items():
            # calculate min_dist and dist_goal for checking terminal
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
            info_dist_list.append(min_dist)
            dist_goal = self.dist_goal(aircraft)

            conflict = False
            # set the conflict flag to false for aircraft
            # elif conflict, set penalty reward and conflict flag but do NOT remove the aircraft from list
            for id2, dist in zip(id_array, dist_array):
                if dist >= self.minimum_separation:  # safe
                    aircraft.conflict_id_set.discard(id2)  # discarding element not in the set won't raise error

                else:  # conflict!!
                    if self.debug:
                        self.render()
                        import ipdb
                        ipdb.set_trace()
                    conflict = True
                    if id2 not in aircraft.conflict_id_set:
                        self.conflicts += 1
                        aircraft.conflict_id_set.add(id2)
                        self.conflict_flag[id] = True
                        self.conflict_flag[id2] = True
                        # info['c'].append('%d and %d' % (aircraft.id, id))
                    aircraft.reward = Config.conflict_penalty

            # if NMAC, set penalty reward and prepare to remove the aircraft from list
            if min_dist < self.NMAC_dist:
                if self.debug:
                    self.render()
                    import ipdb
                    ipdb.set_trace()
                # info['n'].append('%d and %d' % (aircraft.id, close_id))
                aircraft.reward = Config.NMAC_penalty
                aircraft_to_remove.append(aircraft)
                self.NMACs += 1
                self.NMAC_flag[id] = True
                # aircraft_to_remove.append(self.aircraft_dict.get_aircraft_by_id(close_id))

            # give out-of-map aircraft a penalty, and prepare to remove it
            # elif not self.position_range.contains(np.array(aircraft.position)):
            #     aircraft.reward = Config.wall_penalty
            #     # info['w'].append(aircraft.id)
            #     if aircraft not in aircraft_to_remove:
            #         aircraft_to_remove.append(aircraft)

            # set goal-aircraft reward according to simulator, prepare to remove it
            elif dist_goal < self.goal_radius:
                aircraft.reward = Config.goal_reward
                # info['g'].append(aircraft.id)
                self.goals += 1
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)

                # self.route_time[aircraft.priority][aircraft.route].append(self.time_step - aircraft.start_time)

            # for aircraft without NMAC, conflict, out-of-map, goal, set its reward as simulator
            elif not conflict:
                aircraft.reward = Config.step_penalty

            # accumulates reward
            reward += aircraft.reward

        # remove all the out-of-map aircraft and goal-aircraft
        for aircraft in aircraft_to_remove:
            self.aircraft_dict.remove(aircraft)
        # reward = [e.reward for e in self.aircraft_dict]

        # info_dist_list is the min_dist to other aircraft for each aircraft.
        return reward, False, info_dist_list

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        from colour import Color
        red = Color('red')
        black = Color('black')
        colors = list(red.range_to(Color('green'), self.num_aircraft))

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # plot the annulus
        # inner_circle_img = rendering.make_circle(radius=10000 / Config.scale, res=50, filled=False)
        # outer_circle_img = rendering.make_circle(radius=15000 / Config.scale, res=50, filled=False)
        # jtransform = rendering.Transform(rotation=0, translation=(400, 400))
        # inner_circle_img.add_attr(jtransform)
        # outer_circle_img.add_attr(jtransform)
        # self.viewer.onetime_geoms.append(inner_circle_img)
        # self.viewer.onetime_geoms.append(outer_circle_img)

        # draw all the aircraft
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            aircraft_img = rendering.Image(os.path.join(__location__, 'images/aircraft.png'), 32, 32)
            jtransform = rendering.Transform(rotation=aircraft.heading - math.pi / 2, translation=aircraft.position)
            aircraft_img.add_attr(jtransform)
            r, g, b = colors[aircraft.id % self.num_aircraft].get_rgb()
            # r, g, b = black.get_rgb()
            aircraft_img.set_color(r, g, b)
            self.viewer.onetime_geoms.append(aircraft_img)

            goal_img = rendering.Image(os.path.join(__location__, 'images/goal.png'), 20, 20)
            jtransform = rendering.Transform(rotation=0, translation=aircraft.goal.position)
            goal_img.add_attr(jtransform)
            goal_img.set_color(r, g, b)
            self.viewer.onetime_geoms.append(goal_img)

            # plot the line connecting aircraft and its goal
            # line = rendering.DashedLine(start=aircraft.position, end=aircraft.goal.position)
            # self.viewer.onetime_geoms.append(line)

        # draw all the vertiports
        for veriport in self.vertiport_list:
            vertiport_img = rendering.Image(os.path.join(__location__, 'images/verti.png'), 32, 32)
            jtransform = rendering.Transform(rotation=0, translation=veriport.position)
            vertiport_img.add_attr(jtransform)
            self.viewer.onetime_geoms.append(vertiport_img)

        # class DashedLine(rendering.Geom):
        #     def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        #         Geom.__init__(self)
        #         self.start = start
        #         self.end = end
        #         self.linewidth = LineWidth(4)
        #         self.add_attr(self.linewidth)
        #
        #     def render1(self):
        #         glLineStipple(7, 0xAAAA)  # [1]
        #         glEnable(GL_LINE_STIPPLE)
        #         glBegin(GL_LINES)
        #         glVertex2f(*self.start)
        #         glVertex2f(*self.end)
        #         glEnd()

        return self.viewer.render(return_rgb_array=False)

    def draw_point(self, point):
        # for debug
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        img = self.viewer.draw_polygon(Config.point)
        jtransform = rendering.Transform(rotation=0, translation=point)
        img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(img)

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        for veriport in self.vertiport_list:
            vertiport_img = rendering.Image(os.path.join(__location__, 'images/verti.png'), 32, 32)
            jtransform = rendering.Transform(rotation=0, translation=veriport.position)
            vertiport_img.add_attr(jtransform)
            self.viewer.onetime_geoms.append(vertiport_img)

        return self.viewer.render(return_rgb_array=False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # dist to all the aircraft
    def dist_to_all_aircraft(self, aircraft):
        id_list = []
        dist_list = []
        for id, intruder in self.aircraft_dict.ac_dict.items():
            if id != aircraft.id:
                id_list.append(id)
                dist_list.append(self.metric(aircraft.position, intruder.position))

        return np.array(dist_list), np.array(id_list)

    def dist_goal(self, aircraft):
        return self.metric(aircraft.position, aircraft.goal.position)

    def metric(self, pos1, pos2):
        # the distance between two points
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    # def dist(self, pos1, pos2):
    #     return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def random_pos(self):
        return np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height])
        )

    def random_speed(self):
        return np.random.uniform(low=self.min_speed, high=self.max_speed)

    def random_heading(self):
        return np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'pos_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'pos_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
            'vel_x': spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'vel_y': spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=self.min_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'heading': spaces.Box(low=0, high=2 * math.pi, shape=(1,), dtype=np.float32),
            'goal_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'goal_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
        })

        return spaces.Tuple((s,) * self.num_aircraft)


class AircraftDict:
    def __init__(self):
        self.ac_dict = OrderedDict()

    # how many aircraft currently en route
    @property
    def num_aircraft(self):
        return len(self.ac_dict)

    # add aircraft to dict
    def add(self, aircraft):
        # id should always be different
        assert aircraft.id not in self.ac_dict.keys(), 'aircraft id %d already in dict' % aircraft.id
        self.ac_dict[aircraft.id] = aircraft

    # remove aircraft from dict
    def remove(self, aircraft):
        try:
            del self.ac_dict[aircraft.id]
        except KeyError:
            pass

    # get aircraft by its id
    def get_aircraft_by_id(self, aircraft_id):
        return self.ac_dict[aircraft_id]


# class AircraftList:
#     def __init__(self):
#         self.ac_list = []
#         self.id_list = []
#
#     @property
#     def num_aircraft(self):
#         return len(self.ac_list)
#
#     def add(self, aircraft):
#         self.ac_list.append(aircraft)
#         self.id_list.append(aircraft.id)
#         assert len(self.ac_list) == len(self.id_list)
#
#         unique, count = np.unique(np.array(self.id_list), return_counts=True)
#         assert np.all(count < 2), 'ununique id added to list'
#
#     def remove(self, aircraft):
#         try:
#             self.ac_list.remove(aircraft)
#             self.id_list.remove(aircraft.id)
#             assert len(self.ac_list) == len(self.id_list)
#         except ValueError:
#             pass
#
#     def get_aircraft_by_id(self, aircraft_id):
#         index = np.where(np.array(self.id_list) == aircraft_id)[0]
#         assert index.shape[0] == 1, 'find multi aircraft with id %d' % aircraft_id
#         return self.ac_list[int(index)]
#
#         for aircraft in self.buffer_list:
#             if aircraft.id == aircraft_id:
#                 return aircraft


class Goal:
    def __init__(self, position):
        self.position = position

    def __repr__(self):
        s = 'pos: %s' % self.position
        return s


class Aircraft:
    def __init__(self, id, position, speed, heading, goal_pos, goal_vertiport_id=-1, sector_id=-1, priority=0, route=0,
                 start_time=0):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.heading = heading  # rad
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy], dtype=np.float32)

        self.priority = priority
        self.route = route
        self.start_time = start_time

        self.reward = 0
        self.goal = Goal(goal_pos)
        self.sub_goal = Goal(goal_pos)
        self.goal_vertiport_id = goal_vertiport_id
        dx, dy = self.goal.position - self.position
        self.heading = math.atan2(dy, dx)  # set its initial heading point to its goal

        self.load_config()

        self.conflict_id_set = set()  # store the id of all aircraft currently in conflict

    def load_config(self):
        self.G = Config.G
        self.scale = Config.scale
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.speed_sigma = Config.speed_sigma
        # self.position_sigma = Config.position_sigma
        self.d_heading = Config.d_heading

    def step(self, a=1):
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))  # project to range
        self.speed += np.random.normal(0, self.speed_sigma)  # uncertainty
        self.heading += (a - 1) * self.d_heading + np.random.normal(0, Config.heading_sigma)  # change heading
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])

        self.position += self.velocity

    def __repr__(self):
        s = 'id: %d, pos: %.2f,%.2f, speed: %.2f, heading: %.2f goal: %.2f,%.2f, sub-goal: %.2f,%.2f' \
            % (self.id,
               self.position[0],
               self.position[1],
               self.speed,
               math.degrees(self.heading),
               self.goal.position[0],
               self.goal.position[1],
               self.sub_goal.position[0],
               self.sub_goal.position[1]
               )
        return s


class VertiPort:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position)  # position of vertiport
        self.clock_counter = 0
        self.time_next_aircraft = np.random.uniform(0, 60)

    # when the next aircraft will take off
    def generate_interval(self):
        # time interval to generate next aircraft
        self.time_next_aircraft = np.random.uniform(Config.time_interval_lower, Config.time_interval_upper)
        self.clock_counter = 0

    # add the clock counter by 1
    def step(self):
        self.clock_counter += 1

    def __repr__(self):
        return 'vertiport id: %d, pos: %s' % (self.id, self.position)


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return (dx ** 2 + dy ** 2) ** 0.5
