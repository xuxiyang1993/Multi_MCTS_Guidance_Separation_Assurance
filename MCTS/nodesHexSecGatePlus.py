import copy
import math
import numpy as np
# from shapely.geometry import Polygon, Point
import matplotlib.path as mpltPath

from common import MCTSNode, MCTSState
from config_hex_sec import Config


# from config_multi import Config


class MultiAircraftState(MCTSState):
    def __init__(self,
                 state,
                 index,
                 init_action,
                 sector_id,
                 goal_exit_id,
                 hit_wall=False,
                 conflict=False,
                 reach_goal=False,
                 reach_subgoal=False,
                 prev_action=None,
                 depth=0):
        MCTSState.__init__(self, state)
        self.index = index
        self.init_action = init_action
        self.sector_id = sector_id
        self.goal_exit_id = goal_exit_id
        self.hit_wall = hit_wall
        self.conflict = conflict
        self.reach_goal = reach_goal
        self.reach_subgoal = reach_subgoal
        self.prev_action = prev_action
        self.depth = depth

        self.G = Config.G
        self.scale = Config.scale

        self.nearest_x = -1
        self.nearest_y = -1

    # reward needs to be tuned
    def reward(self):
        if not self.reach_subgoal:
            if self.conflict:
                r = 0
            elif self.reach_goal:
                r = 1
            elif self.hit_wall:
                r = 0.1
            else:
                r = 1 - self.dist_goal() / 1200.0  # - self.dist_intruder(self.state, self.ownx, self.owny) / 1200
                r /= 2

        else:
            if self.conflict:
                r = 0
            elif self.hit_wall:
                r = 0.1
            else:
                r = 1
        return r

    def is_terminal_state(self, search_depth):
        if self.reach_goal or self.conflict or self.hit_wall or self.depth == search_depth:
            return True
        return False

    def move(self, a):
        # if self.depth < 1:
        #     next_state = self._move(a)
        # else:
        #     random_action = np.random.randint(0, 3, size=self.state.shape[0])
        #     next_state = self._move(random_action)
        #     # print('rand2')
        next_state = self._move(a)

        return next_state

    def _move(self, a):
        # state: dimension: n by 8
        # [aircraft: x, y, vx, vy, v, heading, gx, gy]
        state = copy.deepcopy(self.state)
        hit_wall = False
        conflict = False
        reach_goal = False
        reach_subgoal = self.reach_subgoal

        for _ in range(Config.simulate_frame):
            for index in range(state.shape[0]):
                heading = state[index, 5] + (a[index] - 1) * Config.d_heading \
                          + np.random.normal(0, Config.heading_sigma)
                speed = Config.init_speed + np.random.normal(0, Config.speed_sigma)
                speed = max(Config.min_speed, min(speed, Config.max_speed))  # restrict to range
                vx = speed * math.cos(heading)
                vy = speed * math.sin(heading)
                state[index, 0] += vx
                state[index, 1] += vy

                state[index, 2] = vx
                state[index, 3] = vy
                state[index, 4] = speed
                state[index, 5] = heading

            ownx = state[self.index][0]
            owny = state[self.index][1]
            goalx = state[self.index][6]
            goaly = state[self.index][7]

            # if self.dist_intruder(state, ownx, owny) < Config.minimum_separation:
            if self.conflict_intruder(state, ownx, owny):
                conflict = True
                break

            # if not sub goal and aircraft reaches goal
            if self.goal_exit_id == -1 and self.metric(ownx, owny, goalx, goaly) < Config.goal_radius:
                reach_goal = True
                break

            # if aircraft close to sector exit gate, sub_goal = True
            if not self.goal_exit_id == -1 and \
                pnt2line(np.array([ownx, owny]),
                         Config.sector_len_exits[self.sector_id][self.goal_exit_id][1],
                         Config.sector_len_exits[self.sector_id][self.goal_exit_id][2])[0] < 4:
                reach_subgoal = True

            # if not Polygon(Config.sector_vertices[self.sector_id]).contains(Point(ownx, owny)) and not reach_subgoal:
            # if not self.in_hull([ownx, owny], Config.sector_vertices[self.sector_id]) and not reach_subgoal:
            if not mpltPath.Path(Config.sector_vertices[self.sector_id]).contains_point([ownx, owny]) and not reach_subgoal:
                hit_wall = True
                break

            # if self.dist_entries(ownx, owny, Config.sector_entries[self.sector_id]) < 2 * Config.minimum_separation:
            #     hit_wall = True
            #     break

        return MultiAircraftState(state=state,
                                  index=self.index,
                                  init_action='random',
                                  sector_id=self.sector_id,
                                  goal_exit_id=self.goal_exit_id,
                                  hit_wall=hit_wall,
                                  conflict=conflict,
                                  reach_goal=reach_goal,
                                  reach_subgoal=reach_subgoal,
                                  prev_action=a,
                                  depth=self.depth + 1)
        return MultiAircraftState(state, self.index, 'random', hit_wall, conflict, reach_goal, a, self.depth + 1)

    # def in_hull(self, p, hull):
    #     """
    #     Test if points in `p` are in `hull`
    #
    #     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    #     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    #     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    #     will be computed
    #     """
    #     from scipy.spatial import Delaunay
    #     if not isinstance(hull, Delaunay):
    #         hull = Delaunay(hull)
    #
    #     return hull.find_simplex(p) >= 0

    def point_to_line_dist(self, point, line):
        """Calculate the distance between a point and a line segment.

        To calculate the closest distance to a line segment, we first need to check
        if the point projects onto the line segment.  If it does, then we calculate
        the orthogonal distance from the point to the line.
        If the point does not project to the line segment, we calculate the
        distance to both endpoints and take the shortest distance.

        :param point: Numpy array of form [x,y], describing the point.
        :type point: numpy.core.multiarray.ndarray
        :param line: list of endpoint arrays of form [P1, P2]
        :type line: list of numpy.core.multiarray.ndarray
        :return: The minimum distance to a point.
        :rtype: float
        """
        # unit vector
        unit_line = line[1] - line[0]
        norm_unit_line = unit_line / np.linalg.norm(unit_line)

        # compute the perpendicular distance to the theoretical infinite line
        segment_dist = (
                np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
                np.linalg.norm(unit_line)
        )

        diff = (
                (norm_unit_line[0] * (point[0] - line[0][0])) +
                (norm_unit_line[1] * (point[1] - line[0][1]))
        )

        x_seg = (norm_unit_line[0] * diff) + line[0][0]
        y_seg = (norm_unit_line[1] * diff) + line[0][1]

        endpoint_dist = min(
            np.linalg.norm(line[0] - point),
            np.linalg.norm(line[1] - point)
        )

        # decide if the intersection point falls on the line segment
        lp1_x = line[0][0]  # line point 1 x
        lp1_y = line[0][1]  # line point 1 y
        lp2_x = line[1][0]  # line point 2 x
        lp2_y = line[1][1]  # line point 2 y
        is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
        is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
        if is_betw_x and is_betw_y:
            return segment_dist
        else:
            # if not, then return the minimum distance to the segment endpoints
            return endpoint_dist

    def get_legal_actions(self):
        return [0, 1, 2]

    def dist_entries(self, x, y, points):
        dists = [self.metric(x, y, e[0], e[1]) for e in points]
        return min(dists)

    def dist_goal(self):
        dx = self.ownx - self.goalx
        dy = self.owny - self.goaly
        return math.sqrt(dx ** 2 + dy ** 2)

    def conflict_intruder(self, state, ownx, owny):
        for i in [x for x in range(state.shape[0]) if x != self.index]:
            otherx = state[i][0]
            othery = state[i][1]
            dist = self.metric(ownx, owny, otherx, othery)
            if dist < Config.minimum_separation:
                return True
        return False

    def dist_intruder(self, state, ownx, owny):
        distance = 5000
        for i in [x for x in range(state.shape[0]) if x != self.index]:
            otherx = state[i][0]
            othery = state[i][1]
            dist = self.metric(ownx, owny, otherx, othery)
            if dist < distance:
                distance = dist
                self.nearest_x = otherx
                self.nearest_y = othery
        return distance

    def metric(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx ** 2 + dy ** 2)

    # state: (x, y, vx, vy, heading angle, gx, gy)
    @property
    def ownx(self):
        return self.state[self.index][0]

    @property
    def owny(self):
        return self.state[self.index][1]

    @property
    def goalx(self):
        return self.state[self.index][6]

    @property
    def goaly(self):
        return self.state[self.index][7]

    def __repr__(self):
        s = 'index: %d, prev action: %s, pos: %.2f,%.2f, goal: %.2f,%.2f, dist goal: %.2f, dist intruder: %f,' \
            'nearest intruder: (%.2f, %.2f), depth: %d' \
            % (self.index,
               self.prev_action,
               self.ownx,
               self.owny,
               self.goalx,
               self.goaly,
               self.dist_goal(),
               self.dist_intruder(self.state, self.ownx, self.owny),
               self.nearest_x,
               self.nearest_y,
               self.depth)
        return s


class MultiAircraftNode(MCTSNode):
    def __init__(self, state: MultiAircraftState, parent=None):
        MCTSNode.__init__(self, parent)
        self.state = state

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def reward(self):
        return self.q / self.n if self.n else 0

    def expand(self):
        a = self.untried_actions.pop()
        if self.state.init_action == 'random':
            all_action = np.random.randint(0, 3, size=self.state.state.shape[0])
            # print('rand1')
        else:
            all_action = self.state.init_action.copy()
        all_action[self.state.index] = a
        next_state = self.state.move(all_action)
        child_node = MultiAircraftNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self, search_depth):
        return self.state.is_terminal_state(search_depth)

    def rollout(self, search_depth):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal_state(search_depth):
            # possible_moves = current_rollout_state.get_legal_actions()
            # action = self.rollout_policy(possible_moves)
            action = np.random.randint(0, 3, size=self.state.state.shape[0])
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.reward()

    def backpropagate(self, result):
        self.n += 1
        self.q += result
        if self.parent:
            self.parent.backpropagate(result)

    def __repr__(self):
        s = 'Agent: %d, Node: children: %d; visits: %d; reward: %.4f; p_action: %s, state: (%.2f, %.2f); ' \
            'goal: (%.2f, %.2f), dist_goal: %.2f, nearest: (%.2f, %.2f)' \
            % (self.state.index + 1,
               len(self.children),
               self.n,
               self.q / (self.n + 1e-2),
               self.state.prev_action,
               self.state.ownx,
               self.state.owny,
               self.state.goalx,
               self.state.goaly,
               self.state.dist_goal(),
               self.state.nearest_x,
               self.state.nearest_y)

        return s


def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def add(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)
