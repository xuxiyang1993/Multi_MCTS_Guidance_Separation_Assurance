import math

print('loading multi configuration')

class Config:
    # experiment setting
    no_episodes = 100

    # airspace setting
    window_width = 800
    window_height = 800
    num_aircraft = 10
    EPISODES = 1000
    G = 9.8
    tick = 30
    scale = 30

    # distance param
    minimum_separation = 555/scale
    NMAC_dist = 150/scale
    horizon_dist = 4000/scale
    initial_min_dist = 3000/scale
    goal_radius = 600/scale

    # speed
    init_speed = 60/scale
    min_speed = 50/scale
    max_speed = 80/scale
    d_speed = 5/scale
    speed_sigma = 0
    position_sigma = 0

    # heading in rad TBD
    d_heading = math.radians(5)
    heading_sigma = math.radians(0)

    # MCTS algorithm
    no_simulations = 100
    search_depth = 3
    simulate_frame = 10

    # reward setting
    NMAC_penalty = -10
    conflict_penalty = -5
    wall_penalty = -5
    step_penalty = -0.01
    goal_reward = 20
    sparse_reward = True
