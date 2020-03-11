import argparse
import numpy as np
import time

import sys
sys.path.extend(['../Simulators'])
from nodesHexSecGatePlus import MultiAircraftNode, MultiAircraftState
from search_multi import MCTS
from config_hex_sec import Config
from MultiAircraftVertiHexSecGatePlusEnv import MultiAircraftEnv
# from testEnv import MultiAircraftEnv


def run_experiment(env, no_episodes, render, save_path):
    text_file = open(save_path, "w")  # save all non-terminal print statements in a txt file
    episode = 0
    epi_returns = []
    conflicts_list = []
    enroute_number_list = []
    num_aircraft = Config.num_aircraft
    time_dict = {}
    route_time = {1: [], 2: [], 3: []}

    while episode < no_episodes:
        # at the beginning of each episode, set done to False, set time step in this episode to 0
        # set reward to 0, reset the environment
        episode += 1
        done = False
        episode_time_step = 0
        episode_reward = 0
        last_observation = env.reset()
        # last_observation = env.pressure_reset()
        action_by_id = {}
        info = None
        near_end = False
        counter = 0  # avoid end episode initially

        while not done:
            if render:
                env.render()
            if episode_time_step % 5 == 0:
                # if env.id_tracker > 84 and env.debug:
                #     import ipdb; ipdb.set_trace()
                action_by_id = {}
                time_list = []
                for i in range(7):

                    ob_by_sector, id_list, goal_exit_id_list = last_observation[i]
                    time_before = int(round(time.time() * 1000))
                    # import datetime
                    # start = datetime.datetime.now()
                    num_considered_aircraft = len(id_list)
                    num_existing_aircraft = ob_by_sector.shape[0]
                    # if not num_considered_aircraft == num_existing_aircraft:
                    #     import ipdb; ipdb.set_trace()
                    action = np.ones(num_existing_aircraft, dtype=np.int32)

                    for index in range(num_considered_aircraft):
                        state = MultiAircraftState(state=ob_by_sector,
                                                   index=index,
                                                   init_action=action,
                                                   sector_id=i,
                                                   goal_exit_id=goal_exit_id_list[index])
                        root = MultiAircraftNode(state=state)
                        mcts = MCTS(root)
                        if info[id_list[index]] < 2 * Config.minimum_separation:
                            best_node = mcts.best_action(Config.no_simulations, Config.search_depth)
                        else:
                            best_node = mcts.best_action(Config.no_simulations_lite, Config.search_depth_lite)

                        # if id_list[index] == 103 or id_list[index] == 123:
                        #     if env.id_tracker > 130:
                        #         import ipdb; ipdb.set_trace()

                        action[index] = best_node.state.prev_action[index]
                        action_by_id[id_list[index]] = best_node.state.prev_action[index]

                    # print("[INFO] {} aircraft took: {}ms".format(num_considered_aircraft,
                    #                                              (datetime.datetime.now() - start).total_seconds()*1000))
                    time_after = int(round(time.time() * 1000))
                    # if num_considered_aircraft in time_dict:
                    #     time_dict[num_considered_aircraft].append(time_after - time_before)
                    # else:
                    #     time_dict[num_considered_aircraft] = [time_after - time_before]

                    time_list.append(time_after - time_before)

                if env.aircraft_dict.num_aircraft in time_dict:
                    time_dict[env.aircraft_dict.num_aircraft].append(max(time_list))
                else:
                    time_dict[env.aircraft_dict.num_aircraft] = [max(time_list)]

            observation, reward, done, info = env.step(action_by_id, near_end)

            episode_reward += reward
            last_observation = observation
            episode_time_step += 1

            if episode_time_step % 100 == 0:
                print('========================== Time Step: %d =============================' % episode_time_step,
                      file=text_file)
                print('Number of conflicts:', env.conflicts / 2, file=text_file)
                print('Total Aircraft Genrated:', env.id_tracker, file=text_file)
                print('Goal Aircraft:', env.goals, file=text_file)
                print('NMACs:', env.NMACs / 2, file=text_file)
                print('NMAC/h:', (env.NMACs / 2) / (env.total_timesteps / 3600), file=text_file)
                print('Total Flight Hours:', env.total_timesteps / 3600, file=text_file)
                print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft, file=text_file)
                print('Time:', file=text_file)
                for key, item in time_dict.items():
                    print(key, np.mean(item), file=text_file)
                enroute_number_list.append(env.aircraft_dict.num_aircraft)
                print('Enroute Aircraft Number:', enroute_number_list, file=text_file)

                print('========================== Time Step: %d =============================' % episode_time_step)
                print('Number of conflicts:', env.conflicts / 2)
                print('Total Aircraft Genrated:', env.id_tracker)
                print('Goal Aircraft:', env.goals)
                print('NMACs:', env.NMACs / 2)
                print('NMAC/h:', (env.NMACs / 2) / (env.total_timesteps / 3600))
                print('Total Flight Hours:', env.total_timesteps / 3600)
                print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft)

                # print('Time:')
                # for key, item in time_dict.items():
                #     print(key, np.mean(item))

            if env.id_tracker - 1 >= 10000:
                counter += 1
                near_end = True

            if episode_time_step > 100 and env.aircraft_dict.num_aircraft == 0:
                break

        print('route 1 time:', env.route_time[0][1] + env.route_time[1][1], file=text_file)
        print('route 2 time:', env.route_time[0][2] + env.route_time[1][2], file=text_file)
        print('route 3 time:', env.route_time[0][3] + env.route_time[1][3], file=text_file)

        print('========================== End =============================', file=text_file)
        print('========================== End =============================')
        print('Number of conflicts:', env.conflicts / 2)
        print('Total Aircraft Genrated:', env.id_tracker)
        print('Goal Aircraft:', env.goals)
        print('NMACs:', env.NMACs / 2)
        print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft)
        for key, item in time_dict.items():
            print('%d aircraft: %.2f' % (key, np.mean(item)))

        # print training information for each training episode
        epi_returns.append(info)
        conflicts_list.append(env.conflicts)
        print('Training Episode:', episode)
        print('Cumulative Reward:', episode_reward)

    time_list = time_dict.values()
    flat_list = [item for sublist in time_list for item in sublist]
    print('----------------------------------------')
    print('Number of aircraft:', Config.num_aircraft)
    print('Search depth:', Config.search_depth)
    print('Simulations:', Config.no_simulations)
    print('Time:', sum(flat_list) / float(len(flat_list)))
    print('NMAC prob:', epi_returns.count('n') / no_episodes)
    print('Goal prob:', epi_returns.count('g') / no_episodes)
    print('Average Conflicts per episode:',
          sum(conflicts_list) / float(len(conflicts_list)) / 2)  # / 2 to ignore duplication
    env.close()
    text_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_episodes', '-e', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--save_path', '-p', type=str, default='output/seed2.txt')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--render', '-r', action='store_true')
    args = parser.parse_args()

    import random
    np.set_printoptions(suppress=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = MultiAircraftEnv(args.seed, args.debug)
    run_experiment(env, args.no_episodes, args.render, args.save_path)


if __name__ == '__main__':
    main()
