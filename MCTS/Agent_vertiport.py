import argparse
import numpy as np
import time

import sys

sys.path.extend(['../Simulators'])
from nodes_multi import MultiAircraftNode, MultiAircraftState
from search_multi import MCTS
from config_vertiport import Config
from MultiAircraftVertiportEnv import MultiAircraftEnv


def run_experiment(env, no_episodes, render, save_path, num_aircraft):
    text_file = open(save_path, "w")  # save all non-terminal print statements in a txt file
    episode = 0
    epi_returns = []
    conflicts_list = []
    NMACs_list = []
    # num_aircraft = Config.num_aircraft
    env.num_aircraft = num_aircraft
    time_dict = {}

    while episode < no_episodes:
        # at the beginning of each episode, set done to False, set time step in this episode to 0
        # set reward to 0, reset the environment
        episode += 1
        done = False
        episode_time_step = 1
        episode_reward = 0
        last_observation, id_list = env.reset()
        # last_observation, id_list = env.pressure_reset()
        # last_observation, id_list = env.pressure_reset1()
        action_by_id = {}
        info = None
        near_end = False
        counter = 0  # avoid end episode initially

        while not done:
            if render:
                env.render()
            if episode_time_step % 5 == 0:

                time_before = int(round(time.time() * 1000))
                num_existing_aircraft = last_observation.shape[0]
                action = np.ones(num_existing_aircraft, dtype=np.int32)
                action_by_id = {}

                for index in range(num_existing_aircraft):
                    # state = MultiAircraftState(state=last_observation, index=index, init_action=action)
                    # root = MultiAircraftNode(state=state)
                    # mcts = MCTS(root)
                    # if info[index] < 3 * Config.minimum_separation:
                    #     best_node = mcts.best_action(Config.no_simulations, Config.search_depth)
                    # else:
                    #     best_node = mcts.best_action(Config.no_simulations_lite, Config.search_depth_lite)
                    # action[index] = best_node.state.prev_action[index]
                    # action_by_id[id_list[index]] = best_node.state.prev_action[index]
                    action_by_id[id_list[index]] = 1

                time_after = int(round(time.time() * 1000))
                if num_existing_aircraft in time_dict:
                    time_dict[num_existing_aircraft].append(time_after - time_before)
                else:
                    time_dict[num_existing_aircraft] = [time_after - time_before]
            (observation, id_list), reward, done, info = env.step(action_by_id, near_end)

            episode_reward += reward
            last_observation = observation
            episode_time_step += 1

            if episode_time_step % 100 == 0 and False:
                print('========================== Time Step: %d =============================' % episode_time_step,
                      file=text_file)
                print('Number of conflicts:', env.conflicts / 2, file=text_file)
                print('Total Aircraft Genrated:', env.id_tracker, file=text_file)
                print('Goal Aircraft:', env.goals, file=text_file)
                print('NMACs:', env.NMACs / 2, file=text_file)
                print('NMAC/h:', (env.NMACs / 2) / (env.total_timesteps / 3600), file=text_file)
                print('Total Flight Hours:', env.total_timesteps / 3600, file=text_file)
                print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft, file=text_file)

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

            if episode_time_step > 10 and env.aircraft_dict.num_aircraft == 0:
                break

        # print('route 1 time:', env.route_time[0][1] + env.route_time[1][1], file=text_file)
        # print('route 2 time:', env.route_time[0][2] + env.route_time[1][2], file=text_file)
        # print('route 3 time:', env.route_time[0][3] + env.route_time[1][3], file=text_file)

        # print('========================== End =============================', file=text_file)
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
        # conflicts_list.append(env.conflicts)
        # NMACs_list.append(env.NMACs)
        conflicts_list.append(sum(env.conflict_flag))
        NMACs_list.append(sum(env.NMAC_flag))
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
    print('Conflict list:', file=text_file)
    print(conflicts_list, file=text_file)
    print('NMAC list:', file=text_file)
    print(NMACs_list, file=text_file)
    # print('Avg Conflicts/episode:', sum(conflicts_list) / float(len(conflicts_list)) / 2, file=text_file)  # / 2 to ignore duplication
    # print('Avg NMACs/episode:', sum(NMACs_list) / float(len(NMACs_list)) / 2, file=text_file)  # /2 to remove duplication
    print('Avg Conflict prob:', sum(conflicts_list) / float(len(conflicts_list)), file=text_file)  # / 2 to ignore duplication
    print('Avg NMACs prob:', sum(NMACs_list) / float(len(NMACs_list)), file=text_file)  # /2 to remove duplication
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

    env = MultiAircraftEnv(args.seed)
    for i in range(5, 21):
        path = 'r_output/f%d.txt' % i
        run_experiment(env, args.no_episodes, args.render, path, i)


if __name__ == '__main__':
    main()
