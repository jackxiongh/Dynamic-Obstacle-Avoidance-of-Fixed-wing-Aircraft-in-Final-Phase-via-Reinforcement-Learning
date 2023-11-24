# 2021.11.26
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

class logger:
    """
    record the information of training
    """
    def __init__(self, name='untitle'):
        self.name = name
        self.episode = 0   # num of episodes
        self.total_step = 0   # num of total steps of all existed episodes
        self.episode_step = 0  # num of total steps of current episodes

        self.episode_rew = []    # sum of rewards in every episodes
        self.cur_episode_rew = 0   # sum of rewards in current episode
        self.mean_rew = []   # mean reward of every steps in every episodes
        self.cur_mean_rew = 0  # mean reward of every steps in current episodes

        self.cur_epi_state = pd.DataFrame(columns=['x', 'y', 'z', 'roll', 'pitch', 'yaw',
                                                   'AS', 'v_x', 'v_y', 'v_z',
                                                   'v_roll', 'v_pitch', 'v_yaw', 'x_i', 'y_i', 'z_i', 'vx_i', 'vy_i', 'vz_i',
                                                   'tar_vx', 'tar_vy', 'tar_vz', 'entropy'])
        self.net_time = []   # time of update network
        self.cur_epi_time = 0
        self.epi_time = []   # time of every episode

        self.info = pd.DataFrame(columns=['warn', 'crash'])

        self.create_path()

    def step_update(self, rew, t, state, action, entropy):
        """
        update as a step over
        """
        self.total_step += 1
        self.episode_step += 1

        self.cur_epi_state.loc[self.episode_step-1] = state + action + [entropy]

        self.cur_episode_rew += rew
        self.cur_epi_time += t

    def epi_update(self, info, save=True, chart=False):
        """
        update as a episode over
        """
        self.episode += 1
        self.cur_mean_rew = self.cur_episode_rew / self.episode_step
        self.episode_rew.append(self.cur_episode_rew)
        self.mean_rew.append(self.cur_episode_rew/self.episode_step)
        self.epi_time.append(self.cur_epi_time)
        self.info.loc[self.episode-1] = info
        if save:
            fp1 = 'data/reward/{}/'.format(self.name)
            fp2 = 'data/states/{}/'.format(self.name)
            fp3 = 'data/infos/{}/'.format(self.name)
            np.savetxt(fp1 + self.name + '_' + 'episode_rew.txt', self.episode_rew, delimiter="\n")
            np.savetxt(fp1 + self.name + '_' + 'mean_rew.txt', self.mean_rew, delimiter="\n")
            self.cur_epi_state.to_excel(fp2 + self.name + '_' + 'episode{}.xlsx'.format(self.episode), index=False)
            np.savetxt(fp3 + self.name + '_' + 'info.txt', self.info.to_numpy())
            np.savetxt(fp3 + self.name + '_' + 'net_time.txt', np.array(self.net_time))
            np.savetxt(fp3 + self.name + '_' + 'epi_time.txt', np.array(self.epi_time))
        if chart:
            plt.ion()

            plt.xlabel('episodes')
            plt.ylabel('mean_rew')
            plt.plot(self.mean_rew)

            plt.pause(3.)
            plt.close()

        self.cur_episode_rew = 0
        self.episode_step = 0
        self.cur_epi_state = pd.DataFrame(columns=['x', 'y', 'z', 'roll', 'pitch', 'yaw',
                                                   'AS', 'v_x', 'v_y', 'v_z',
                                                   'v_roll', 'v_pitch', 'v_yaw', 'x_i', 'y_i', 'z_i', 'vx_i', 'vy_i', 'vz_i',
                                                   'tar_vx', 'tar_vy', 'tar_vz', 'entropy'])
        self.cur_epi_time = 0

    def update(self, chart=False):
        """
        update as entire training over
        """
        if chart:
            plt.figure(1)
            plt.xlabel('episodes')
            plt.ylabel('mean_rew')
            plt.plot(self.mean_rew)

            plt.figure(2)
            plt.xlabel('episodes')
            plt.ylabel('epi_time')
            plt.plot(self.epi_time)

            plt.figure(3)
            plt.xlabel('index')
            plt.ylabel('net_time')
            plt.plot(self.epi_time)

            plt.show()
        self.episode = 0  # num of episodes
        self.total_step = 0  # num of total steps of all existed episodes
        self.episode_step = 0  # num of total steps of current episodes

        self.episode_rew = []  # sum of rewards in every episodes
        self.cur_episode_rew = 0  # sum of rewards in current episode
        self.mean_rew = []  # mean reward of every steps
        self.cur_mean_rew = 0

        self.net_time = []  # time of update network
        self.cur_epi_time = 0
        self.epi_time = []  # time of every episode

        pd.DataFrame(columns=['crash', 'success'])

    def create_path(self):
        if not os.path.exists('./model'):
            os.mkdir('./model')
        if not os.path.exists('./data'):
            os.mkdir('./data')
        if not os.path.exists('./buffer'):
            os.mkdir('./buffer')

        if not os.path.exists('./data/reward'):
            os.mkdir('./data/reward')
        if not os.path.exists('./data/infos'):
            os.mkdir('./data/infos')
        if not os.path.exists('./data/states'):
            os.mkdir('./data/states')

        if not os.path.exists('./model/' + self.name):
            os.mkdir('./model/' + self.name)
        if not os.path.exists('./data/reward/' + self.name):
            os.mkdir('./data/reward/' + self.name)
        if not os.path.exists('./data/infos/' + self.name):
            os.mkdir('./data/infos/' + self.name)
        if not os.path.exists('./data/states/' + self.name):
            os.mkdir('./data/states/' + self.name)
        if not os.path.exists('./buffer/' + self.name):
            os.mkdir('./buffer/' + self.name)
