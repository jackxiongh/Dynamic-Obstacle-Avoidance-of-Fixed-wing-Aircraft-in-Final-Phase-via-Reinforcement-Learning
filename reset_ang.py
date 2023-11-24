import numpy as np
from gym import spaces
import scipy.spatial.distance as distance
import os
import pandas as pd

def reset_an(num=1000, name=None, init_radius=5000, ss_r=3000, random_seed=0):
    if not os.path.exists('./reset_ang'):
        os.mkdir('./reset_ang')

    if name is None:
        name = 'ss_r{}'.format(ss_r)

    np.random.seed(random_seed)

    initial_space = spaces.Box(low=np.array([0, 2000, init_radius, 0, 0, 0, 51, 0, 0, 0]),
                               high=np.array([0, 2000, init_radius, 0, 0, 0, 51, 0, 0, 0]), dtype=np.float32)
    init_list = pd.DataFrame(columns=['ang_rad', 'cour_ran', 'alt_ran', 'speed_obs', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'AS', 'v_roll', 'v_pitch', 'v_yaw'])
    for i in range(num):
        initial_s = initial_space.sample()
        speed_obs = np.random.uniform(0, 80)
        while True:
            ang_rad = np.random.uniform(-np.pi, np.pi)
            dis_obs = initial_s[2] *speed_obs /initial_s[6]
            cour_random = np.random.uniform(-150, 150)
            alt_random = np.random.uniform(-150, 150)
            pos_i = [dis_obs * np.cos(ang_rad) + cour_random * np.sin(ang_rad),
                     initial_s[1] + alt_random,
                     dis_obs * np.sin(ang_rad) + cour_random * np.cos(ang_rad)]
            if distance.euclidean(initial_s[0:3], pos_i) > ss_r:
                break
        init_list.loc[i] = [ang_rad, cour_random, alt_random, speed_obs] + list(initial_s)
    fp = './reset_ang/{}.xlsx'.format(name)
    init_list.to_excel(fp, index=False)

if __name__ == '__main__':
    for i in range(3):
        ss_r = 1000*(i+1)
        name = 'obs_ran{}'.format(ss_r)
        reset_an(name=name, ss_r=ss_r)



