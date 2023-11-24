from gym import spaces
import numpy as np
import toolbox.xpc_20211025 as xp
from toolbox import pid_new as pid2
from toolbox import pid_vel as pid
import scipy.spatial.distance as distance


class parameters:
    def __init__(self, num_AI):
        '''
        The potentially used flight parameters of the aircraft in X-plane, see in
        https://www.siminnovations.com/xplane/dataref/index.php

        '''
        self.num_AI = num_AI
        self.geoPos = ['sim/flightmodel/position/latitude', 'sim/flightmodel/position/longitude']
        self.pos = ['sim/flightmodel/position/local_x',
                    'sim/flightmodel/position/local_y',
                    'sim/flightmodel/position/local_z',
                    'sim/flightmodel/position/phi',  # roll
                    'sim/flightmodel/position/theta',  # pitch
                    'sim/flightmodel/position/psi']  # yaw

        self.vel = ['sim/flightmodel/position/local_vx',
                    'sim/flightmodel/position/local_vy',
                    'sim/flightmodel/position/local_vz',
                    'sim/flightmodel/position/P',  # roll_rate
                    'sim/flightmodel/position/Q',  # pitch_rate
                    'sim/flightmodel/position/R']  # yaw_rate
        self.true_as = ['sim/flightmodel/position/true_airspeed']


class my_env:
    def __init__(self, name, clientAddr='0.0.0.0', xpHost='127.0.0.1', xpPort=49009, clientPort=1, timeout=10000):
        self.name = name
        self.para = parameters(1)
        # state space:
        low = np.array([-1000, -5000, -1000, -180, -180, -90, -60, 0, -100, -5000, -10000, -1500, -150, -165, -100])
        high = np.array([1000, 5000, 500, 180, 180, 90, 60, 60, 25, 5000, 10000, 1500, 150, 75, 100])
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # action space:
        self.action_space = spaces.Box(low=np.array([-60, 0, -60]), high=np.array([60, 60, 20]),
                                       dtype=np.float32)
        # The initial state of the aircraft(world frame of X-plane): x, y, z, roll, pitch, yaw
        self.initial_space = spaces.Box(low=np.array([0, 2000, 5000, 0, 0, 0, 40, 0, 0, 0]),
                                        high=np.array([0, 2000, 5000, 0, 0, 0, 60, 0, 0, 0]), dtype=np.float32)

        self.done = False
        self.crash = False
        self.warn = False   # the same as crash

        self.R = 170  # R_pz + 20
        self.ss_r = 1000   # sensing range
        self.tar_alt = 2000   # target altitude
        self.cruise_as = 51   # cruising airspeed(m/s)
        self.exist_obs = 1    # is obstacle existed, yes 1, no 0.

        try:
            self.client = xp.XPlaneConnect(clientAddr, xpHost, xpPort, clientPort, timeout)
        except:
            print('error parameter\n')
        print('I am client', self.client, '\n')

    def pause_sim(self):
        self.client.sendCOMM('sim/operation/pause_toggle')

    def get_state(self, rdata, norm=True, in_ssr=False):
        """
        rdata：the raw data from X-plane, represented in the world frame in the X-plane software.
        state：the state in RL, represented in local frame.
        """
        d_yaw = self.d_rpy(rdata[5], 0)
        pos_r = np.array(rdata[13:16]) - np.array(rdata[0:3])
        vel_r = np.array(rdata[16:19]) - np.array(rdata[7:10])
        state = [rdata[0], -rdata[2], rdata[1] - self.tar_alt] + rdata[3:5] + [d_yaw] + [rdata[7], -rdata[9],
                                                                                          rdata[8]] + \
              [pos_r[0], -pos_r[2], pos_r[1]] + [vel_r[0], -vel_r[2], vel_r[1]]
        if not in_ssr:
            state[9:] = [0, 0, 0, 0, 0, 0]
        state[1] = 0
        if norm:  # state normalization
            state = self.s_norm(state, self.state_space, clip=True)

        return np.array(state)

    def reset(self):
        self.done = False
        self.crash = False
        self.warn = False
        self.ss_r = np.random.uniform(1000, 3000)
        self.exist_obs = np.random.randint(0, 2)
        initial_s = self.initial_space.sample()
        speed_obs = np.random.uniform(0, 80)
        while True:
            ang = np.random.uniform(-np.pi, np.pi)
            dis_obs = initial_s[2]*speed_obs/self.cruise_as
            cour_random = np.random.uniform(-150, 150)
            alt_random = np.random.uniform(-150, 75)
            pos_i = [dis_obs * np.cos(ang) + cour_random*np.sin(ang),  # position of obstacle
                     initial_s[1]+alt_random,
                     dis_obs * np.sin(ang) + cour_random*np.cos(ang)]

            if distance.euclidean(initial_s[0:3], pos_i) > self.ss_r:
                break
        vel_i = [-speed_obs * np.cos(ang), 0, -speed_obs * np.sin(ang)]   # velocity of obstacle
        initial_s, pos_i, vel_i = self.cruise_numerical(initial_s, pos_i, vel_i)
        initial_s[5] = np.random.uniform(-10, 10)

        initial_s[0:3] = np.array(self.coor_trans(initial_s[0:3]))
        init_yaw = initial_s[5] + 360 if initial_s[5] < 0 else initial_s[5]
        pos = [initial_s[0], initial_s[1], initial_s[2], initial_s[4], initial_s[3], init_yaw, -998]
        vel = self.as2vel(initial_s[6], initial_s[3], initial_s[4], initial_s[5])
        self.client.sendDREFs(['sim/time/zulu_time_sec'], [36000, ])  # daytime
        self.client.sendCOMM('sim/operation/reset_flight')  # reset plane
        self.client.sendCOMM('sim/view/circle')  # set the external view
        self.client.sendPOSI(pos)  # set position
        self.client.sendDREFs(self.para.vel, vel + list(initial_s[7:10]))
        self.client.sendCOMM('sim/instruments/DG_sync_mag')  # vacuum DG sync to magnetic north.
        rdata = self.get_rdata() + pos_i + vel_i
        rdata = self.cruise(rdata)
        dis = distance.euclidean(rdata[0:3], pos_i)
        in_ssr = dis < self.ss_r and self.exist_obs
        state = self.get_state(rdata, in_ssr=in_ssr)
        return list(rdata), np.array(state), self.R  # list, np

    def step(self, action, rdata, tau=1.):
        pos_i, vel_i = rdata[13:16], rdata[16:19]
        act = self.s_antinorm(action, self.action_space)
        act = [act[0], act[2], -act[1]]
        target_fp, target_course_ang = self.tra_angle(act[0], act[1], act[2])
        target_AS = np.linalg.norm(act)
        target = [target_fp, target_course_ang, 0, target_AS]  # roll，flight path angle，course angle，sideslip angle，airspeed.
        kp = [0.03, 0.03, 0.1, 0.03, 0.1]
        ki = [5e-5, 1e-4, 1e-4, 1e-4, 1e-4]
        kd = [1e-2, 5e-3, 0.0, 0.0, 0.0]
        step_num = int(tau / 0.0135)
        controller = pid.xplanePID(target, kp, ki, kd)
        for i in range(step_num):
            cur_rdata = self.get_rdata()
            fp_ang, course_ang = self.tra_angle(cur_rdata[7], cur_rdata[8], cur_rdata[9])
            alpha, beta = self.vel_angle(cur_rdata[7], cur_rdata[8], cur_rdata[9],  # angle of attack, sideslip angle
                                         cur_rdata[3], cur_rdata[4], cur_rdata[5])
            state_f = [cur_rdata[3], fp_ang, course_ang, beta, cur_rdata[6]]

            values = controller.cal_values(state_f)
            self.client.sendCTRL(values)
        next_vel_i = vel_i
        next_pos_i = pos_i[:]
        next_pos_i[0] += tau * vel_i[0]
        next_pos_i[1] += tau * vel_i[1]
        next_pos_i[2] += tau * vel_i[2]
        next_rdata = self.get_rdata() + next_pos_i + next_vel_i
        dis = distance.euclidean(next_rdata[0:3], next_pos_i)
        in_ssr = dis < self.ss_r and self.exist_obs
        next_state = self.get_state(next_rdata, in_ssr=in_ssr)
        rew = self.reward(action, next_rdata, rdata)
        done, crash, warn = self.is_done()
        info = [warn, crash]
        return next_rdata, np.array(next_state), rew, done, info, act

    def cruise_numerical(self, init_s, init_pos_i, init_vel_i, tau=0.1):
        '''
        The numerical simulation of cruising, until the obstacle flies into a range(sensing range + random(50, 150))
        '''
        print("cruising......")
        pos_i, vel_i = init_pos_i, init_vel_i
        pos = init_s[0:3]
        vel = self.as2vel(init_s[6], init_s[3], init_s[4], init_s[5])  # vx,vy,vz
        start_dis = np.random.uniform(50, 150)
        for ii in range(1000):
            pos_i[0] += tau * vel_i[0]
            pos_i[1] += tau * vel_i[1]
            pos_i[2] += tau * vel_i[2]
            pos[0] += tau * vel[0]
            pos[1] += tau * vel[1]
            pos[2] += tau * vel[2]
            dis = distance.euclidean(pos, pos_i)
            if dis < self.ss_r + start_dis:
                s = np.append(pos, init_s[3:])
                return s, pos_i, vel_i  # np,list,list

    def cruise(self, rdata, tau=0.1):
        '''
        cruising, until the obstacle fly into the sensing range
        '''
        alt_c = pid2.PID_posi(0.1, 0, 0.001, 2000, up=20, low=-20)  # altitude
        tar_yaw = rdata[5]
        target = [0, 0, 0, 51]
        kp = [0.03, 0.03, 0.1, 0.03, 0.1]
        ki = [5e-5, 1e-4, 1e-4, 1e-4, 1e-4]
        kd = [1e-2, 5e-3, 0.0, 0.0, 0.0]
        step_num = int(tau / 0.0135)
        controller = pid2.xplanePID(target, kp, ki, kd)
        for ii in range(1000):
            pos_i, vel_i = rdata[13:16], rdata[16:19]
            for i in range(step_num):
                cur_rdata = self.get_rdata()
                tar_pitch = alt_c.increase(cur_rdata[1])
                target = [tar_pitch, tar_yaw, 0, 51]
                course_ang = self.d_rpy(cur_rdata[5], 0)
                alpha, beta = self.vel_angle(cur_rdata[7], cur_rdata[8], cur_rdata[9],
                                             cur_rdata[3], cur_rdata[4], cur_rdata[5])
                state_f = [cur_rdata[3], cur_rdata[4], course_ang, beta, cur_rdata[6]]

                values = controller.cal_values(state_f, target=target)
                values[3] = 1
                self.client.sendCTRL(values)
            next_vel_i = vel_i
            next_pos_i = pos_i[:]
            next_pos_i[0] += tau * vel_i[0]
            next_pos_i[1] += tau * vel_i[1]
            next_pos_i[2] += tau * vel_i[2]
            next_rdata = self.get_rdata() + next_pos_i + next_vel_i
            dis = distance.euclidean(next_rdata[0:3], next_rdata[13:16])
            if dis < self.ss_r:
                return next_rdata
            rdata = next_rdata

    def get_rdata(self):
        '''
        Gain the raw data from X-plane software, see in
        https://github.com/nasa/XPlaneConnect/wiki/Network-Information
        '''
        posi_tuple = self.client.getDREFs(self.para.pos)  # [(),()]
        as_tuple = self.client.getDREFs(self.para.true_as)  # [(),()]
        v_tuple = self.client.getDREFs(self.para.vel)
        S = posi_tuple + as_tuple + v_tuple
        rdata = [s[0] for s in S]

        return rdata

    def reward(self, action, next_rdata, rdata):
        rew = 0
        pos_r = np.array(rdata[13:16]) - np.array(rdata[0:3])
        vel_r = np.array(rdata[16:19]) - np.array(rdata[7:10])
        alt = rdata[1]
        AS = rdata[6]
        cour_err = rdata[0]
        los_angle = self.LOS_angle(-pos_r, -vel_r)
        min_los_angle = self.min_LOS_angle(-pos_r, self.R)
        dis = distance.euclidean(rdata[0:3], rdata[13:16])

        rew_cour = -(0.002 * abs(cour_err)) ** 2
        tar_action = self.s_norm([0, 51, 0], self.action_space)
        rew_action = -(distance.euclidean(action, tar_action) / 2.329) ** 2
        rew_alt = -(0.002 * abs(alt - self.tar_alt)) ** 2
        rew_v = -(abs(AS - 51) / 20) ** 2
        rew += 1.5 * rew_alt + 0.001 * rew_action + rew_cour + 0.5 * rew_v + 0.15
        if self.exist_obs:
            if alt < 100:
                rew -= 1000
                self.warn = True
                self.crash = True
                self.done = True
            elif dis < self.R:
                rew -= 1000
                self.warn = True
                self.crash = True
                self.done = True
            elif los_angle < min_los_angle and dis < self.ss_r:
                rew -= 0.2
        else:
            if alt < 100:
                rew -= 1000
                self.warn = True
                self.crash = True
                self.done = True
        return rew

    def is_done(self):
        return self.done, self.crash, self.warn

    def space_sample(self, spac):
        high, low = spac.high, spac.low
        return np.random.uniform(low, high)

    def coor2angle(self, co):
        '''
        Calculate the argument of a vector
        '''
        cos_ = co[0] / np.linalg.norm(co)
        sin_ = co[1] / np.linalg.norm(co)
        angle = np.arccos(cos_)
        if sin_ < 0:
            angle = -angle
        angle_ = angle * 180 / np.pi
        return angle, angle_  # rad, deg

    def as2vel(self, airspeed, roll, pitch, yaw, z_unit=np.array([0, 0, -1])):
        """
        Airspeed and attitude to velocity vector, if the angle of attack and the sideslip angle is zero.
        z_unit=np.array([0, 0, -1]) conrresponding to yaw is zero.
        """
        roll = roll / 180 * np.pi
        pitch = pitch / 180 * np.pi
        yaw = yaw / 180 * np.pi
        Rr = np.array([[np.cos(roll), np.sin(roll), 0],
                       [-np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])
        Rp = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        Rh = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                       [0, 1, 0],
                       [np.sin(yaw), 0, np.cos(yaw)]])

        UnitVec = np.einsum('ij,jk,kl,l->i', Rh, Rp, Rr, z_unit)
        vel = [airspeed * UnitVec[0], airspeed * UnitVec[1], airspeed * UnitVec[2]]

        return vel  # list

    def s_norm(self, s, space, upper=1., clip=False):
        '''
        normalization
        '''
        s = np.array(s)
        low = space.low
        high = space.high
        s_nor = (2 * (s - low) / (high - low) - 1) * upper
        if clip:
            s_nor = s_nor.clip(-1, 1)

        return list(s_nor)

    def s_antinorm(self, s_nor, space, upper=1.):
        '''
        Inverse normalization
        '''
        s_nor = np.array(s_nor) / upper
        low = space.low
        high = space.high
        s = (s_nor + 1) * (high - low) / 2 + low
        return list(s)

    def coor_trans(self, coor):
        '''
        coordination of world frame to longitude, latitude and altitude
        '''
        coor_geo = [-0.49998563917643774, 0., 0.]
        R = 6371393  # radius of the Earth
        x, y, z = coor[0], coor[1], coor[2]
        coor_geo[0] -= (z / R) * 180 / np.pi
        coor_geo[1] += (x / R) * 180 / np.pi
        coor_geo[2] += y

        return coor_geo  # list

    def World2Body(self, u_world, roll, pitch, yaw):
        '''
        world frame to body frame
        '''
        roll = roll / 180 * np.pi
        pitch = pitch / 180 * np.pi
        yaw = yaw / 180 * np.pi
        Rr = np.array([[np.cos(roll), np.sin(roll), 0],
                       [-np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])
        Rp = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        Rh = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                       [0, 1, 0],
                       [np.sin(yaw), 0, np.cos(yaw)]])
        u_world = np.array(u_world)
        u_body = np.einsum('ij,jk,kl,l->i', Rr.T, Rp.T, Rh.T, u_world)

        return u_body

    def vel_angle(self, vx, vy, vz, roll, pitch, yaw):
        '''
        calculate angle of attack(alpha) and sideslip angle(beta)
        '''
        V_body = self.World2Body([vx, vy, vz], roll, pitch, yaw)
        alpha = np.arctan(-V_body[1] / V_body[2]) * 180 / np.pi
        beta = np.arcsin(V_body[0] / np.linalg.norm(V_body)) * 180 / np.pi

        return alpha, beta

    def tra_angle(self, vx, vy, vz):
        '''
        calculate course angle and flight path angle
        '''
        v_xz = np.linalg.norm([vx, vz])
        flight_path_angle = np.arctan(vy / v_xz) * 180 / np.pi
        _, course_angle = self.coor2angle([-vz, vx])

        return flight_path_angle, course_angle

    def d_rpy(self, target_rpy, rpy):
        '''
        angle subtract
        '''
        delta = target_rpy - rpy
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        return delta

    def LOS_angle(self, d_pos, d_vel):
        '''
        line of sight angle: The angle between the velocity of the aircraft related to the obstacle,
        and the line from the aircraft to the center of the protected zone
        '''
        dis = np.linalg.norm(d_pos)
        cos_ = np.dot(d_vel, -d_pos) / (dis * np.linalg.norm(d_vel))
        los_angle = np.arccos(cos_)   # rad

        return los_angle * 180/np.pi  # deg

    def min_LOS_angle(self, d_pos, R):
        '''
        min feasible line of sight angle: The angle between the tangent from the aircraft to the protected zone,
        and the line from the aircraft to the center of the protected zone
        '''
        dis = np.linalg.norm(d_pos)
        min_los_angle = np.arcsin(R / dis) if R < dis else np.pi / 2  # rad
        return min_los_angle * 180 / np.pi  # deg
