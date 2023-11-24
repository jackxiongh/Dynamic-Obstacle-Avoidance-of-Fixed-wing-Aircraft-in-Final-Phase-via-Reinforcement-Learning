import numpy as np
import scipy.spatial.distance as distance

class VO_navi:
    def __init__(self, model_pre, R_pz=150, ss_r=1000):
        self.maintain_vel = None
        self.vo_set = None
        self.R_pz = R_pz
        self.ss_r = ss_r
        self.model_pre = model_pre

        self.tar_AS = 51

    def cal_action(self, action_space, rdata, state, tau=0.1):
        pos_owner = rdata[0:3]
        pos_obs = rdata[13:16]
        vel_owner = rdata[7:10]
        vel_obs = rdata[16:19]
        if distance.euclidean(pos_owner, pos_obs) < self.ss_r:
            self.vo_set = VO_set(self.R_pz, pos_owner, pos_obs, vel_owner, vel_obs)
            Vavo, fp_angle_d, course_angle_d, AS_d, info = self.vo_set.cal_Vavo(tau=tau)
            if not info[0]:    # maintain mode
                if self.maintain_vel is None:
                    self.maintain_vel = vel_owner[:]
                Vavo = self.maintain_vel[:]
            else:   # avoid mode
                self.maintain_vel = None
            Vavo = [Vavo[0], -Vavo[2], Vavo[1]]
            action = self.s_norm(Vavo, action_space)
        else:   # mission mode, use the pretrained model
            info = [False, -999, -999, -999, -999, -999, -999]
            action, entropy = self.model_pre.select_action(state, deterministic=True, with_logprob=True)

        return action, info

    def s_norm(self, s, space, upper=1., clip=False):
        s = np.array(s)
        low = space.low
        high = space.high
        s_nor = (2 * (s - low) / (high - low) - 1) * upper
        if clip:
            s_nor = s_nor.clip(-1, 1)
        return list(s_nor)

class VO_set:
    '''
    The reference paper: Three-Dimensional Velocity Obstacle Method for Uncoordinated Avoidance Maneuvers of Unmanned Aerial Vehicles
    '''
    def __init__(self, R_pz, pos_owner, pos_obs, vel_owner, vel_obs):
        '''
        :param R_pz: radius of protected zone
        :param pos_owner: position of the aircraft in world frame of X-plane
        :param pos_obs: position of the obstacle in world frame of X-plane
        :param vel_owner: velocity of the aircraft in world frame of X-plane
        :param vel_obs: velocity of the obstacle in world frame of X-plane
        '''
        self.R_pz = R_pz
        self.vel_owner = np.array(vel_owner)
        self.AS = np.linalg.norm(self.vel_owner)   # airspeed
        self.fp_angle, self.course_angle = self.tra_angle(vel_owner[0], vel_owner[1], vel_owner[2])
        self.vel_obs = np.array(vel_obs)

        '''
        The values of the obstacle relative to the aircraft, all expressed in velocity coordinate frame.
        '''
        self.origin = np.array([0, 0, 0])  # origin of velocity coordinate frame
        self.vel_obs_r = self.World2Vel(self.fp_angle, self.course_angle, self.vel_obs - self.vel_owner)
        self.pos_obs_r = self.World2Vel(self.fp_angle, self.course_angle, np.array(pos_obs) - np.array(pos_owner))
        self.dis = np.linalg.norm(self.pos_obs_r)   # Distance between obstacles and aircraft
        self.apex = self.origin + self.World2Vel(self.fp_angle, self.course_angle, self.vel_obs)  # apex of VO set

        self.min_los_angle = self.min_LOS_angle(-self.pos_obs_r, self.R_pz)

    def cal_Vavo(self, rdata=None, cruise=False, tau=0.1):
        '''
        Calculate avoidance velocity
        '''
        plane_angle, evo, theta = -999, -999, -999
        if rdata is not None:
            pos_owner = rdata[0:3]
            pos_obs = rdata[6:9]
            vel_owner = rdata[3:6]
            vel_obs = rdata[9:12]
            R_pz = self.R_pz
            self.__init__(R_pz, pos_owner, pos_obs, vel_owner, vel_obs)
        in_VO = self.is_in_VO(self.pos_obs_r, self.vel_obs_r)
        los_angle, min_los_angle, del_LOS_ang = self.d_los(self.pos_obs_r, self.vel_obs_r)
        if self.dis < self.R_pz:   # For performance test only
            cruise = True
        if in_VO and not cruise:
            plane_angle, Evo, evo = self.select_plane()
            omiga_acr, omiga_avo = self.cal_omiga_avo(plane_angle)
            if abs(omiga_avo*tau) < abs(evo):
                theta_rad = np.sign(evo)*omiga_avo*tau*np.pi/180
                Vavo_p_xz = [self.AS*np.sin(theta_rad), -self.AS*np.cos(theta_rad)]  # Vavo_p in xoz plane of avoidance-plane frame
                Vavo_p = np.array([Vavo_p_xz[0], 0, Vavo_p_xz[1]])
                Vavo = self.Plane2World(self.fp_angle, self.course_angle, plane_angle, Vavo_p)  # Vavo_p in world frame
                theta = theta_rad * 180/np.pi
            else:
                theta = evo
                Vavo = self.Vel2World(self.fp_angle, self.course_angle, Evo)   # Evo in world frame
            fp_angle_d, course_angle_d = self.tra_angle(Vavo[0], Vavo[1], Vavo[2])  # desired flight path angle, course_angle
            AS_d = self.AS  # desired airspeed
        else:
            Vavo = self.vel_owner
            fp_angle_d, course_angle_d, AS_d = self.fp_angle, self.course_angle, self.AS
        info = [in_VO, los_angle, min_los_angle, del_LOS_ang, plane_angle, evo, theta]
        return Vavo, fp_angle_d, course_angle_d, AS_d, info

    def select_plane(self, plane_num=12):
        '''
        select the best avoidance plane
        zOx plane is zero degree plane
        '''
        delta = 180 / plane_num
        plane_ang = -90
        min_evo = float('inf')
        selected_plane_ang, selected_Evo = None, None
        dropped_plane_ang = []
        # 1. Abandoning Triangle and Hyperbolic cross Sections
        for i in range(plane_num):
            plane_ang += delta
            pos_obs_r_p = self.Vel2Plane(plane_ang, self.pos_obs_r)
            if np.arcsin(abs(pos_obs_r_p[1])/np.linalg.norm(pos_obs_r_p))*180/np.pi < self.min_los_angle:
                # Triangle or Hyperbolic cross Sections
                dropped_plane_ang.append(plane_ang)
                continue
            Evo, evo = self.seek_Evo(plane_ang)
            if abs(evo) < abs(min_evo):
                min_evo = evo
                selected_Evo = Evo
                selected_plane_ang = plane_ang
        # 2. if can not abandon Triangle and Hyperbolic cross Sections
        if selected_plane_ang is None:
            for plane_ang in dropped_plane_ang:
                Evo, evo = self.seek_Evo(plane_ang)
                if abs(evo) < abs(min_evo):
                    min_evo = evo
                    selected_Evo = Evo
                    selected_plane_ang = plane_ang
        return selected_plane_ang, selected_Evo, min_evo

    def seek_Evo(self, plane_angle):
        '''
        The velocity vector is considered to be within VO
        return the Evo and the angle evo of an avoidance plane
        '''
        delta, evo = 10, 0
        min_evo = 0
        # right
        while True:
            evo += delta
            if evo > 180:
                break
            min_evo = evo
            Evo1 = self.AS * np.array([np.sin(evo*np.pi/180), 0, -np.cos(evo*np.pi/180)])  # Evo in avoidance-plane frame
            Evo1 = self.Plane2Vel(plane_angle, Evo1)  # Evo in velocity frame
            if not self.is_in_VO(self.pos_obs_r, self.World2Vel(self.fp_angle, self.course_angle, self.vel_obs)-Evo1):
                if delta < 0.1:
                    break
                evo -= delta
                delta /= 2
        # left
        delta, evo = 10, 0
        while True:
            evo -= delta
            if evo < -180:
                break
            Evo2 = self.AS*np.array([np.sin(evo*np.pi/180), 0, -np.cos(evo*np.pi/180)])
            Evo2 = self.Plane2Vel(plane_angle, Evo2)
            if not self.is_in_VO(self.pos_obs_r, self.World2Vel(self.fp_angle, self.course_angle, self.vel_obs)-Evo2):
                if delta < 0.1:
                    break
                evo += delta
                delta /= 2
        if -evo < min_evo:
            return Evo2, evo
        else:
            return Evo1, min_evo

    def cal_omiga_avo(self, plane_angle, buffer_ratio=0.1):
        '''
        The omiga_avo in a avoidance plane
        '''
        pos_obs_r_p = self.Vel2Plane(plane_angle, self.pos_obs_r)  # The position of the obstacle relative to the aircraft in avoidance-plane frame
        vel_obs_p = self.World2Plane(self.fp_angle, self.course_angle, plane_angle, self.vel_obs)  # The velocity of the obstacle in avoidance-plane frame
        vel_owner_p = np.array([0, 0, -self.AS])

        t0 = np.linalg.norm(self.pos_obs_r) / np.linalg.norm(self.vel_obs_r)   # Newton Iteration start point
        omiga_avo_rad0 = np.pi / (4*t0)

        t_right, omiga_avo_rad_right = self.Newton_iter(t0, omiga_avo_rad0, pos_obs_r_p, vel_obs_p)
        t_left, omiga_avo_rad_left = self.Newton_iter(t0, -omiga_avo_rad0, pos_obs_r_p, vel_obs_p)

        if abs(omiga_avo_rad_right) < abs(omiga_avo_rad_left):
            omiga_acr = abs(omiga_avo_rad_left) * 180/np.pi
        else:
            omiga_acr = abs(omiga_avo_rad_right) * 180/np.pi

        omiga_avo = (1+buffer_ratio)*omiga_acr

        return omiga_acr, omiga_avo

    def Newton_iter(self, t0, omiga_avo_rad0, pos_obs_r_p, vel_obs_p, C=1, e=0.01):
        '''
        Newton Iteration to solve an equation system {check_dis=0,partial derivative of check_dis() over t=0}
        :param t0: Newton Iteration start point
        :param omiga_avo_rad0: Newton Iteration start point
        :param C: error control constant
        :param e: allowable error
        '''
        x0 = np.array([t0, omiga_avo_rad0])
        for ii in range(1000):
            result, Jacobian = self.check_dis(x0[0], x0[1], pos_obs_r_p, vel_obs_p)
            y0 = np.array([result, Jacobian[0][0]])
            x1 = x0 - np.dot(inverse_2D(Jacobian), y0)
            if np.linalg.norm(x1) < C:
                sigma = np.linalg.norm(x1 - x0)
            else:
                sigma = np.linalg.norm(x1 - x0)/np.linalg.norm(x1)
            if np.linalg.norm(sigma) < e:
                return x1[0], x1[1]
            x0 = x1

        return x1[0], x1[1]

    def check_dis(self, t, omiga_avo_rad, pos_obs_r_p, vel_obs_p):
        '''
        function: check_dis(t, omiga_avo_rad), In an avoidance plane, the distance from the aircraft to the obstacle subtracts the radius of the protection zone
        parameter: pos_obs_r_p, vel_obs_p, R_pz
        return: function values and a Jacobian Matrix
        '''
        if omiga_avo_rad == 0:
            omiga_avo_rad += 1e-4
        omiga_avo = omiga_avo_rad
        # function values：
        dis_sqrx = (vel_obs_p[0] * t + pos_obs_r_p[0] + (self.AS / omiga_avo) * (np.cos(omiga_avo * t) - 1)) ** 2
        dis_sqry = (pos_obs_r_p[1] + vel_obs_p[1] * t) ** 2
        dis_sqrz = (vel_obs_p[2] * t + pos_obs_r_p[2] + (self.AS / omiga_avo) * np.sin(omiga_avo * t)) ** 2
        dis_sqr = dis_sqrx + dis_sqry + dis_sqrz
        dis = np.sqrt(dis_sqr)

        result = dis - self.R_pz

        # Partial derivative
        dt_dis_sqrx = 2 * (vel_obs_p[0] * t + pos_obs_r_p[0] +
                           (self.AS / omiga_avo) * (np.cos(omiga_avo * t) - 1)) * (
                                  vel_obs_p[0] - self.AS * np.sin(omiga_avo * t))
        dt_dis_sqry = 2 * (pos_obs_r_p[1] + vel_obs_p[1] * t) * vel_obs_p[1]
        dt_dis_sqrz = 2 * (vel_obs_p[2] * t + pos_obs_r_p[2] +
                           (self.AS / omiga_avo) * np.sin(omiga_avo * t)) * (
                                  vel_obs_p[2] + self.AS * np.cos(omiga_avo * t))
        dt_dis_sqr = dt_dis_sqrx + dt_dis_sqry + dt_dis_sqrz
        dt_dis = dt_dis_sqr / (2 * dis)
        dt_result = dt_dis  # Partial derivative of check_dis() over t

        dw_dis_sqrx = 2 * (vel_obs_p[0] * t + pos_obs_r_p[0] + (self.AS / omiga_avo) * (np.cos(omiga_avo * t) - 1)) * (
                (self.AS / omiga_avo ** 2) * (1 - np.cos(omiga_avo * t)) - (self.AS * t / omiga_avo) * np.sin(
            omiga_avo * t))
        dw_dis_sqry = 0
        dw_dis_sqrz = 2 * (vel_obs_p[2] * t + pos_obs_r_p[2] + (self.AS / omiga_avo) * np.sin(omiga_avo * t)) * (
                (self.AS * t / omiga_avo) * np.cos(omiga_avo * t) - (self.AS / omiga_avo ** 2) * np.sin(omiga_avo * t))
        dw_dis_sqr = dw_dis_sqrx + dw_dis_sqry + dw_dis_sqrz
        dw_dis = dw_dis_sqr / (2 * dis)
        dw_result = dw_dis

        # Second-order partial derivative
        ddt_dis_sqrx = (vel_obs_p[0] - self.AS * np.sin(omiga_avo * t)) ** 2 - \
                       (vel_obs_p[0] * t + pos_obs_r_p[0] + (self.AS / omiga_avo) * (np.cos(omiga_avo * t) - 1)) * \
                       (self.AS * omiga_avo * np.cos(omiga_avo * t))
        ddt_dis_sqrx *= 2
        ddt_dis_sqry = 2 * vel_obs_p[1] ** 2
        ddt_dis_sqrz = (vel_obs_p[2] + self.AS * np.cos(omiga_avo * t)) ** 2 - \
                       (vel_obs_p[2] * t + pos_obs_r_p[2] + (self.AS / omiga_avo) * np.sin(omiga_avo * t)) * \
                       (self.AS * omiga_avo * np.sin(omiga_avo * t))
        ddt_dis_sqrz *= 2
        ddt_dis_sqr = ddt_dis_sqrx + ddt_dis_sqry + ddt_dis_sqrz
        ddt_dis = -dt_dis ** 2 / (4 * dis ** 3) + ddt_dis_sqr / (2 * dis)
        ddt_result = ddt_dis

        dtdw_dis_sqrx = ((self.AS / omiga_avo ** 2) * (1 - np.cos(omiga_avo * t)) - (self.AS * t / omiga_avo) * np.sin(
            omiga_avo * t)) * \
                        (vel_obs_p[0] - self.AS * np.sin(omiga_avo * t)) - \
                        (vel_obs_p[0] * t + pos_obs_r_p[0] + (self.AS / omiga_avo) * (np.cos(omiga_avo * t) - 1)) * \
                        (self.AS * t * np.cos(omiga_avo * t))
        dtdw_dis_sqrx *= 2
        dtdw_dis_sqry = 0
        dtdw_dis_sqrz = ((self.AS * t / omiga_avo) * np.cos(omiga_avo * t) - (self.AS / omiga_avo ** 2) * np.sin(
            omiga_avo * t)) * \
                        (vel_obs_p[2] + self.AS * np.cos(omiga_avo * t)) - \
                        (vel_obs_p[2] * t + pos_obs_r_p[2] + (self.AS / omiga_avo) * np.sin(omiga_avo * t)) * \
                        (self.AS * t * np.sin(omiga_avo * t))
        dtdw_dis_sqrz *= 2
        dtdw_dis_sqr = dtdw_dis_sqrx + dtdw_dis_sqry + dtdw_dis_sqrz
        dtdw_dis = -dw_dis * dt_dis_sqr / (2 * dis ** 2) + dtdw_dis_sqr / (2 * dis)

        dtdw_result = dtdw_dis

        # Jacobian matrix of equation system {check_dis=0,partial derivative of check_dis() over t=0}
        Jacobian = np.array([[dt_result, dw_result],
                             [ddt_result, dtdw_result]])

        return result, Jacobian

    def LOS_angle(self, d_pos, d_vel):
        '''
        line of sight angle: The angle between the velocity of the aircraft related to the obstacle,
        and the line from the aircraft to the center of the protected zone
        '''
        dis = np.linalg.norm(d_pos)
        cos_ = np.dot(d_vel, -d_pos) / (dis * np.linalg.norm(d_vel))
        los_angle = np.arccos(cos_)   # rad

        return los_angle * 180/np.pi

    def min_LOS_angle(self, d_pos, R):
        '''
        min feasible line of sight angle: The angle between the tangent from the aircraft to the protected zone,
        and the line from the aircraft to the center of the protected zone
        '''
        dis = np.linalg.norm(d_pos)
        min_los_angle = np.arcsin(R / dis) if R < dis else np.pi / 2  # rad
        return min_los_angle * 180 / np.pi

    def is_in_VO(self, pos_obs_r, vel_obs_r):
        '''
        is the velocity vector of the aircraft in Vo set
        '''
        if self.dis < self.R_pz:
            return True
        los_angle = self.LOS_angle(-pos_obs_r, -vel_obs_r)
        min_los_angle = self.min_LOS_angle(-pos_obs_r, self.R_pz)
        if los_angle < min_los_angle:
            return True
        else:
            return False

    def d_los(self, pos_obs_r, vel_obs_r):
        '''
        return: line of sight angle, min feasible line of sight angle
        line of sight angle subtracts min feasible line of sight angle
        '''
        los_angle = self.LOS_angle(-pos_obs_r, -vel_obs_r)
        min_los_angle = self.min_LOS_angle(-pos_obs_r, self.R_pz)
        if self.dis < self.R_pz:
            return los_angle, min_los_angle, -float('inf')
        return los_angle, min_los_angle, los_angle - min_los_angle


    def tra_angle(self, vx, vy, vz):
        '''
        calculate course angle and flight path angle
        '''
        v_xz = np.linalg.norm([vx, vz])
        flight_path_angle = np.arctan(vy / v_xz) * 180 / np.pi
        course_angle = np.arctan(-vx / vz) * 180 / np.pi

        return flight_path_angle, course_angle

    def World2Vel(self, flight_path_angle, course_angle, u=np.array([0, 0, -1])):
        '''
        world frame to velocity frame
        '''
        flight_path_angle = flight_path_angle / 180 * np.pi
        course_angle = course_angle / 180 * np.pi
        plane_angle = 0
        Rp = np.array([[np.cos(plane_angle), np.sin(plane_angle), 0],
                       [-np.sin(plane_angle), np.cos(plane_angle), 0],
                       [0, 0, 1]])
        Rfp = np.array([[1, 0, 0],
                       [0, np.cos(flight_path_angle), -np.sin(flight_path_angle)],
                       [0, np.sin(flight_path_angle), np.cos(flight_path_angle)]])
        Rc = np.array([[np.cos(course_angle), 0, -np.sin(course_angle)],
                       [0, 1, 0],
                       [np.sin(course_angle), 0, np.cos(course_angle)]])

        u_new = np.einsum('ij,jk,kl,l->i', Rp.T, Rfp.T, Rc.T, np.array(u))

        return u_new  # np.array

    def Vel2World(self, flight_path_angle, course_angle, u=np.array([0, 0, -1])):
        '''
        velocity frame to world frame
        '''
        return yxz_trans(flight_path_angle, course_angle, 0, u=u, is_1to2=False)


    def Vel2Plane(self, plane_angle, u):
        """
        velocity frame to avoidance-plane frame
        """
        flight_path_angle = 0
        course_angle = 0
        plane_angle = plane_angle / 180 * np.pi
        Rp = np.array([[np.cos(plane_angle), np.sin(plane_angle), 0],
                       [-np.sin(plane_angle), np.cos(plane_angle), 0],
                       [0, 0, 1]])
        Rfp = np.array([[1, 0, 0],
                       [0, np.cos(flight_path_angle), -np.sin(flight_path_angle)],
                       [0, np.sin(flight_path_angle), np.cos(flight_path_angle)]])
        Rc = np.array([[np.cos(course_angle), 0, -np.sin(course_angle)],
                       [0, 1, 0],
                       [np.sin(course_angle), 0, np.cos(course_angle)]])

        u_new = np.einsum('ij,jk,kl,l->i', Rp.T, Rfp.T, Rc.T, np.array(u))

        return u_new  # np.array

    def Plane2Vel(self, plane_angle, u):
        flight_path_angle = 0
        course_angle = 0
        plane_angle = plane_angle / 180 * np.pi
        Rp = np.array([[np.cos(plane_angle), np.sin(plane_angle), 0],
                       [-np.sin(plane_angle), np.cos(plane_angle), 0],
                       [0, 0, 1]])
        Rfp = np.array([[1, 0, 0],
                       [0, np.cos(flight_path_angle), -np.sin(flight_path_angle)],
                       [0, np.sin(flight_path_angle), np.cos(flight_path_angle)]])
        Rc = np.array([[np.cos(course_angle), 0, -np.sin(course_angle)],
                       [0, 1, 0],
                       [np.sin(course_angle), 0, np.cos(course_angle)]])

        u_new = np.einsum('ij,jk,kl,l->i', Rc, Rfp, Rp, np.array(u))

        return u_new  # np.array

    def World2Plane(self, flight_path_angle, course_angle, plane_angle, u=np.array([0, 0, -1])):
        flight_path_angle = flight_path_angle / 180 * np.pi
        course_angle = course_angle / 180 * np.pi
        plane_angle = plane_angle / 180 * np.pi
        Rp = np.array([[np.cos(plane_angle), np.sin(plane_angle), 0],
                       [-np.sin(plane_angle), np.cos(plane_angle), 0],
                       [0, 0, 1]])
        Rfp = np.array([[1, 0, 0],
                       [0, np.cos(flight_path_angle), -np.sin(flight_path_angle)],
                       [0, np.sin(flight_path_angle), np.cos(flight_path_angle)]])
        Rc = np.array([[np.cos(course_angle), 0, -np.sin(course_angle)],
                       [0, 1, 0],
                       [np.sin(course_angle), 0, np.cos(course_angle)]])

        u_new = np.einsum('ij,jk,kl,l->i', Rp.T, Rfp.T, Rc.T, np.array(u))  # yxz

        return u_new  # np.array

    def Plane2World(self, flight_path_angle, course_angle, plane_angle, u=np.array([0, 0, -1])):
        return yxz_trans(flight_path_angle, course_angle, plane_angle, u=u, is_1to2=False)

    def s_norm(self, s, space, upper=1.):
        s = np.array(s)
        low = space.low
        high = space.high
        s_nor = (2 * (s - low) / (high - low) - 1) * upper

        return list(s_nor)

    def d_rpy(self, target_rpy, rpy):
        delta = target_rpy - rpy
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        return delta

def inverse_2D(U):
    '''
    inverse matrix
    '''
    det_U = np.linalg.det(U)
    result = np.zeros([2, 2])
    result[0][0] = U[1][1]
    result[1][1] = U[0][0]
    result[0][1] = -U[0][1]
    result[1][0] = -U[1][0]

    result /= det_U

    return result

def yxz_trans(x_ang, y_ang, z_ang, u=np.array([0, 0, -1]), is_1to2=True):
    """
    Universal coordinate conversion
    """
    x_ang = x_ang / 180 * np.pi
    y_ang = y_ang / 180 * np.pi
    z_ang = z_ang / 180 * np.pi
    Rz = np.array([[np.cos(z_ang), np.sin(z_ang), 0],
                   [-np.sin(z_ang), np.cos(z_ang), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(x_ang), -np.sin(x_ang)],
                    [0, np.sin(x_ang), np.cos(x_ang)]])
    Ry = np.array([[np.cos(y_ang), 0, -np.sin(y_ang)],
                   [0, 1, 0],
                   [np.sin(y_ang), 0, np.cos(y_ang)]])
    if is_1to2:
        u_new = np.einsum('ij,jk,kl,l->i', Rz.T, Rx.T, Ry.T, np.array(u))  # yxz
    else:
        u_new = np.einsum('ij,jk,kl,l->i', Ry, Rx, Rz, np.array(u))  # yxz

    return u_new  # np.array




