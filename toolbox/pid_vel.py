import os
import numpy as np
import pandas as pd
import math
from time import sleep
from time import time
import matplotlib.pyplot as plt
import copy

class PID_posi:
    def __init__(self, kp, ki, kd, target, up=1., low=-1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0
        self.err_last = 0
        self.err_all = 0
        self.target = target
        self.up = up
        self.low = low
        self.value = 0

    def increase(self, state, target=None, angle=False):
        if target is None:
            target = copy.deepcopy(self.target)
        if angle:
            self.err = self.d_rpy(target, state)
        else:
            self.err = target - state
        self.value = self.kp*self.err + self.ki * self.err_all + self.kd*(self.err-self.err_last)
        self.update()
        return self.value

    def update(self):
        self.err_last = self.err
        self.err_all = self.err_all + self.err
        if self.value > self.up:
            self.value = self.up
        elif self.value < self.low:
            self.value = self.low

    def set_pid(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0

    def set_target(self, target):
        self.target = target

    def d_rpy(self, target_rpy, rpy):
        delta = target_rpy - rpy
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        return delta

class xplanePID:
    def __init__(self, target, kp, ki, kd):
        self.el = PID_posi(kp[1], ki[1], kd[1], target[0])  # control flight path angle
        self.tar_roll = PID_posi(kp[2], ki[2], kd[2], target[1])  # control course angle
        self.ai = PID_posi(kp[0], ki[0], kd[0], self.tar_roll.value)  # control roll
        self.ru = PID_posi(kp[3], ki[3], kd[3], target[2])  # control sideslip
        self.th = PID_posi(kp[4], ki[4], kd[4], target[3], low=0)  # control airspeed
        self.target = target

    def cal_values(self, state, target=None, roll_max=20.):
        '''
        state：roll，pitch，yaw(or course angle)，sideslip，airspeed
        target：pitch，yaw(or course angle)，sideslip，airspeed
        '''
        if target is None:
            target = copy.deepcopy(self.target)
        elevator = self.el.increase(state[1], target=target[0], angle=True)  # flight path angle
        target_roll = self.tar_roll.increase(state[2], target=target[1], angle=True)   # course
        target_roll *= roll_max   # [-1, 1]——>[-roll_max, roll_max]
        aileron = self.ai.increase(state[0], target=target_roll, angle=True)   # roll
        rudder = -self.ru.increase(state[3], target=target[2], angle=True)  # sideslip
        throttle = self.th.increase(state[4], target=target[3])  # airspeed
        return [elevator, aileron, rudder, throttle]

    def reset(self):
        self.el.reset()
        self.ai.reset()
        self.ru.reset()
        self.th.reset()


def vel_angle(vx, vy, vz, roll, pitch, yaw):
    roll = roll / 180 * np.pi
    pitch = pitch / 180 * np.pi
    heading = yaw / 180 * np.pi
    Rr = np.array([[np.cos(roll), np.sin(roll), 0],
                   [-np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    Rp = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Rh = np.array([[np.cos(heading), 0, -np.sin(heading)],
                   [0, 1, 0],
                   [np.sin(heading), 0, np.cos(heading)]])
    V_world = np.array([vx, vy, vz])
    V_body = np.einsum('ij,jk,kl,l->i', Rr.T, Rp.T, Rh.T, V_world)
    alpha = np.arctan(-V_body[1]/V_body[2])*180/np.pi
    beta = np.arcsin(V_body[0]/np.linalg.norm(V_body))*180/np.pi

    return alpha, beta


def tra_angle(vx, vy, vz):
    v_xz = np.linalg.norm([vx, vz])
    flight_path_angle = np.arctan(vy/v_xz)*180/np.pi
    course_angle = np.arctan(-vx/vz)*180/np.pi

    return flight_path_angle, course_angle

def d_rpy(target_rpy, rpy):
    delta = target_rpy - rpy
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360
    return delta

