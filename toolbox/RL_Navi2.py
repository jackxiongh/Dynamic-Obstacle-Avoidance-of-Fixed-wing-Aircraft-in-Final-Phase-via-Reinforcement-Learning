import numpy as np
import scipy.spatial.distance as distance

class RL_navi:
    def __init__(self, model, model_pre, R_pz=150, ss_r=1000):
        self.model = model
        self.model_pre = model_pre
        self.ss_r = ss_r
        self.dis = None

    def cal_action(self, rdata, state, action_space, recover=True):
        pos_owner = rdata[0:3]
        pos_obs = rdata[13:16]
        self.dis = distance.euclidean(pos_owner, pos_obs)
        if self.dis < self.ss_r or not recover:
            action, entropy = self.model.select_action(state, deterministic=True, with_logprob=True) #a∈[-1,1]
        else:   # use the pretrained model
            action, entropy = self.model_pre.select_action(state, deterministic=True, with_logprob=True) #a∈[-1,1]

        return action, self.dis

    def s_norm(self, s, space, upper=1., clip=False):
        s = np.array(s)
        low = space.low
        high = space.high
        s_nor = (2 * (s - low) / (high - low) - 1) * upper
        if clip:
            s_nor = s_nor.clip(-1, 1)

        return list(s_nor)