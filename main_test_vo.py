import numpy as np
import torch
import pandas as pd
import time
from SAC import SAC_Agent
from toolbox.VO_3D5 import VO_navi
import argparse
from Env_test_vo import my_env
from toolbox.logger2_3 import logger

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test8_6vo', help='agent name')
parser.add_argument('--Loadname', type=str, default='avoid8_6a1', help='loaded agent name')
parser.add_argument('--model_pre', type=str, default='avoid7_12', help='pretrained agent name')
parser.add_argument('--env_with_dw', type=str2bool, default=False, help='Env with die and win, or not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=1500000, help='which model to load')
parser.add_argument('--ModelIdex_pre', type=int, default=100000, help='which pretrained model to load')

parser.add_argument('--tau', type=float, default=0.1, help='Time consumption of one step.')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=10, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=125000, help='Max training steps')
parser.add_argument('--max_e_steps', type=int, default=2500, help='Max steps in an episode')
parser.add_argument('--save_interval', type=int, default=300, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=20, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
print(opt)


def main():
    # Env config:
    env = my_env(opt.name)
    log = logger(opt.name)
    state_dim = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action, min_action = 1, -1
    max_e_steps = opt.max_e_steps
    steps_per_epoch = opt.max_e_steps
    print('Env:', opt.name, '  state_dim:', state_dim, '  action_dim:', action_dim,
          '  max_a:', max_action, '  min_a:', min_action, 'max_episode_steps', steps_per_epoch)

    #Interaction config:
    total_steps = int(opt.Max_train_steps)

    #Random seed config:
    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    #Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "critic_hid_shape": (opt.net_width, opt.net_width, opt.net_width),
        "actor_hid_shape": (opt.net_width, opt.net_width, opt.net_width),
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "alpha": opt.alpha,
        "adaptive_alpha": opt.adaptive_alpha,
        "l": 2,
        "weight_decay": 0.01
    }

    model_pre = SAC_Agent(**kwargs)
    model_pre.load(opt.model_pre, opt.ModelIdex_pre)

    fp = './reset_ang/obs_ran{}.xlsx'.format(env.ss_r)
    init_list = pd.read_excel(fp)
    rdata, state, R = env.reset(init_list.loc[log.episode, :])
    navigator = VO_navi(model_pre, R_pz=R, ss_r=env.ss_r)
    done, current_steps = False, 0
    for t in range(total_steps):
        t0 = time.time()
        current_steps += 1
        '''Interact & trian'''
        action, info_VO = navigator.cal_action(env.action_space, rdata, state) #a∈[-1,1]
        next_rdata, next_state, r, done, info, act = env.step(action, rdata, tau=opt.tau)
        t1 = time.time() - t0
        log.step_update(r, t1, next_rdata, act, 0)
        done = done or current_steps >= max_e_steps
        print('steps:', current_steps, ' action:', action, 'xyzrpy,as:', next_rdata[0:7])
        rdata = next_rdata
        state = next_state

        if done:
            log.epi_update(info, save=True, chart=False)
            done, current_steps = False, 0
            print('episode:', log.episode, 'totalsteps:', t + 1, ' average reward:', log.cur_mean_rew, 'is_crash:',
                  info)
            rdata, state, R = env.reset(init_list.loc[log.episode, :])
            navigator = VO_navi(model_pre, R_pz=R, ss_r=env.ss_r)
    log.update(chart=False)
if __name__ == '__main__':
    main()






