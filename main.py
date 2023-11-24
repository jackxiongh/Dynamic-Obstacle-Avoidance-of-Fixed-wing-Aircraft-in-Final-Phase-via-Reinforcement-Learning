import numpy as np
import torch
import time
from SAC import SAC_Agent
from ReplayBuffer import RandomBuffer
import argparse
from Env import my_env
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
parser.add_argument('--name', type=str, default='avoid8_6t', help='agent name')
parser.add_argument('--Loadname', type=str, default='avoid7_12', help='agent name')
parser.add_argument('--env_with_dw', type=str2bool, default=False, help='Env with die and win, or not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100000, help='which model to load')

parser.add_argument('--tau', type=float, default=0.1, help='Time consumption of one step.')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=10, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=1500000, help='Max training steps')
parser.add_argument('--max_e_steps', type=int, default=10, help='Max steps in an episode')
parser.add_argument('--save_interval', type=int, default=2000, help='Model saving interval, in steps.')
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
    env_with_Dead = opt.env_with_dw
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
    start_steps = 10 * steps_per_epoch
    update_after = 1 * opt.batch_size
    update_every = opt.update_every
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


    model = SAC_Agent(**kwargs)
    if opt.Loadmodel:
        model.load(opt.Loadname, opt.ModelIdex)

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_Dead, max_size=int(1e6))

    rdata, state, R = env.reset()
    done, current_steps = False, 0
    for t in range(total_steps):
        t0 = time.time()
        current_steps += 1
        '''Interact & trian'''

        action, entropy = model.select_action(state, deterministic=False, with_logprob=True)  # a∈[-1,1]
        next_rdata, next_state, r, done, info, act = env.step(action, rdata, tau=opt.tau)
        t1 = time.time() - t0
        log.step_update(r, t1, next_rdata, act, entropy)
        done = done or current_steps >= max_e_steps
        print('steps:', current_steps, ' action:', action, 'xyzrpy,as:', next_rdata[0:7])
        replay_buffer.add(state, action, r, next_state, done)
        rdata = next_rdata
        state = next_state

        if t >= update_after and t % update_every == 0:
            env.pause_sim()
            t0 = time.time()
            for j in range(2*update_every):
                model.train(replay_buffer)
            t1 = time.time() - t0
            log.net_time.append(t1)  # record time
            print('successfully update network=====================================================')
            env.pause_sim()

        if done:
            model.save(opt.name, t + 1)
            replay_buffer.save(opt.name)
            log.epi_update(info, save=True, chart=False)

            done, current_steps = False, 0
            print('episode:', log.episode, 'totalsteps:', t+1, ' average reward:', log.cur_mean_rew, 'is_crash:', info)
            rdata, state, R = env.reset()
    model.save(opt.name, t + 1)
    log.update(chart=False)

if __name__ == '__main__':
    main()






