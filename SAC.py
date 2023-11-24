import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from ReplayBuffer import device


def build_net(layer_shape, activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
		super(Actor, self).__init__()

		layers = [state_dim] + list(hid_shape)
		self.a_net = build_net(layers, h_acti, o_acti)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20


	def forward(self, state, deterministic=False, with_logprob=True):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
		std = torch.exp(log_std)
		dist = Normal(mu, std)

		if deterministic: u = mu
		else: u = dist.rsample() #'''reparameterization trick of Gaussian'''#
		a = torch.tanh(u)

		if with_logprob:
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a



class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		return q1



class SAC_Agent(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		gamma=0.99,
		critic_hid_shape=(256, 256),
		actor_hid_shape=(256, 256),
		a_lr=3e-4,
		c_lr=3e-4,
		batch_size = 256,
		alpha = 0.2,
		adaptive_alpha = True,
		l=2,
		weight_decay=0.01
	):

		self.actor = Actor(state_dim, action_dim, actor_hid_shape).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

		self.q_critic1 = Q_Critic(state_dim, action_dim, critic_hid_shape).to(device)
		self.q_critic2 = Q_Critic(state_dim, action_dim, critic_hid_shape).to(device)
		self.q_critic1_optimizer = torch.optim.Adam(self.q_critic1.parameters(), lr=c_lr)
		self.q_critic2_optimizer = torch.optim.Adam(self.q_critic2.parameters(), lr=c_lr)
		self.q_critic1_target = copy.deepcopy(self.q_critic1)
		self.q_critic2_target = copy.deepcopy(self.q_critic2)
		for p in self.q_critic1_target.parameters():
			p.requires_grad = False
		for p in self.q_critic2_target.parameters():
			p.requires_grad = False

		self.action_dim = action_dim
		self.gamma = gamma
		self.tau = 0.005
		self.batch_size = batch_size
		self.l = l
		self.weight_decay = weight_decay

		self.alpha = alpha
		self.adaptive_alpha = adaptive_alpha
		if adaptive_alpha:
			self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
			self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)



	def select_action(self, state, deterministic, with_logprob=False):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a, logp_pi_a = self.actor(state, deterministic, with_logprob)
		return a.cpu().numpy().flatten(), logp_pi_a.cpu().numpy().flatten()[0]



	def train(self,replay_buffer):
		s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_prime, log_pi_a_prime = self.actor(s_prime)
			target_Q1, target_Q2 = self.q_critic1_target(s_prime, a_prime), self.q_critic2_target(s_prime, a_prime)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (1 - dead_mask) * self.gamma * (target_Q - self.alpha * log_pi_a_prime) #Dead or Done is tackled by Randombuffer

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic1(s, a), self.q_critic2(s, a)
		q_reg_loss1 = Regularization(self.q_critic1, weight_decay=self.weight_decay, p=self.l).to(device)
		q_reg_loss2 = Regularization(self.q_critic2, weight_decay=self.weight_decay, p=self.l).to(device)

		q_loss1, q_loss2 = F.mse_loss(current_Q1, target_Q) + q_reg_loss1(self.q_critic1), \
						   F.mse_loss(current_Q2, target_Q) + q_reg_loss2(self.q_critic2)
		self.q_critic1_optimizer.zero_grad()
		q_loss1.backward(retain_graph=True)
		self.q_critic1_optimizer.step()
		self.q_critic2_optimizer.zero_grad()
		q_loss2.backward(retain_graph=True)
		self.q_critic2_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze Q-networks so you don't waste computational effort
		# computing gradients for them during the policy learning step.
		for params in self.q_critic1.parameters():
			params.requires_grad = False
		for params in self.q_critic2.parameters():
			params.requires_grad = False

		a, log_pi_a = self.actor(s)
		current_Q1, current_Q2 = self.q_critic1(s, a), self.q_critic2(s, a)
		Q = torch.min(current_Q1, current_Q2)
		a_reg_loss = Regularization(self.actor, weight_decay=self.weight_decay, p=self.l).to(device)

		a_loss = (self.alpha * log_pi_a - Q).mean() + a_reg_loss(self.actor)
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic1.parameters():
			params.requires_grad = True
		for params in self.q_critic2.parameters():
			params.requires_grad = True
		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
			# if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic1.parameters(), self.q_critic1_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		for param, target_param in zip(self.q_critic2.parameters(), self.q_critic2_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



	def save(self,EnvName, episode):
		torch.save(self.actor.state_dict(), "./model/{}/{}_actor{}.pth".format(EnvName, EnvName, episode))
		torch.save(self.q_critic1.state_dict(), "./model/{}/{}_q_critic1{}.pth".format(EnvName, EnvName, episode))
		torch.save(self.q_critic2.state_dict(), "./model/{}/{}_q_critic2{}.pth".format(EnvName, EnvName, episode))


	def load(self, EnvName, episode):
		self.actor.load_state_dict(torch.load("./model/{}/{}_actor{}.pth".format(EnvName, EnvName,episode)))
		self.q_critic1.load_state_dict(torch.load("./model/{}/{}_q_critic1{}.pth".format(EnvName, EnvName,episode)))
		self.q_critic2.load_state_dict(torch.load("./model/{}/{}_q_critic2{}.pth".format(EnvName, EnvName, episode)))

class Regularization(torch.nn.Module):
	def __init__(self, model, weight_decay, p=2):
		super(Regularization, self).__init__()
		if weight_decay <= 0:
			print("param weight_decay can not <=0")
			exit(0)
		self.model = model
		self.weight_decay = weight_decay
		self.p = p
		self.weight_list = self.get_weight(model)

	def to(self, device):
		self.device = device
		super().to(device)
		return self

	def forward(self, model):
		self.weight_list = self.get_weight(model)  # get the newest weight
		reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
		return reg_loss

	def get_weight(self, model):
		weight_list = []
		for name, param in model.named_parameters():
			if 'weight' in name:
				weight = (name, param)
				weight_list.append(weight)
		return weight_list

	def regularization_loss(self, weight_list, weight_decay, p=2):
		reg_loss = 0
		if p == 2:
			for name, w in weight_list:
				l2_reg = torch.sum(torch.pow(w, 2))
				reg_loss = reg_loss + l2_reg
			reg_loss /= 2
		elif p == 1:
			for name, w in weight_list:
				l1_reg = torch.sum(torch.abs(w))
				reg_loss = reg_loss + l1_reg
		reg_loss = weight_decay * reg_loss
		return reg_loss

	def weight_info(self, weight_list):
		print("---------------regularization weight---------------")
		for name, w in weight_list:
			print(name)
		print("---------------------------------------------------")









