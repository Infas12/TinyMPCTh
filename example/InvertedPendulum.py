import gymnasium as gym
import numpy as np
import torch
import sys
import os

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(DIR)
sys.path.append(MAIN_DIR)
from tinympcTh.tinympc_torch import MPCSolver, LinearDynamics, LinearCost, LinearConstraints, MPCParams

# environment infomation
# gym vecsenv does not support setting the initial state, so only 1 env is used
env = gym.make('Pendulum-v1', render_mode="human")
observation, info = env.reset()
env.state[0] = -0.2
num_envs = 1

# system dynamics
g = 10
m = 1
l = 1
dt = 0.05
A = torch.tensor([[1, dt], [3 * g / (2 * l), 1]]).unsqueeze(0).repeat(num_envs,1,1)
B = torch.tensor([[0],[3./(m*l**2)*dt]]).unsqueeze(0).repeat(num_envs,1,1)
dyn = LinearDynamics(A,B)

# cost function
Q = torch.eye(2).unsqueeze(0).repeat(num_envs,1,1)
R = torch.eye(1).unsqueeze(0).repeat(num_envs,1,1)
Qf = torch.eye(2).unsqueeze(0).repeat(num_envs,1,1)
cost = LinearCost(Q,R,Qf)

# constraints
xlb = torch.ones((2,1)).unsqueeze(0).repeat(num_envs,1,1) * -0.2
xub = torch.ones((2,1)).unsqueeze(0).repeat(num_envs,1,1) * 0.2
ulb = torch.ones((1,1)).unsqueeze(0).repeat(num_envs,1,1) * -2
uub = torch.ones((1,1)).unsqueeze(0).repeat(num_envs,1,1) * 2
constraints = LinearConstraints(xlb,xub,ulb,uub)

# solver parameters
mpcsteps = 10
MPC_params = MPCParams(mpc_steps=mpcsteps, 
                       rho=0.001, 
                       recatti_iter=5000, 
                       mpc_max_iter=100)


# initialize solver
mpc = MPCSolver(dyn=dyn, cost=cost, constraints=constraints, params=MPC_params, num_envs=num_envs, device='cpu')


for i in range(1000):

    x0 = env.state.copy()
    xref = np.zeros((2,mpcsteps))

    x0 = torch.tensor(x0).reshape(2,1)
    x0 = x0.unsqueeze(0).repeat(num_envs,1,1)
    xref =  torch.tensor(xref).unsqueeze(0).repeat(num_envs,1,1)

    mpc.solve(xref,x0)
    action = mpc.u[0,:,0].numpy()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()