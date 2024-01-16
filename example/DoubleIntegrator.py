import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(DIR)
sys.path.append(MAIN_DIR)

from tinympcTh.tinyiLQR_torch import MPCSolver, LinearDynamics, LinearCost, LinearConstraints, MPCParams

# system parameters
dt = 0.1
num_envs = 1
mpc_steps = 100

# dynamics
A = torch.tensor([[1, dt], [0, 1]]).unsqueeze(0).unsqueeze(-1).repeat(num_envs,1,1,mpc_steps)
B = torch.tensor([[0], [dt]]).unsqueeze(0).unsqueeze(-1).repeat(num_envs,1,1,mpc_steps)
dyn = LinearDynamics(A,B)

# cost
Q = torch.eye(2).unsqueeze(0).repeat(num_envs,1,1)
R = torch.eye(1).unsqueeze(0).repeat(num_envs,1,1) * 0.05
Qf = torch.eye(2).unsqueeze(0).repeat(num_envs,1,1)
cost = LinearCost(Q, R, Qf)

# constraints
xlb = torch.ones((2,1)).unsqueeze(0).repeat(num_envs,1,1) * -20
xub = torch.ones((2,1)).unsqueeze(0).repeat(num_envs,1,1) * 20
ulb = torch.ones((1,1)).unsqueeze(0).repeat(num_envs,1,1) * -1
uub = torch.ones((1,1)).unsqueeze(0).repeat(num_envs,1,1) * 1
constraints = LinearConstraints(xlb,xub,ulb,uub)

# mpc params

params = MPCParams(mpc_steps, 0.001, 100)

# solver
mpc = MPCSolver(dyn, cost, constraints, params ,num_envs,'cpu')

# reference trajectory and initial state
xref = np.zeros((2,mpc_steps))+[[4],[0]]
x0 = np.array([[1],[1]])

xref =  torch.tensor(xref).unsqueeze(0).repeat(num_envs,1,1)
x0 = torch.tensor(x0).unsqueeze(0).repeat(num_envs,1,1)

mpc.solve(xref,x0)


plt.plot(mpc.x[0,0,:].numpy())
plt.plot(mpc.x[0,1,:].numpy())
plt.plot(mpc.u[0,0,:].numpy())
plt.show()