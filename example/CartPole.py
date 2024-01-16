import isaacgym
import isaacgymenvs
import torch
import sys
import os

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(DIR)
sys.path.append(MAIN_DIR)

from tinympcTh.tinyiLQR_torch import MPCSolver, LinearDynamics, LinearCost, LinearConstraints, MPCParams



# setup environment
num_envs = 128
envs = isaacgymenvs.make(
	seed=0, 
	task="Cartpole", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=False,
	force_render=True
)
envs.is_vector_env = True
envs.reset()

# Default dt is 1/60, which is too noisy; hacky way to reset sim dt 
params = envs.gym.get_sim_params(envs.sim)
params.dt = 0.005
envs.gym.set_sim_params(envs.sim, params)


# mpc parameters
g = 9.81
m_pole = 1
M_cart = 1
l_pole = 0.47
mpc_dt = 0.015
mpc_steps = 50
params = MPCParams(mpc_steps, 1e-3, 3)

# system dynamics
A = torch.tensor([[0, 1, 0, 0], 
                     [0, 0, m_pole*g/M_cart,0],
                     [0, 0, 0, 1],
                     [0, 0, (M_cart+m_pole)*g/(M_cart*l_pole), 0]]
                    ) * mpc_dt + torch.eye(4)
A_dt = A.unsqueeze(0).unsqueeze(-1).repeat(num_envs,1,1,mpc_steps)


B_ct = torch.tensor([[0],
                     [1.0/M_cart],
					 [0],
					 [1.0/(M_cart*l_pole)]]
					).unsqueeze(0).unsqueeze(-1).repeat(num_envs,1,1,mpc_steps)


B_dt = B_ct * mpc_dt
dyn = LinearDynamics(A_dt,B_dt)

# system cost
Q = torch.tensor([[50,0,0,0],
				  [0,0.1,0,0],
				  [0,0,500,0],
				  [0,0,0,1]])
Q = torch.zeros(num_envs,4,4) + Q
R = torch.eye(1).unsqueeze(0).repeat(num_envs,1,1) * 0.01
Qf = torch.zeros(num_envs,4,4) + Q
cost = LinearCost(Q,R,Qf)

# system constraints
xub = torch.zeros((num_envs, 4,1))
xub[:,0,:] = 3
xub[:,1,:] = 5
xub[:,2,:] = 0.1
xub[:,3,:] = 1.5
xlb = -xub
ulb = torch.ones((1,1)).unsqueeze(0).repeat(num_envs,1,1) * -4
uub = torch.ones((1,1)).unsqueeze(0).repeat(num_envs,1,1) * 4
constraints = LinearConstraints(xlb,xub,ulb,uub)




# mpc
mpc = MPCSolver(dyn, cost, constraints, params, num_envs, 'cpu')



for _ in range(3000):

	x0 = envs.obs_buf.reshape(num_envs,4,1)
	xref = torch.zeros((num_envs,4,mpc_steps))
 
	mpc.solve(xref,x0)
	actions = mpc.u[:,:,0]
	envs.step(actions)

