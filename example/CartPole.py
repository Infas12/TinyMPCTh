import isaacgym
import isaacgymenvs
import torch
import sys
import os

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(DIR)
sys.path.append(MAIN_DIR)

from tinympcTh.tinympc_torch import MPCSolver, LinearDynamics, LinearCost, LinearConstraints, MPCParams



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
mpc_dt = 0.005

# system dynamics
A_ct = torch.tensor([[0, 1, 0, 0], 
                     [0, 0, m_pole*g/M_cart,0],
                     [0, 0, 0, 1],
                     [0, 0, (M_cart+m_pole)*g/(M_cart*l_pole), 0]]
                    ).unsqueeze(0).repeat(num_envs,1,1)
B_ct = torch.tensor([[0],
                     [1.0/M_cart],
					 [0],
					 [1.0/(M_cart*l_pole)]]
					).unsqueeze(0).repeat(num_envs,1,1)
A_dt = torch.zeros((num_envs,4,4)) + torch.eye(4) + A_ct * mpc_dt
B_dt = B_ct * mpc_dt
dyn = LinearDynamics(A_dt,B_dt)

# system cost
Q = torch.tensor([[100,0,0,0],
				  [0,0.1,0,0],
				  [0,0,5,0],
				  [0,0,0,1]])
Q = torch.zeros(num_envs,4,4) + Q
R = torch.eye(1).unsqueeze(0).repeat(num_envs,1,1) 
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

# params
mpc_steps = 5
params = MPCParams(mpc_steps, 1e-3, 5000, 100)


# mpc
mpc = MPCSolver(dyn, cost, constraints, params, num_envs, 'cpu')



for _ in range(3000):

	x0 = envs.obs_buf.reshape(num_envs,4,1)
	xref = torch.zeros((num_envs,4,mpc_steps))
 
	mpc.solve(xref,x0)
	actions = mpc.u[:,:,0]
	envs.step(actions)

