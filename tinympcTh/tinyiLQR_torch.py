import torch

class LinearDynamics:
    def __init__(self, A, B):
        self.A = A
        self.B = B

class LinearCost:
    def __init__(self, Q, R, Qf):
        self.Q = Q
        self.R = R
        self.Qf = Qf

class LinearConstraints:
    def __init__(self, xlb, xub, ulb, uub):
        self.xlb = xlb
        self.xub = xub
        self.ulb = ulb
        self.uub = uub

class MPCParams:
    def __init__(self, mpc_steps, rho=0.001, mpc_max_iter=100):
        self.mpc_steps = mpc_steps
        self.rho = rho
        self.mpc_max_iter = mpc_max_iter
        

class MPCSolver:
    def __init__(self, dyn:LinearDynamics, cost:LinearCost, constraints:LinearConstraints, params:MPCParams, num_envs, device):
        self.dyn = dyn
        self.cost = cost
        self.constraints = constraints
        self.mpcparams = params

        self.num_envs = num_envs
        self.device = device

        # Problem dimensions
        self.N = params.mpc_steps 
        self.nu = dyn.B.shape[2]
        self.nx = dyn.A.shape[1]

        # MPC parameters
        self.rho = params.rho

        self.P = torch.zeros((num_envs, self.nx, self.nx, self.N),device=device)
        self.K = torch.zeros((num_envs, self.nu, self.nx, self.N),device=device)

        # reference state trajectory
        self.xref = torch.zeros((num_envs, self.nx, self.N),device=device)
        
        # State and input trajectories
        self.x = torch.zeros((num_envs, self.nx, self.N),device=device)
        self.u = torch.zeros((num_envs, self.nu, self.N-1),device=device)

        # Linear control cost terms
        self.q = torch.zeros((num_envs, self.nx, self.N),device=device)
        self.r = torch.zeros((num_envs, self.nu, self.N-1),device=device)

        # Linear Riccati backward pass terms
        self.p = torch.zeros((num_envs, self.nx, self.N),device=device)
        self.d = torch.zeros((num_envs, self.nu, self.N-1),device=device)

        # auxiliary variables; notation is different from paper.
        # TODO: change notation
        self.v = torch.zeros((num_envs, self.nx, self.N),device=device)
        self.vnew = torch.zeros((num_envs, self.nx, self.N),device=device)
        self.z = torch.zeros((num_envs, self.nu, self.N-1),device=device)
        self.znew = torch.zeros((num_envs, self.nu, self.N-1),device=device)

        # Dual variables
        self.g = torch.zeros((num_envs, self.nx, self.N),device=device)
        self.y = torch.zeros((num_envs, self.nu, self.N-1),device=device)

 
    def finite_horizon_lqr(self):
        
        self.P[:,:,:,self.N-1] = self.cost.Qf + torch.eye(self.nx,device=self.device)[None,:] * self.rho
        for i in range(self.N-2, -1, -1):
            RpBTpkp1B = self.cost.R + torch.bmm(torch.bmm(self.dyn.B[:,:,:,i].transpose(1,2),self.P[:,:,:,i+1]),self.dyn.B[:,:,:,i])
            self.K[:,:,:,i] = torch.bmm(
                RpBTpkp1B,
                torch.bmm(torch.bmm(self.dyn.B[:,:,:,i].transpose(1,2),self.P[:,:,:,i+1]),self.dyn.A[:,:,:,i]))
            self.P[:,:,:,i] = self.cost.Q + \
                    torch.bmm(torch.bmm(self.dyn.A[:,:,:,i].transpose(1,2),self.P[:,:,:,i+1]),self.dyn.A[:,:,:,i]) - \
                    torch.bmm(torch.bmm(torch.bmm(self.dyn.A[:,:,:,i].transpose(1,2),self.P[:,:,:,i+1]),self.dyn.B[:,:,:,i]),self.K[:,:,:,i])
            
    
    def backward_pass(self):
                
        for i in range(self.N-2, -1, -1):
        
            RpBtPB = self.cost.R + torch.bmm(torch.bmm(self.dyn.B[:,:,:,i].transpose(1,2),self.P[:,:,:,i+1]),self.dyn.B[:,:,:,i])
            C1 = torch.linalg.inv(RpBtPB)

            AmBK = self.dyn.A[:,:,:,i] - torch.bmm(self.dyn.B[:,:,:,i],self.K[:,:,:,i])
            C2 = AmBK.transpose(1,2)

            C3 = torch.bmm(self.K[:,:,:,i].transpose(1,2),self.cost.R)  - \
                torch.bmm(torch.bmm(C2,self.P[:,:,:,i+1]),self.dyn.B[:,:,:,i])        
        
            self.d[:,:,i:i+1] = torch.bmm(C1,(torch.bmm(self.dyn.B[:,:,:,i].transpose(1,2),self.p[:,:,i+1:i+2]) + self.r[:,:,i:i+1]))
            self.p[:,:,i:i+1] = self.q[:,:,i:i+1] + \
                torch.bmm(C2,self.p[:,:,i+1:i+2]) - \
                torch.bmm(self.K[:,:,:,i].transpose(1,2),self.r[:,:,i:i+1]) + \
                torch.bmm(C3,self.d[:,:,i:i+1])
            
    def forward_pass(self):
        for i in range(self.N-1):
            self.u[:,:,i:i+1] =  - torch.bmm(self.K[:,:,:,i],self.x[:,:,i:i+1]) - self.d[:,:,i:i+1]
            self.x[:,:,i+1:i+2] = torch.bmm(self.dyn.A[:,:,:,i],self.x[:,:,i:i+1]) + torch.bmm(self.dyn.B[:,:,:,i],self.u[:,:,i:i+1])
    
    def update_slack(self):
        # project to feasible set (element-wise min and max)
        tmp_znew = self.u + self.y
        tmp_vnew = self.x + self.g

        self.znew[:] = torch.clip(tmp_znew, self.constraints.ulb, self.constraints.uub) # might be auto-broadcasting here, gonna fix later
        self.vnew[:] = torch.clip(tmp_vnew, self.constraints.xlb, self.constraints.xub)

    def update_dual(self):
        self.y[:] = self.y + self.u - self.znew
        self.g[:] = self.g + self.x - self.vnew

    def update_linear_cost(self):
        self.r[:] = 0 - self.rho * (self.znew - self.y) # update r 1:N-1; uref = 0 in this case.
        self.q[:] = - torch.bmm(self.cost.Q.transpose(1,2),self.xref) - self.rho * (self.vnew - self.g) # update q 1:N 
        
        self.p[:,:,self.N-1:self.N] = - torch.bmm(self.xref[:,:,self.N-1:self.N].transpose(1,2),self.P[:,:,:,self.N-1]).transpose(1,2) - \
            self.rho * (self.vnew[:,:,self.N-1:self.N] - self.g[:,:,self.N-1:self.N])# update p N

    def termination_condition(self):
        primal_residual_state = torch.max(torch.abs(self.x - self.vnew))
        solver_residual_state = torch.max(torch.abs(self.v - self.vnew))
        primal_residual_input = torch.max(torch.abs(self.u - self.znew))
        solver_residual_input = torch.max(torch.abs(self.z - self.znew))
        if  primal_residual_state < 1e-3 and \
            solver_residual_state < 1e-3 and \
            primal_residual_input < 1e-3 and \
            solver_residual_input < 1e-3: #hard coded
            return True
        else:
            return False
    
    def reset(self):
        self.x[:] = 0
        self.u[:] = 0
        self.q[:] = 0
        self.r[:] = 0
        self.p[:] = 0
        self.d[:] = 0
        self.v[:] = 0
        self.vnew[:] = 0
        self.z[:] = 0
        self.znew[:] = 0
        self.g[:] = 0
        self.y[:] = 0

    def solve(self, xref, x0):
        
        self.reset()
        self.x[:,:,0:1] = x0
        
        self.xref[:] = xref
        
        self.finite_horizon_lqr()

        for i in range(self.mpcparams.mpc_max_iter):
            self.forward_pass()
            self.update_slack()
            self.update_dual()
            self.update_linear_cost()
            self.v[:] = self.vnew
            self.z[:] = self.znew
            
            if i > 0 and i % 10 == 0:
               if self.termination_condition():
                    return
            
            self.backward_pass()