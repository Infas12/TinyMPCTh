# TinyMPCTh
Pytorch Implementation of TinyMPC, a lightweight ADMM-based mpc solver. TinyMPC is division-free and requires no matrix factorization, which makes it robust and efficient.

TinyMPCTh can handle convex QP MPC problems in the following form:

$$
\begin{array}{cl}
\operatorname{minimize} & \frac{1}{2}\left(x_N-\bar{x}_N\right)^T Q_f\left(x_N-\bar{x}_N\right)+ \\
& \sum_{k=0}^N\left(\frac{1}{2}\left(x_k-\bar{x}_k\right)^T Q\left(x_k-\bar{x}_k\right)+\frac{1}{2}\left(u_k\right)^T R\left(u_k\right)\right) \\
\text { subject to } & x_{k+1}=A x_k+B u_k \\
& \bar{u} \leq u_k \leq \underline{u} \\
& \bar{x} \leq x_k \leq \underline{x}
\end{array}
$$


## Dependencies

Pytorch

**For inverted pendulum example:** Gymnasium, Gymnaisum[classic-control]

**For cart-pole example:** isaacgym(Previwe 4), isaacgymenvs

## Examples
**example with single robot:** 

Double Integrator
```
python3 example/DoubleIntegrator.py
```

Inverted pendulum with revolute joint
```
python3 example/InvertedPendulum.py
```

**example with multiple robots:** 

Cartpole
```
python3 example/CartPole.py
```

Quadrupedal Robots:

TBD: document under construction. We used a modified solver to handle the friction cone constraints.


## Note:
Gymnaium requires numpy-1.24.4 while isaacgym requires numpy-1.20.0. You may need multiple venvs to run these examples.


