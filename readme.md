# TinyMPCTh
Pytorch Implementation of TinyMPC, a lightweight ADMM-based mpc solver. TinyMPC is division-free and requires no matrix factorization, which makes it robust and efficient.

TinyMPCTh can handle convex QP MPC problems in the following form:

(Latex here)


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


