# Modified version of TD-MPC2

This directory contains a modified version of the model-based RL algorithm TD-MPC2. The original code can be found [here](https://github.com/nicklashansen/tdmpc2). The license file can be found in this directory.

The main modifications are:
* Support for vector environments (add a dimension for the environment instance to everything, also in the planning code).
* Also learn termination probability and take it into account when planning.
* Add a goal-based version of TD-MPC2 (in `goal_based_tdmpc2.py`). The goal can also be a skill vector (just something that modulates the behavior). 
* Add a SPlaTES (TODO: Link) version of TD-MPC2 (in `splates_tdmpc2.py`) which
    * keeps track of the intra-skill time step, the state the skill started in, and the skill vector.
    * bootstraps from a special Q function that is averaged over a uniform distribution of skill vectors at the end of a skill execution.