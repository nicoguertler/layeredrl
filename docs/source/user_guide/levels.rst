Levels
======

Levels are the building blocks of hierarchies in LayeredRL. Each level implements a particular control strategy
like an RL or planning algorithm. To this end, it

* Takes an environment observation and level input (from the level above) and produces a level output
* Processes each (semi-MDP) transition, e.g. by calculating a reward and storing the transition in a replay buffer
* Learns from this experience, e.g. by sampling a batch from the replay buffer and performing gradient descent on a loss

Available Level Types
---------------------

Available level types:

+------------------------------------------------------------------+------------------------+
| Supported levels                                                 | Type                   |
+==================================================================+========================+
| :class:`TD-MPC2 <layeredrl.levels.TDMPC2Level>`                  | Model-based RL         |
+------------------------------------------------------------------+------------------------+
| :class:`Tianshou <layeredrl.levels.TianshouLevel>`               | Model-free RL          |
+------------------------------------------------------------------+------------------------+
| :class:`DADS <layeredrl.levels.DADSLevel>`                       | Skill (TD-MPC2 version)|
+------------------------------------------------------------------+------------------------+
| :class:`SPlaTES <layeredrl.levels.SPlaTESLevel>`                 | Skill                  |
+------------------------------------------------------------------+------------------------+
| :class:`Planner with (i)CEM <layeredrl.levels.PlannerLevel>`     | Planner                |
+------------------------------------------------------------------+------------------------+

TD-MPC2 Level
^^^^^^^^^^^^^

Model-based RL using a vectorized version of the `TD-MPC2 algorithm <https://arxiv.org/abs/2310.16828>`__.

.. code-block:: python

    from layeredrl.levels import TDMPC2Level 
    from layeredrl.tdmpc2 import get_default_tdmpc2_config

    tdmpc2_config = get_default_tdmpc2_config()
    tdmpc2_config["discount_factor"] = [0.95]
    level = TDMPC2Level(tdmpc2_config=tdmpc2_config)

Setting ``goal_based=True`` will switch to a goal-based version of TD-MPC2 which treats the goal separately. This requires
specifying which parts of the observation contain the desired and achieved goal. Using HER requires specifying a reward
function. See :class:`~layeredrl.levels.TDMPC2Level` for details.

Tianshou Level
^^^^^^^^^^^^^^

Model-free RL using the `Tianshou library <https://github.com/thu-ml/tianshou>`__. Hydra's instantiate
function is used to instantiate actor, critics etc. as specified in a config dictionary.

.. code-block:: python

    from layeredrl.levels import TianshouLevel

    tianshou_config = {
        "n_critics": 2,
        "actor": {
            "_target_": "tianshou.utils.net.continuous.ActorProb",
            ...
        },
        "critic": {
            "_target_": "layeredrl.nets.Critic",
            ...
        },
        "optims": {
            ...
        },
        "policy": {
            "_target_": "tianshou.policy.SACPolicy",
            ...
        },
        "policy_dynamic_args": {
            "actor": "__actor__",
            "actor_optim": "__actor_optim__",
            "critic1": "__critic_0__",
            "critic1_optim": "__critic_optim_0__",
            "critic2": "__critic_1__",
            "critic2_optim": "__critic_optim_1__",
            "action_space": "__action_space__",
        },
    }

    level = TianshouLevel(tianshou_config=tianshou_config, buffer_size=10000)

**Features:**

* Access to various Tianshou algorithms (SAC, PPO, TD3, etc.)
* Efficient model-free learning

Note that the level was only tested with SAC and DQN so far. See :class:`~layeredrl.levels.TianshouLevel` for details.

DADS Level
^^^^^^^^^^

Diversity-driven skill discovery with the `DADS algorithm <https://arxiv.org/abs/1907.01657>`__. Note that this implementation
is based on TD-MPC2 while the original implementation used SAC. For details see :class:`~layeredrl.levels.DADSLevel`.

.. code-block:: python

    from layeredrl.levels import DADSLevel

    dads_level = DADSLevel(
        skill_space_dim=...,  # dimension of skill vector
    )

**Features:**

* Learns diverse skills by maximizing mutual information between one-step transitions and skill vector
* Requires parent level to contain a predictor with an encoder that defines the space in which to diversify the skills

SPlaTES Level
^^^^^^^^^^^^^

Skill discovery with the `SPlaTES algorithm <https://nicoguertler.github.io/splates-pages/>`__. The discovered skills
are

* temporally extended,
* predictable, and
* diverse.

Whether the skills are useful for the task depends on the encoder of the predictor of the parent level, which defines
the space in which the skills are trained to be predictable and diverse. It can either be fixed manually with domain 
knowledge (see `examples/splates_hierarchy.py`) or learned from a dense reward (see TODO).

.. code-block:: python

    from layeredrl.levels import SPlaTESLevel

    splates_level = SPlaTESLevel(
        skill_space_dim=...,  # dimension of skill vector
    )

**Features:**

* Temporally extended, predictable and diverse skills
* Skills enable long-horizon planning
* Requires parent level to contain a predictor with an encoder that defines the space in which to diversify the skills

For details see :class:`~layeredrl.levels.SPlaTESLevel`.

Planner Level
^^^^^^^^^^^^^

Learns a world model in a predictor, which enables it to pick good actions via running a planner. Note that the
predictor models the transitions the planner level sees. These might be Semi-MDP transitions, i.e., they last
several environment time steps, if the planner level is used as a higher level (not at the bottom of the hierarchy).
At the moment, LayeredRL implements CEM and iCEM as zeroth-order optimization methods for planning.

The example code below uses a default predictor which is based on :class:`~layeredrl.predictors.RewardPredictor` 
and the :class:`~layeredrl.models.ProbabilisticEnsemble` model and assumes that the environment is goal-based. 
The encoder treats the desired goal as the context and the achieved goal as the state. The predictor can 
be customized to adapt to different environments and to tune hyperparameters, however.

.. code-block:: python

    from layeredrl.levels import PlannerLevel
    from layeredrl.predictors import get_default_predictor_factory
    from layeredrl.planners import CEMPlanner

    predictor_factory = get_default_predictor_factory(env)
    cem_params = {
        "n_samples": 256,
        "n_iterations": 6,
        "elite_ratio": 0.05,
        "momentum": momentum,
        "clip": clip,
        "return_mode": "mean",
    }
    planner_factory = partial(
        CEMPlanner,
        cem_params=cem_params,
    )
    planner_level = PlannerLevel(
        partial_planner=planner_factory,
        predictor_factory=predictor_factory,
        initial_guess=torch.zeros(...),
        horizon=horizon,
    )

**Features:**

* Model-based planning using a world model
* Works with learned value/reward functions
* Can alternate with noise for good coverage of action space during world model learning
* Good for high-level strategic planning

Custom Levels
-------------

You can create custom levels by inheriting from the :class:`~layeredrl.levels.Level` base class and implementing
the required abstract methods. This incomplete example demonstrates what needs to be implemented for a class
which generates actions based on some form of policy.

.. code-block:: python

    from typing import Dict, Optional, Tuple
    import torch
    from gymnasium.spaces import Space, Box
    from layeredrl.levels import Level

    class MyCustomLevel(Level):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Initialize your level-specific parameters
            self.policy = None  # Will be created in initialize()

        def initialize(
            self,
            env_obs_space: Space,
            action_space: Space,
            n_env_instances: int,
            parent_predictor,
            env_obs_map=None,
            mapped_env_obs_shape=None,
            keep_params=False,
        ):
            """Initialize the level after the hierarchy is constructed."""
            super().initialize(
                env_obs_space,
                action_space,
                n_env_instances,
                parent_predictor,
                env_obs_map,
                mapped_env_obs_shape,
                keep_params,
            )
            if not keep_params:
                # Create your policy/networks here
                self.policy = ...  # Initialize based on action_space, etc.

        def get_input_space(self) -> Space:
            """Define what input this level expects from the level above.

            Returns None if no input is needed (e.g., for top-level).
            For middle levels, return a Box or Discrete space.
            """
            return None  # or Box(...) for levels expecting input, e.g. goals or skill vectors

        def get_action(
            self,
            mapped_env_obs: torch.Tensor,
            level_input: Optional[torch.Tensor],
            level_input_info: Optional[Dict],
            active_instances: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
            """Compute action for active environment instances.

            Args:
                mapped_env_obs: Environment observation (batch_size, ...)
                level_input: Input from level above (batch_size, input_dim)
                level_input_info: Additional info about level input
                active_instances: Boolean mask of active instances

            Returns:
                action: Action tensor (num_active, action_dim)
                action_info: Dict with additional action information
            """
            # Get level state if needed
            level_state = self.get_level_state(
                mapped_env_obs, level_input, active_instances
            )
            # Compute action using your policy
            action = self.policy(mapped_env_obs[active_instances], level_state, ...)
            return action, {}

        def process_transition(
            self,
            mapped_env_obs: torch.Tensor,
            level_input: Optional[torch.Tensor],
            action: torch.Tensor,
            next_mapped_env_obs: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            active_instances: torch.Tensor,
        ) -> torch.Tensor:
            """Process completed transitions.

            Store transition in replay buffer, compute rewards, etc.

            Returns:
                done: Boolean tensor indicating which instances want to return control to the 
                    level above.
            """
            # Compute reward
            level_state = self.get_level_state(
                mapped_env_obs, level_input, active_instances
            )
            next_level_state = self.get_level_state(
                next_mapped_env_obs, level_input, active_instances
            )
            reward, _ = self.get_reward(
                mapped_env_obs,
                level_input,
                level_state,
                action,
                next_mapped_env_obs,
                next_level_state,
                terminated,
                self.cum_reward[active_instances],
                self.elapsed_env_steps[active_instances],
            )

            # Call parent to update counters and reset cum_reward
            super().process_transition(
                mapped_env_obs,
                level_input,
                action,
                next_mapped_env_obs,
                terminated,
                truncated,
                active_instances,
            )

            # Store transition in your buffer
            # self.buffer.add(...)

            # Return False to keep control, True to return to level above
            return torch.zeros(
                active_instances.shape, dtype=torch.bool, device=self.device
            )

        def learn(self):
            """Update the policy from collected experience."""
            # Sample from buffer and update policy
            # batch = self.buffer.sample(batch_size)
            # loss = ...
            # ...
            pass

**Key methods to implement:**

* :meth:`~layeredrl.levels.Level.get_input_space`: Define the input this level expects
* :meth:`~layeredrl.levels.Level.get_action`: Compute actions from observations
* :meth:`~layeredrl.levels.Level.process_transition`: Handle completed transitions
* :meth:`~layeredrl.levels.Level.learn`: Update the policy (optional, but necessary for trainable levels)

**Optional methods you may want to override:**

* :meth:`~layeredrl.levels.Level.get_level_state`: Add recurrent state or other context
* :meth:`~layeredrl.levels.Level.get_reward`: Define custom rewards (defaults to environment reward)
* :meth:`~layeredrl.levels.Level.save`/:meth:`~layeredrl.levels.Level.load`: Save and load level parameters

If you want the custom level to be compatible with vector environments, make sure to respect the ``active_instances``
tensor which indicates for which of the environments the level is currently active.

API Reference
-------------

For detailed API documentation, see :doc:`../api/levels`.
