Quickstart
==========

This guide will help you get started with LayeredRL by creating a simple hierarchical RL setup.

Basic Hierarchy
---------------

This is a minimal example of creating and training a hierarchy. The lower level learns
SPlaTES skills and the higher level generates plans that chain these skills to achieve
a high return. Setting the autoreset mode of gymnasium to ``SAME_STEP`` is necessary to make
sure LayeredRL can process environment resets correctly.

.. code-block:: python

    import gymnasium as gym
    from layeredrl.hierarchies import Hierarchy
    from layeredrl.levels import PlannerLevel, SPlaTESLevel
    from layeredrl.predictors import get_default_predictor_factory

    skill_space_dim = ...  # dimensionality of skill vector space

    env = gym.make_vec(
        id="...",
        num_envs=...,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    predictor_factory = get_default_predictor_factory(env)
    planner_factory = partial(CEMPlanner)
    planner_level = PlannerLevel(
        partial_planner=planner_factory,
        predictor_factory=predictor_factory,
        initial_guess=torch.zeros(skill_space_dim),
        horizon=...,
    )

    splates_level = SPlaTESLevel(
        skill_space_dim=skill_space_dim,
        control_interval=...,
    )

    # Create a simple two-level hierarchy
    hierarchy = Hierarchy(
        levels=[
            planner_level,  # Higher level
            splates_level,  # Lower level
        ]
    )

    # Train the hierarchy
    hierarchy.train()
    collector = Collector(hierarchy=hierarchy, env=env)
    collector.reset()
    stats = collector.collect(n_steps=..., learn=True)
    print(f"Training stats: {stats}")

:meth:`~layeredrl.predictors.get_default_predictor_factory` returns a predictor factory that creates a `Predictor` object
that models high-level transitions. It assumes that the environment is goal-based and interprets the
desired goal as the context and the achieved goal as the state for the planner level.

For full code with reasonable hyperparameters for the Maze2D-Medium-v0 environment, see the 
``splates_hierarchy.py`` example.

Achieving good performance on a specific environment generally requires choosing appropriate
hyperparameters and potentially choosing or learning a custom encoder for the planner level.
For an example of SPlaTES running on more challenging MuJoCo environments, see the SPlaTES repository (TODO).

Logging with Tensorboard and Weights & Biases
---------------------------------------------

For logging with tensorboard, pass a ``SummaryWriter`` object to each level you want to
participate in logging and to the collector to monitor return and success rate during training:

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter("path/to/logdir")
    planner_level = PlannerLevel(...,writer=writer)
    splates_level = SPlaTESLevel(...,writer=writer)
    ...
    collector = Collector(...,writer=writer)

To additionally log with Weights & Biases, set ``sync_tensorboard=True``:

.. code-block:: python

    import wandb

    wandb.init(
        project="project_name",
        sync_tensorboard=True,
        name="run_name",
        dir="/log/dir",
    )

Testing periodically during training
------------------------------------

While training return and success rate are monitored by default by :meth:`~layeredrl.collectors.Collector.collect`,
it can make sense to also periodically test with :meth:`layeredrl.hierarchies.Hierarchy.eval()` as this 
may disable exploration noise (depending on the level type). Simply instantiate a second test 
vector environment and pass it to :meth:`~layeredrl.collectors.Collector.collect`:

.. code-block:: python

    test_env = gym.make_vec(
        id="...",
        num_envs=...,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    stats = collector.collect(
        n_steps=...,
        learn=True,
        test_interval=...,  # how often to test
        n_test_steps=...,  # for how many vec env steps to test
    )


Saving and loading hierarchies
-------------------------------

To save the parameters of a hierarchy simply run:

.. code-block:: python

    hierarchy.save("path/to/model/dir")

This will create a directory in which each level will save its parameters. To load a set of
saved parameters, run:

.. code-block:: python

    hierarchy.load("path/to/model/dir")

The same pattern works for saving and loading buffers:

.. code-block:: python

    hierarchy.save_buffers("path/to/buffer/dir")
    # and
    hierarchy.load_buffers("path/to/buffer/dir")

If you want to save hierarchy checkpoints periodically during training, specify a checkpoint interval and directory when instantiating :class:`~layeredrl.collectors.Collector`:

.. code-block:: python

    from pathlib import Path

    collector = Collector(
        hierarchy=hierarchy,
        env=env,
        ...,
        checkpoint_dir=Path("/path/to/checkpoint/dir"),
        checkpoint_interval=...,  # checkpoint every ... vec env steps
    )

Next Steps
----------

* Learn more about :doc:`user_guide/hierarchies`
* Explore available :doc:`user_guide/levels`
* Check out the :doc:`examples/gallery`
