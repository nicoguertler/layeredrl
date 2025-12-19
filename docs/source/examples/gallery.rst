Examples
========

This page lists example scripts that illustrate different use cases for LayeredRL. These scripts can be found in the ``examples`` directory.

Flat TD-MPC2 Hierarchy
-----------------------

A simple hierarchy with a single TD-MPC2 level (flat TD-MPC2):

**File:** ``examples/tdmpc2_hierarchy.py``


SPlaTES + Planning
------------------

Learn skills with SPlaTES and plan over them to solve tasks. This demo shows how to create a hierarchy consisting of a planner level and a SPlaTES level.

**File:** ``examples/splates_hierarchy.py``


DADS + Planning
---------------

Learn skills with DADS and plan over them to solve tasks. This demo shows how to create a hierarchy consisting of a planner level and a DADS level.

**File:** ``examples/dads_hierarchy.py``


Model-Based Planning
--------------------

Learn a model from experience and use it for planning. This demo shows how to create a hierarchy with a single planner level that uses a learned model for planning. This script is mostly for illustrative purposes as the planner level is mostly intended to work with a skill level below it. The TD-MPC2 level, on the other hand, is tuned for direct application to continous control.

**File:** ``examples/planning_with_model.py``


Ensemble Rollout
----------------

Demonstrate rollout of an ensemble of probabilistic models with continuous and discrete action spaces.

**File:** ``examples/ensemble_rollout.py``


Blackbox Planning
-----------------

Demonstrate planning with iCEM and a fixed predictor.

**File:** ``examples/blackbox_planning.py``

Running the Examples
--------------------

To run any example (after activating an appropriate virtual environment):

.. code-block:: bash

    cd examples/
    python tdmpc2_hierarchy.py

Examples accept command-line arguments for configuration:

.. code-block:: bash

    python splates_hierarchy.py --n_envs 4 --n_steps 100000 --device cuda:0

LayeredRL on More Challenging Environments
------------------------------------------

Check out the SPlaTES repository (TODO) for examples of how to run LayeredRL on more challenging environments. 


Tests
-----

The ``tests/`` directory contains unit tests that may be helpful for gaining a better understanding of LayeredRL:

* ``tests/test_hierarchies.py`` - Hierarchy composition
* ``tests/test_levels.py`` - Individual level usage
* ``tests/test_collector.py`` - Data collection
* ``tests/test_model_learning.py`` - Model training

Next Steps
----------

* Read the :doc:`../user_guide/hierarchies` guide
* Explore the :doc:`../api/levels` documentation
* Check out the source code on GitHub
