LayeredRL Documentation
=======================

**LayeredRL** is a modular hierarchical reinforcement learning library that enables flexible composition of RL algorithms into hierarchies.

Hierarchical reinforcement learning (HRL) decomposes complex tasks into simpler subtasks, allowing agents to learn and plan at multiple
levels of abstraction. While this approach holds great promise for tackling complex long-horizon tasks, implementations of HRL methods are
usually scattered over many small, incompatible repositories. Hierarchical algorithms are often naturally decomposed into levels or layers,
however. LayeredRL attempts to make use of this inherent modularity by providing a framework for flexibly combining levels corresponding to
different algorithms into hierarchies. This approach allows researchers and practitioners to easily experiment with different hierarchical
designs and leverage existing implementations of RL algorithms.

Key features:

* **Modular Design**: Freely combine different RL algorithms (model-free, model-based, pure planning, and skill learning) as levels in a hierarchy
* **Automatic Dimension Adaptation**: Outputs of one level are automatically adapted to next level's inputs
* **Gymnasium Support**: Supports gymnasium environments
* **Vectorization Support**: Full support for vectorized environments (batched parallel execution)
* **Multiple Algorithms**: TD-MPC2, model-free RL from the Tianshou library, DADS, SPlaTES, and planning methods
* **Semi-MDP Transitions**: Handles variable-duration transitions in hierarchies

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/hierarchies
   user_guide/levels
   user_guide/models

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/gallery

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/hierarchies
   api/levels
   api/planners
   api/policies
   api/models
   api/predictors
   api/nets
   api/collectors
   api/envs
   api/optimizers
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   citation
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
