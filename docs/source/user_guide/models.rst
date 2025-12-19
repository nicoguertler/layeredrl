Models and Predictors
=====================

Each level can have a :class:`~layeredrl.predictors.Predictor` object, which contains all models that predict certain
aspects of the (Semi-)MDP the level interacts with, like

* dynamics model,
* reward model,
* value function, and
* encoder.

The predictor is responsible for implementing and training these components. The main purpose of the predictor is
to enable planning. The encoder additionally predicts a state and a context (assumed to be constant in each episode)
from observations.

Furthermore, each level has access to the predictor
of its parent level (if it exists) via the ``parent_predictor`` attribute. This is useful for skill learning 
methods that compute an estimate of a mutual-information-based reward based on the dynamics model of the predictor.


Models
------

In LayeredRL, models refer to dynamics models, which predict the next state given the current state and an action.

.. code-block:: python

    s_next_mean, weights, s_next_std, term_prob = model.predict(state, context, action)

Here, `weights` refers to the weights of a mixture of Gaussians with the corresponding means in `s_next_mean`.

Predicting repeatedly yields a rollout:

.. code-block:: python

    traj = model.rollout(initial_state, context, actions)

where ``traj`` is a Batch object which contains keys for states, termination probabilities etc.


Probabilistic Ensemble
^^^^^^^^^^^^^^^^^^^^^^

:class:`~layeredrl.models.ProbabilisticEnsemble` is a model that enables uncertainty estimation by learning an ensemble of 
dynamics models. It furthermore supports several particles per ensemble member and several Gaussian modes per network particle.

.. code-block:: python

    from layeredrl.models import ProbabilisticEnsemble

    ensemble = ProbabilisticEnsemble(
        state_space=...,
        context_space=...,
        action_space=...,
        partial_net=...,  # factory function for ensemble member networks
        n_models=...,
        n_modes=...,
        n_particles_per_model=...,
    )

**Benefits:**

* Estimates uncertainty via ensemble disagreement
* Can be used for exploration bonuses
* Can model transitions via mixture of Gaussians


Predictors
----------

The following predictors are currently implemented in LayeredRL:

Static Predictor
^^^^^^^^^^^^^^^^

Does not train any of its components. Can be used with known/manually defined dynamics and reward models, for example:

.. code-block:: python

    from layeredrl.predictors import StaticPredictor
    from layeredrl.nets import IdentityEncoder
    
    encoder = IdentityEncoder(obs.shape)
    predictor = StaticPredictor(
        model=model,
        val_func=val_func,
        rew_func=rew_func,
        encoder=encoder,
        latent_state_dim=...,
        context_dim=0,
    )

Reward Predictor
^^^^^^^^^^^^^^^^

Trains dynamics model, reward function, and value function. It optionally also trains the encoder by backpropagating through it 
from the reward loss.

.. code-block:: python

    from layeredrl.predictors import RewardPredictor
    from layeredrl.nets import EncoderNet

    encoder = EncoderNet(
        mapped_env_obs_shape=obs.shape,
        latent_state_dim=...,
        context_dim=...,
    )
    predictor = RewardPredictor(
        model=model,
        val_func=val_func,
        rew_func=rew_func,
        encoder=encoder,
        latent_state_dim=...,
        context_dim=...,
        learn_encoder=True,
    )


Encoders
--------

Encoders map raw observations to learned representations of states and contexts.

Identity Encoder
^^^^^^^^^^^^^^^^

Pass observations through unchanged as state:

.. code-block:: python

    from layeredrl.nets import IdentityEncoder

    encoder = IdentityEncoder(obs.shape)
    state, context = encoder(obs)

    # state is equal to obs
    # context.numel() is 0

Fixed Encoder
^^^^^^^^^^^^^

Picks out specified dimensions as state and context:

.. code-block:: python

    from layeredrl.nets import FixedEncoderNet

    encoder = FixedEncoderNet(
        mapped_env_obs_shape=obs.shape,
        latent_state_dims=...,  # List of indices making up state
        context_dims=...,  # List of indices making up context
    )


Learned Encoder
^^^^^^^^^^^^^^^

Trainable linear encoder:

.. code-block:: python

    from layeredrl.nets import EncoderNet

    encoder = EncoderNet(
        mapped_env_obs_shape=obs.shape,
        latent_state_dim=...,
        context_dim=...,
    )


Predictor training
------------------

The most convenient way of training a predictor is to use it as part of a :class:`~layeredrl.levels.PlannerLevel`. However,
it is possible to train a predictor manually:

.. code-block:: python

    loss, loss_info = predictor.learn(
        buffer,
        n_updates,
        batch_size,
        model_batch_size,
        n_total_env_steps,
    )

With the arguments

* ``buffer``: A replay buffer containing transitions
* ``n_updates``: Desired number of updates
* ``model_batch_size`` and ``batch_size``: Batch sizes for dynamics model and other models respectively
* ``n_total_env_steps``: Number of environment steps passed in all environments collectively. Relevant for learning rate schedules.


API Reference
-------------

For detailed API documentation, see:

* :doc:`../api/models`
* :doc:`../api/predictors`
* :doc:`../api/nets`
