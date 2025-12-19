Installation
============

Basic Installation
------------------

We recommend installing LayeredRL into a separate environment. With `uv <https://github.com/astral-sh/uv>`_ creating a new virtual environment and installing LayeredRL from source can be done like this:

.. code-block:: bash

    git clone https://github.com/nicoguertler/layeredrl.git
    cd layeredrl
    uv venv
    uv pip install .

This will write the venv to the path ``.venv/``. You can alternatively specify a path with ``uv venv path-to-venv``.

Without uv, creating a venv and installing the package can be achieved like this (in the root directory of the repository):

.. code-block:: bash

    python -m venv path-to-venv
    . path-to-venv/bin/activate
    pip install .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

    uv pip install -e ".[dev]"

This includes testing tools and pre-commit hooks.

Verification
------------

Verify your installation by running:

.. code-block:: python

    import layeredrl
    print(layeredrl.__version__)

Or run the test suite:

.. code-block:: bash

    pytest tests/
