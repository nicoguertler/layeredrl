Contributing
============

Thank you for your interest in LayeredRL! The library is actively developed as a research project, but maintenance time is
limited. Contributions are welcome, in particular bug reports and small fixes. For requesting/adding new features,
please open a discussion first.

Development Setup
-----------------

1. Fork and clone the repository:

.. code-block:: bash

    git clone https://github.com/nicoguertler/layeredrl.git
    cd layeredrl

2. Set up and activate virtual environment

3. Install in development mode:

.. code-block:: bash

    pip install -e ".[dev]"

4. Install pre-commit hooks:

.. code-block:: bash

    pre-commit install

Running Tests
-------------

Run the test suite to ensure everything works:

.. code-block:: bash

    pytest tests/

Code Style
----------

LayeredRL uses:

* **Black** for code formatting
* **flake8** for linting

The pre-commit hooks will automatically format your code. You can also run manually:

.. code-block:: bash

    black layeredrl/ tests/
    flake8 layeredrl/ tests/

Pull Requests
^^^^^^^^^^^^^

**For bug fixes and small improvements**: Please submit a PR following the checklist below.

**For new features**: Best to open a discussion first to make sure integrating the feature is feasible.

Pull Request Checklist:

* [ ] Tests pass locally
* [ ] Code follows style guidelines (pre-commit hooks pass)
* [ ] Documentation updated (if applicable)
* [ ] Added tests for new functionality
* [ ] PR description explains the changes

Adding a New Level Type
-----------------------

Checkout the :doc:`user_guide/levels` page for instructions on how to implement a new level.


Documentation
-------------

Building Documentation Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build and view the documentation:

.. code-block:: bash

    cd docs/
    make html
    open build/html/index.html  # or xdg-open on Linux

The documentation uses Sphinx with the Read the Docs theme.

Code of Conduct
---------------

Be respectful and constructive.
