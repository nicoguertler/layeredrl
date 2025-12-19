# LayeredRL Documentation

This directory contains the Sphinx documentation for LayeredRL.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
# From the repository root
pip install -e ".[docs]"

# Or install directly from docs/requirements.txt
pip install -r docs/requirements.txt
```

### Building Locally

From the `docs/` directory:

```bash
# Build HTML documentation
make html

# View the documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
start build/html/index.html  # Windows
```

Other build formats:

```bash
make pdf      # Build PDF (requires LaTeX)
make epub     # Build EPUB
make latexpdf # Build PDF via LaTeX
make clean    # Clean build directory
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Main documentation page
│   ├── installation.rst        # Installation guide
│   ├── quickstart.rst          # Quick start tutorial
│   ├── user_guide/             # User guides
│   │   ├── hierarchies.rst
│   │   ├── levels.rst
│   │   ├── planning.rst
│   │   ├── models.rst
│   │   └── collectors.rst
│   ├── api/                    # API reference (auto-generated)
│   │   ├── hierarchies.rst
│   │   ├── levels.rst
│   │   ├── planners.rst
│   │   └── ...
│   ├── examples/               # Examples and tutorials
│   │   └── gallery.rst
│   ├── citation.rst            # Citation information
│   └── contributing.rst        # Contribution guidelines
├── Makefile                    # Build commands (Unix)
├── make.bat                    # Build commands (Windows)
└── requirements.txt            # Documentation dependencies
```

## Read the Docs

The documentation is configured for Read the Docs via `.readthedocs.yaml` in the repository root.