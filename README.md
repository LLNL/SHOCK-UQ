# Overview

This repository contains material for an uncertainty quantification (UQ) class on equation of state (EOS) that was presented at Science of Compression in Condensed Matter (SCCM) 2025. It includes a Python package called eosuq for analyzing linear shock compression data. The package performs Bayesian linear regression analytically and with Markov Chain Monte Carlo, and provides a bootstrap approach for comparison. The methods are demonstrated in a notebook on publicly available data contained in the notebooks directory. The dataset, from shock compression experiments on copper, is from pages 57-60 of Marsh, S. P. (1980), "LASL Shock Hugoniot Data".

## Getting Started

1. **Clone the repository**
```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
```

2. **Create a virtual environment**
```bash
    python3 -m venv .venv
```

3. **Activate the virtual environment**
```bash
    source .venv/bin/activate
```

4. **Install dependencies**
```bash
    pip install -r requirements.txt
```

5. **Install the `eosuq` package locally**
```bash
    pip install -e .
```

6. **Launch Jupyter notebook** The notebook is located at `notebooks/data_analysis.ipynb`.

## Contributors

- Jason Bernstein (Lawrence Livermore National Laboratory)
- Justin Brown (Sandia National Laboratory)
- Beth Lindquist (Los Alamos National Laboratory)

## License

This software is distributed under the terms of the MIT license.  All new contributions must be made under the MIT license.

See Link to LICENSE and NOTICE for details.

## Release

LLNL-CODE-2005336
