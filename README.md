# RTG-HMC (2025) – Proton-like Wave-Function Resonance Simulator

The **RTG-HMC** project implements a Hamiltonian Monte Carlo approach for
modeling proton-like wave-function resonances.  It is a reference
implementation used in exploring concepts from [Relational Time
Geometry](https://rtgtheory.com).

## Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd RTG

# (Optional) create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and run the simulator
pip install -r requirements.txt
python simulate.py --steps 1000 --seed 42 --output results.h5
```

## Command-Line Flags

| Flag         | Description                                  |
|--------------|----------------------------------------------|
| `--steps`    | Number of integration steps to perform.      |
| `--seed`     | Random seed controlling stochastic behavior. |
| `--output`   | Path to save the resulting data.              |
| `--help`     | Show additional command usage information.   |

## Repository Layout

```
/ (project root)
├── LICENSE            # MIT license
├── README.md          # Project overview (this file)
├── requirements.txt   # Python dependencies
├── simulate.py        # Entry point for the simulator
└── docs/              # Additional documentation
```

## License

The project is released under the MIT License. See
[`LICENSE`](LICENSE) for more information.
