# QSVM-EEG

Project structure

```shell
.
├── data
│   └── raw
│       ├── patient48_bis.csv
│       ├── patient48_eeg.csv
│       └── patient58_eeg.csv
├── main.py  # Run experiment
├── notebooks  # Experiments notebooks
│   ├── classical_48.ipynb
│   ├── quantum_48.ipynb
│   └── shared
│       └── tu.ipynb
├── pyproject.toml
├── qsvm_eeg  # Quantum kernel implementation
│   ├── circuit.py
│   ├── data.py
│   ├── features.py
│   ├── __init__.py
├── README.md
├── reports
│   ├── figures
│   │   ├── correlation_plot_20260122_170200.png
│   │   └── prediction_vs_actual_20260122_170159.png
│   └── logs
│       └── experiment_log.csv
├── requirements.txt
└── uv.lock
```

## Prerequisites

- Python version >= 3.12

## Experiments

Run script `main.py`. Set the sub-sample by edit the `SAMPLE_LIMIT`
in `main.py` e.g.

```python
SAMPLE_LIMIT = 200
```

Or use the full dataset by set `SAMPLE_LIMIT` to `None`.

Experiments logs and plots will be generated to `reports` directory.
