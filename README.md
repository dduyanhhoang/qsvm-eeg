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

Run script `main.py`. 

```shell
python main.py -h
usage: main.py [-h] [-p PATIENTS [PATIENTS ...]] [-n SAMPLES] [-j JOBS]

Run Kernel-based QSVM Experiment on EEG Data.

options:
  -h, --help            show this help message and exit
  -p PATIENTS [PATIENTS ...], --patients PATIENTS [PATIENTS ...]
                        List of Patient IDs to use. Default: ['48', '411', '58']
  -n SAMPLES, --samples SAMPLES
                        Total number of samples to use (distributed equally among patients). Default: Use all data.
  -j JOBS, --jobs JOBS  Number of CPU cores for Kernel computation. Default: -1 (All cores).
```

Example

```shell
uv run main.py -p 48 -n 500 -j 2
```

The default run uses all available patients' data samples, all available processors.

```shell
python main.py
```

Experiments logs and plots will be generated to `reports` directory.
