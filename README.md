# QSVM-EEG

Project structure

```shell
.
├── README.md
├── quantum.py  # Run kernel-based QSVR experiments
├── data
│   └── raw
│       ├── patient411_bis.csv
│       ├── patient411_eeg.csv
│       ├── patient48_bis.csv
│       ├── patient48_eeg.csv
│       └── patient58_eeg.csv
├── pyproject.toml
├── qsvm_eeg
│   ├── __init__.py
│   ├── circuit.py
│   ├── data.py
│   └── features.py
├── reports
│   ├── figures
│   │   ├── corr_Mix_48_411_20260129_054055.png
│   │   ├── corr_Mix_48_411_20260129_054837.png
│   │   ├── pred_actual_Mix_48_411_20260129_054054.png
│   │   └── pred_actual_Mix_48_411_20260129_054837.png
│   └── logs
│       └── experiment_log.csv
├── requirements.txt
└── uv.lock
```

## Prerequisites

- Python version >= 3.12
- `requirements.txt`
  - Using `lightning.gpu` device for simulation.

```shell
pip install -r requirements.txt
```

## Experiments

Running experiments using `quantum.py`. 

```shell
python quantum.py -h
usage: quantum.py [-h] [-p PATIENTS [PATIENTS ...]] [-n SAMPLES] [-j JOBS]

Run Kernel-based QSVM Experiment on EEG Data.

options:
  -h, --help            show this help message and exit
  -p PATIENTS [PATIENTS ...], --patients PATIENTS [PATIENTS ...]
                        List of Patient IDs to use. Default: ['48', '411']
  -n SAMPLES, --samples SAMPLES
                        Total number of samples to use (distributed equally among patients). Default: Use all data.
  -j JOBS, --jobs JOBS  Number of CPU cores for Kernel computation. Default: 8 (8 cores).
```

Example

The default run uses all available patients' data samples, using 8 cores/processors.

```shell
python quantum.py
```

Custom run with patient ID, number of samples, number of jobs e.g. patient 48, using 500 samples only, 2 jobs

```shell
python quantum.py -p 48 -n 500 -j 2
```

Experiments logs and plots will be generated to `reports` directory.
