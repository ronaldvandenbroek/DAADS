import pandas as pd
import multiprocessing as mp
import pathlib

from evaluate import aggregate_dataframe, test_then_train

# N_PROCESSES = 3
# Increasing N_Processes to make use of more cores
N_PROCESSES = 12

DATASETS = ["covertype", "creditcard", "shuttle"]

# Original
# MODELS = ["AE", "AE", "DAE", "RRCF", "HST", "PW-AE", "xStream", "Kit-Net", "ILOF"]

# Double "AE" seems to cause a problem with a missing:
# FileNotFoundError: [Errno 2] No such file or directory: '/home/ronald/river_data/CreditCard/creditcardfraud.zip'
# Possible interfering process threads
# MODELS = ["AE", "DAE", "RRCF", "HST", "PW-AE", "xStream", "Kit-Net", "ILOF"]

# Running only new models to compare them with original paper
# MODELS = ["AE", "DAE", "PW-AE"]

# Running only the baselines (excluding RRCF due to long runtime, excluding ILOF due to freezing code)
# RRCF on local hardware 6.5 hours per each of the 30 settings
MODELS = ["HST", "xStream", "Kit-Net"]

# Running full benchmark (excluding RRCF and ILOF)
# MODELS = ["AE", "DAE", "PW-AE", "HST", "xStream", "Kit-Net"]

# Original seed
SEEDS = range(42, 52)
# SEEDS = range(22, 32)

SUBSAMPLE = 500_000

SAVE_STR = "Benchmark"

CONFIGS = {
    "AE": {"lr": 0.02, "latent_dim": 0.1},
    "DAE": {"lr": 0.02},
    "PW-AE": {"lr": 0.1},
    "OC-SVM": {},
    "HST": {"n_trees": 25, "height": 15},
}


pool = mp.Pool(processes=N_PROCESSES)
runs = [
    pool.apply_async(
        test_then_train,
        kwds=dict(
            dataset=dataset,
            model=model,
            seed=seed,
            subsample=SUBSAMPLE,
            **CONFIGS.get(model, {}),
        ),
    )
    for dataset in DATASETS
    for model in MODELS
    for seed in SEEDS
]

metrics = [run.get()[0] for run in runs]

metrics_raw = pd.DataFrame(metrics)
metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])

path = pathlib.Path(__file__).parent.parent.resolve()
metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
