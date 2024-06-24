"""Train PaiNN models as a basis for two stage learning with SOAP"""

from pathlib import Path
from barriernn import train, eval
from barriernn.input_generation import create_meta_dataset_predictions
import time
import random
import helpers
import numpy as np
import sys
import uuid

# Each rank (here the job array ID passed as the second argument) will train one expert of the ensemble
assert len(sys.argv) == 2, "No rank passed"
rank = int(sys.argv[1])
print(f"Rank {rank} started")

cwd = Path(__file__).resolve().parent
project_root = cwd.parents[1]


def train_model(*, base_name, train_datasets, eval_datasets, base_name_dir, seed):
    """Trains a PaiNN model"""
    data_root = project_root / "data" / "pdb"

    log_base_dir = output_folder / base_name_dir
    run_name = "run"
    log_dir = log_base_dir / run_name
    model_dir = None
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"# SAVING in {log_dir.name} #")

    # Dir for TF model cache
    cache_dir = output_folder / f"cache_{base_name}"
    if not cache_dir.exists():
        cache_dir.mkdir()

    print(f"# SAVING in {log_base_dir} #")

    # Search for training metafiles, i.e. files with the metadata of the structure data
    print(f"Searching meta files..  ", end="")
    meta_files = [
        i
        for d in train_datasets
        for i in data_root.glob(f"{d}/train/[0-9]*_*[0-9].npz")
    ]
    print(f"Found {len(meta_files)} meta files", end="")
    random.Random(seed).shuffle(meta_files)

    # Filter out those files with structures which are not part of the data/atoms.pkl pickle file
    # Do not train on the optimized structures
    meta_files, _ = helpers.filter_metafiles(meta_files, remove_opt=True)

    print(f"n_total_train: {len(meta_files)}")
    print(f", using {len(meta_files)}")

    # Hyperparameters used for training
    hparas = {
        "loss": "MAE",
        "lr_start": 8e-4,
        "lr_scheduler": "cos",
        "lr_fraction": 1e-2,
        "out_emb": "poi",
        "mlp_style": "static",
        "mlp_layers": 2,
        "mlp_size": 128,
        "mlp_rep": 1,
        "depth": 2,
        "equiv_norm": False,
        "node_norm": False,
        "pooling": "sum",
        "epochs": 200,
        "batchsize": 128,
        "cache": str(cache_dir),
        "val_split": 0.1,
        "scale": False,
        "max_dist": None,
        "min_dist": None,
        "opt": False,
        "datasets": " ".join(train_datasets),
        "date": time.ctime(),
        "initial_epoch": 0,
        "freeze_graph": False,
        "new_mlp": False,
    }

    train_ds, val_ds, model, callbacks = train(
        meta_files,
        hparas=hparas.copy(),
        log_dir=log_dir,
        run_name=run_name,
        mod_model_p=model_dir,
        tb=False,
        early_stopping=True,
        save=False,
    )

    test_files = [
        i for d in eval_datasets for i in data_root.glob(f"{d}/test/[0-9]*_*[0-9].npz")
    ]
    test_files = test_files + [
        i
        for d in train_datasets
        for i in data_root.glob(f"{d}/train/[0-9]*_*[0-9].npz")
    ]

    test_files, _ = helpers.filter_metafiles(test_files)

    test_ds, energies, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
        meta_files=test_files, batch_size=128, opt=False
    )

    predictions = model.predict(test_ds).squeeze()
    deltas = predictions - energies
    df_d = helpers.meta_d_to_df_d(meta_ds, deltas.flatten(), metas_masked=metas_masked)

    cache_dir.rmdir()
    print(f"\n\n {run_name} done at {time.ctime()} \n\n")
    return df_d


ds_s = ["synth"]
ds_t = ["traj"]
datasets = ds_s + ds_t

print("Run train all")
# Seeds to be used.
sample_seeds = [0, 1, 2, 3, 4, 5, 6, 7]

output_folder = Path(__file__).resolve().parent / "models" / "two_stage_learning"

df_pred = None  # This will save the predictions
train_datasets = datasets
test_datasets = datasets

log = {
    "sample_seed": [],
    "rank": [],
    "hashid": [],
}
# Redo training with each seed and save the results
for j, sample_seed in enumerate(sample_seeds):
    print(f"Rank ({rank}) | seed: {sample_seed}", flush=True)
    hashid = f"<<<{uuid.uuid4()}>>>{rank}"
    base_name_dir = f"rank_{rank}_seed_{sample_seed}"
    base_name = f"rank_{rank}_seed_{sample_seed}"

    # Train the PaiNN model and save the predictions in a dataframe
    df_all = train_model(
        base_name=base_name,
        train_datasets=train_datasets,
        eval_datasets=test_datasets,
        base_name_dir=base_name_dir,
        seed=sample_seed,
    )
    df_all["hash_direction"] = [
        f'{df_all["transition_hash"].iloc[i]}_{df_all["direction"].iloc[i]}'
        for i in range(df_all.shape[0])
    ]

    # Collect the predictions for every seed in a single dataframe
    if df_pred is None:
        df_pred = df_all.copy().set_index(["hash_direction"])
        df_pred.rename(
            columns={
                "GNN_E_barrier_predicted": f"E_barrier_predict-{hashid}",
            },
            inplace=True,
        )
    else:
        df_all = df_all.copy().set_index(["hash_direction"])
        assert len(np.setdiff1d(df_all.index.to_numpy(), df_pred.index.to_numpy())) == 0
        assert np.all(np.isin(df_all.index.to_numpy(), df_pred.index.to_numpy()))
        E_ref = np.array(
            df_pred.loc[df_all.index]["E_barrier"].to_numpy(), dtype=np.float64
        )
        E_test = np.array(df_all["E_barrier"].to_numpy(), dtype=np.float64)
        assert np.allclose(E_ref, E_test)
        E_pred = np.ascontiguousarray(
            np.array(df_all["GNN_E_barrier_predicted"], dtype=np.float64)
        )
        df_pred.loc[df_all.index, f"E_barrier_predict-{hashid}"] = E_pred

    log["sample_seed"].append(sample_seed)
    log["rank"].append(rank)
    log["hashid"].append(hashid)

    print(f"Done Rank {rank}; data sample {sample_seed}")

print("Save")
df_pred.to_pickle(cwd / "results" / "two_stage_learning" / f"df_pred_rank_{rank}.pkl")
# Save this rank's predictions. Note that each rank's prediction is equivalent one ensemble member later
np.savez(cwd / "results" / "two_stage_learning" / f"log_rank_{rank}.npz", log=log)
