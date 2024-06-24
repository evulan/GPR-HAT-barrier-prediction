"""Function to create a single dataframe with all PaiNN predictions"""

import numpy as np
import pandas as pd
from pathlib import Path


def get_pretrained_linear_predictions():
    """Get the ensemble predictions for all seeds"""
    cwd = Path(__file__).resolve().parent
    project_root = cwd.parent

    # PaiNN prediction files to load
    path_painn_results_dir = (
        project_root / "PaiNN" / "usage" / "results" / "two_stage_learning"
    )
    painn_ranks = np.sort(
        [
            int(res_file.stem.split("df_pred_rank_")[-1])
            for res_file in path_painn_results_dir.glob(f"df_pred_rank_*.pkl")
        ]
    )
    paths_painn_dfs = [
        pd.read_pickle(path_painn_results_dir / f"df_pred_rank_{rank}.pkl")
        for rank in painn_ranks
    ]
    paths_painn_logs = [
        np.load(path_painn_results_dir / f"log_rank_{rank}.npz", allow_pickle=True)
        for rank in painn_ranks
    ]

    assert len(paths_painn_dfs) > 0 and len(paths_painn_logs) > 0

    hash_direction_order = paths_painn_dfs[0].index.to_numpy().copy()
    seeds = np.array(paths_painn_logs[0]["log"].item()["sample_seed"])

    # Load the results and save them into one numpy array
    Y_painn = np.zeros(
        (seeds.size, painn_ranks.size, hash_direction_order.size)
    )  # seed, rank, val

    for i_rank, rank in enumerate(painn_ranks):
        log = paths_painn_logs[i_rank]["log"].item()
        assert (
            len(
                np.setdiff1d(
                    paths_painn_dfs[i_rank].index.to_numpy(), hash_direction_order
                )
            )
            == 0
        )
        df_rank = paths_painn_dfs[i_rank].loc[hash_direction_order].copy()

        # Sometimes the first seed still has the old column name. If this is the case rename it for consistency
        if "GNN_E_barrier_predicted" in df_rank.columns:
            df_rank.rename(
                columns={
                    "GNN_E_barrier_predicted": f"E_barrier_predict-{log['hashid'][0]}"
                },
                inplace=True,
            )

        # Save the predictions for the correct seed and rank
        for i_seed, seed in enumerate(seeds):
            print(i_rank, i_seed, flush=True)
            hashid = log["hashid"][i_seed]
            select_column = f"E_barrier_predict-{hashid}"
            E_predict = df_rank[select_column].to_numpy()
            Y_painn[i_seed, i_rank, :] = E_predict

    return Y_painn, hash_direction_order, (seeds, painn_ranks)


def get_ensemble_predictions(seed_select=0, load_saved=True):
    """Return the PaiNN predictions for a selected seed. Create file for faster access"""

    # File to save the ensemble predictions to
    save_file = (
        Path(__file__).resolve().parent
        / "results"
        / f"ensemble_predictions_{seed_select}.pkl"
    )

    if not load_saved or not save_file.exists():
        # Load all PaiNN unoptimized predictions
        Y_painn, hash_direction_order, _ = get_pretrained_linear_predictions()
        print(seed_select, Y_painn.shape, flush=True)

        # Ensemble mean
        Y_predict_ensemble = np.mean(Y_painn[seed_select], axis=0)

        transition_hashes = [x.split("_")[0] for x in hash_direction_order]
        directions = [x.split("_")[1] for x in hash_direction_order]

        # Create dataframe with the ensemble unoptimized barrier predictions
        df_predict = pd.DataFrame(
            data={
                "hash_direction": hash_direction_order,
                "transition_hash": transition_hashes,
                "direction": directions,
                "E_predict": Y_predict_ensemble,
            }
        ).set_index("hash_direction")
        df_predict.to_pickle(save_file)
    else:  # if already saved, no need to recompute
        print("Load saved")
        df_predict = pd.read_pickle(save_file)
    return df_predict


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2, "No rank passed"
    rank = int(sys.argv[1])
    print(f"Rank {rank} started")

    df = get_ensemble_predictions(seed_select=rank, load_saved=True)
    print(df)
