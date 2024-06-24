from barriernn.train import prep_training, compile_scaled_model
from pathlib import Path
from barriernn import gpus
import tensorflow as tf

try:
    import keras_tuner as kt
except ImportError as e:
    print("Error: keras tuner could not be imported, make sure it is installed!")


def hpara_training(
    meta_files: list,
    hparas: kt.HyperParameters,
    log_dir: Path,
    run_name: str,
    tb=True,
    early_stopping=False,
    max_trials=50,
    num_initial_points=5
):
    train_ds, val_ds, callbacks, scale, decay_steps = prep_training(
        meta_files, hparas, log_dir, run_name, tb, early_stopping
    )

    strat = None
    # Might influence performance!
    # if len(gpus) > 1:
    #     print(f"Using {len(gpus)} GPUS with mirrored strategy")
    #     strat = tf.distribute.MirroredStrategy([f"/gpu:{i}" for i in range(len(gpus))])

    # tuner = kt.RandomSearch(
    # tuner = kt.BayesianOptimization(
    #     compile_scaled_model(*scale),
    #     objective="val_loss",
    #     max_trials=max_trials,
    #     num_initial_points=num_initial_points,
    #     hyperparameters=hparas,
    #     alpha=1e-1,
    #     beta=2.6,
    #     distribution_strategy=strat,
    #     directory=f"{log_dir}/{run_name}",
    # )

    tuner = kt.Hyperband(
        hypermodel=compile_scaled_model(*scale, decay_steps),
        objective="val_loss",
        max_epochs=500,
        factor=3,
        hyperband_iterations=3,
        seed=None,
        hyperparameters=hparas,
        tune_new_entries=True,
        allow_new_entries=True,
        directory=f"{log_dir}/{run_name}",
    )


    tuner.search(
        train_ds, validation_data=val_ds, epochs=hparas["epochs"], callbacks=callbacks,
    )
    

    tuner.results_summary()
    