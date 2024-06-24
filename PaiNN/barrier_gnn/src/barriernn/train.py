import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay, CosineDecay
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    CosineDecay,
    LearningRateSchedule,
)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow_addons as tfa
from barriernn.utils import (
    CosineLearningRateScheduler,
    ExponentialLearningRateScheduler,
)
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError
from barriernn.input_generation import create_meta_dataset, create_meta_dataset_predictions
from barriernn.PAiNN_adapted import make_model
from pathlib import Path
from itertools import repeat
import json


def compile_scaled_model(mean=0, std=1, decay_steps=1000):
    def compile_model(hparas) -> Model:
        lr = hparas["lr_start"]
        if hparas.get("lr_scheduler") == "cos":
            lr = CosineDecay(
                hparas["lr_start"],
                (decay_steps * 10) / hparas["batchsize"],
                alpha=hparas["lr_fraction"],
            )
        elif hparas.get("lr_scheduler") == "exp":
            lr = ExponentialDecay(
                hparas["lr_start"],
                int(decay_steps / hparas["batchsize"]),
                decay_rate=hparas["lr_decay"],
            )

        optimizer = Adam(learning_rate=lr)

        out_mlp = None
        if "out_mlp" in hparas:
            out_mlp = hparas["out_mlp"]

        if out_mlp is None:
            if hparas.get("mlp_style") == "static":
                assert all([n in hparas for n in ("mlp_layers", "mlp_size")]),\
                    "mlp_layers and mlp_size must be provided with mlp_style=shrinking"
                out_mlp = {
                    "use_bias": list(repeat(True, hparas["mlp_layers"] + 1)),
                    "units": list(repeat(hparas["mlp_size"], hparas["mlp_layers"])) + [1],
                    "activation": list(repeat("swish", hparas["mlp_layers"])) + ["linear"],
                }
            elif hparas.get("mlp_style") == "shrinking":
                assert "mlp_layers_sh" in hparas, "mlp_layers_sh must be provided with mlp_style=shrinking"
                out_mlp = {
                    "use_bias": list(repeat(True, hparas["mlp_layers_sh"] + 1)),
                    "units": list(2**(np.arange(6,hparas["mlp_layers_sh"]+6)[::-1])) + [1],
                    "activation": list(repeat("swish", hparas["mlp_layers_sh"])) + ["linear"],
                }

        model : Model= make_model(
            output_embedding=hparas.get("out_emb"),
            output_mlp=out_mlp,
            depth=hparas.get("depth"),
            pooling_args={"pooling_method": hparas.get("pooling")},
            equiv_normalization=hparas.get("equiv_norm"),
            node_normalization=hparas.get("node_norm"),
            mlp_rep=hparas.get("mlp_rep"),
        )

        metrics = ["mean_absolute_error"]
        if hparas.get("scale"):
            scaler = ScaledMeanAbsoluteError()
            scaler.set_scale(std)
            metrics.append(scaler)
        
        loss = hparas.get("loss")
        if loss is None:
            loss="mean_absolute_error"
        if loss == "MALE":
            from barriernn.utils import MALE
            loss = MALE

        model.compile(
            loss=loss,
            # loss="mean_squared_error",
            optimizer=optimizer,
            metrics=metrics,
        )
        return model

    return compile_model

def make_callbacks(tb, early_stopping, log_dir, run_name):
    callbacks = []
    log_dir.mkdir(exist_ok=True)
    if tb:
        callbacks.append(
            TensorBoard(
                f"{log_dir}/tb_{run_name}",
                update_freq="epoch",
                write_graph=True,
                histogram_freq=5,
                embeddings_freq=0,
                profile_batch=0,
            )
        )
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                "val_loss",
                min_delta=0.01,
                patience=30,
                restore_best_weights=True,
            )
        )
    return callbacks

def prep_training(
    meta_files: list,
    hparas: dict,
    log_dir: Path,
    run_name: str,
    tb=True,
    early_stopping=False,
):

    if (p := hparas.get("cache")) is not None:
        for f in Path(p).glob("cache*"):
            f.unlink()
        cache = Path(p) / "cache"
    else:
        cache = None

    old_scale = None
    max_dist = None
    min_dist = None
    descriptors = None
    
    if "old_scale" in hparas:
        old_scale = hparas["old_scale"]
    if "max_dist" in hparas:
        max_dist = hparas["max_dist"]
    if "min_dist" in hparas:
        min_dist = hparas["min_dist"]
    if "descriptors" in hparas:
        descriptors = hparas["descriptors"]

    train_ds, val_ds, scale_t, n_data = create_meta_dataset(
        meta_files,
        val_split=hparas["val_split"],
        batch_size=hparas["batchsize"],
        cache=cache,
        scale=hparas.get("scale"),
        old_scale=old_scale,
        max_dist=max_dist,
        min_dist=min_dist,
        opt=hparas["opt"],
        descriptors=descriptors
    )
    decay_steps = n_data * hparas["epochs"] / 10

    if isinstance(hparas, dict):
        if old_scale is None:
            log_dir.mkdir(exist_ok=True)
            if not (log_dir / "hparas.json").exists():
                with open(log_dir / "hparas.json", "x") as f:
                    json.dump(hparas, f, skipkeys=True, indent=2)

                with open(log_dir / "scale", "x") as f:
                    f.writelines(list(map(lambda s: s + "\n", map(str, scale_t))))
        
        hparas["old_scale"] = scale_t  # used in subsequent runs
    
    callbacks = make_callbacks(tb, early_stopping, log_dir, run_name)

    return train_ds, val_ds, callbacks, scale_t, decay_steps


def mod_model(model, fresh_model, hparas):

    if hparas.get("freeze_graph"):
        for layer in model.layers[:-1]:
            layer.trainable = False
    
    if hparas.get("new_mlp"):
        new_mlp = fresh_model.layers[-1]
        new_out = new_mlp(model.layers[-1].input)
        model = Model(inputs=model.inputs, outputs=new_out)
    
    model.compile(
        loss=fresh_model.loss,
        optimizer=fresh_model.optimizer,
        metrics=fresh_model.metrics,
    )
    return model


def train(
    meta_files: list,
    hparas: dict,
    log_dir: Path,
    run_name: str = None,
    tb=True,
    early_stopping=False,
    save=True,
    mod_model_p: Path=None,
    old_model=None,
    old_clbks=None,
    val_meta_files=None,
    old_val_ds=None,
    old_train_ds=None,
):
    if run_name is None:
        run_name = log_dir.name
    assert not ((val_meta_files is not None) and (old_val_ds is not None))
    assert (old_train_ds is None) == (old_val_ds is None), "old val and old train must be given!"

    # New model
    if all([i is None for i in [old_model, old_clbks]]):
        train_ds, val_ds, callbacks, scale, decay_steps = prep_training(
            meta_files, hparas, log_dir, run_name, tb, early_stopping
        )
        hparas["initial_epoch"] = 0

        # Separat set of meta files for validation
        if val_meta_files is not None:
            assert (
                hparas["val_split"] == 0.0
            ), "ERROR: Using separate val meta files, but val_split is not 0!"

            hcopy = hparas.copy()
            if hcopy["cache"] is not None:
                hcopy["cache"] = hcopy["cache"] + "_val"
                for p in Path(hcopy["cache"]).glob("cache*"):
                    p.unlink()
            val_ds, _, _, _, _ = prep_training(
                val_meta_files, hcopy, log_dir, run_name, tb=False, early_stopping=False
            )

        if mod_model_p is None:
            model = compile_scaled_model(*scale, decay_steps)(hparas)
        else:
            print("Using model from", mod_model_p)
            fresh_model = compile_scaled_model(*scale, decay_steps)(hparas)
            old_model = load_model(mod_model_p)
            model = mod_model(old_model, fresh_model, hparas)



    # Retrain model:
    elif None not in [old_model, old_clbks]:
        model, callbacks = old_model, old_clbks
        hparas["epochs"] = hparas["epochs"] + hparas["initial_epoch"]
        with open(log_dir / "scale", "r") as f:
            scale = [float(s.strip()) for s in f.readlines()]

        if (old_train_ds is None) and (old_val_ds is None):
            train_ds, val_ds, _, _, _ = prep_training(
                meta_files, hparas, log_dir, run_name, tb=False, early_stopping=False
            )   
        else:
            train_ds = old_train_ds
            val_ds = old_val_ds

        with open(log_dir / "hparas_retrain.json", "a") as f:
            json.dump(hparas, f, skipkeys=True, indent=2)

    else:
        raise ValueError("For retraining, model and callbacks must be given!")

    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=hparas["epochs"],
        callbacks=callbacks,
        verbose=2,
        initial_epoch=hparas["initial_epoch"],
    )
    if save:
        model.save(log_dir / (run_name + ".tf"))
    # clean cache
    if (cache := hparas.get("cache")) is not None:
        for p in Path(cache).glob("cache*"):
            p.unlink()

    return train_ds, val_ds, model, callbacks


def eval(eval_hparas, meta_files, model, log_dir, callbacks=[], out_name="eval"):
    # Evaluation after training on different settings
    if (p := eval_hparas.get("cache")) is not None:
        cache = Path(p) / "cache"
        for p in Path(cache).glob("cache*"):
            p.unlink()
    else:
        cache = None

    test_ds, energies, scale_t, meta_d, metas_masked = create_meta_dataset_predictions(
        meta_files=meta_files,
        batch_size=eval_hparas.get("batchsize"),
        scale=eval_hparas.get("old_scale"),
        max_dist=eval_hparas.get("max_dist"),
        min_dist=eval_hparas.get("min_dist"),
        opt=eval_hparas.get("opt"),
        descriptors=eval_hparas.get("descriptors"),
    )

    print("predicting...", end="")
    e_p = model.predict(test_ds)[:, 0]
    e_p = (e_p * scale_t[1]) + scale_t[0]
    energies = (energies * scale_t[1]) + scale_t[0]
    errors = (e_p - energies)
    mae = {"mean_absolute_error" : float(np.abs(errors).mean())}
    print(" done! MAE: ", mae)

    with open(log_dir / out_name, "a") as f:
        json.dump(eval_hparas, f, skipkeys=True, indent=2)
        f.write("\n")
        json.dump(mae, f, skipkeys=True)
        f.write("\n##################\n")

    # clean cache
    if (cache := eval_hparas.get("cache")) is not None:
        for p in Path(cache).glob("cache*"):
            p.unlink()

    return errors, meta_d