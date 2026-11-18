"""Constants shared by the Table 1 hyperparameter search (ISEM 735, task 29).

research/hparam_torch.py (GRU, Transformer) and research/hparam_xgb.py (XGBoost) run in separate
processes because torch and XGBoost cannot share one, so this module imports neither. The
protocol is in research/hparam_notes.md section 1.
"""

from __future__ import annotations

import json

import run_pipeline as rp

N_TRIALS = 50
SAMPLER_SEED = 42
TRAIN_SEED = 42  # torch trials and XGBoost trials during the search
PRUNER_STARTUP, PRUNER_WARMUP = 10, 5
OUT_DIR = rp.PROJECT_ROOT / "data" / "processed" / "seed_predictions_tuned"

# Trial 0 for every model: the settings Table 1 used (tasks 7, 13, 24 and src/modeling.py).
DEFAULTS = {
    "XGBoost": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9,
                "colsample_bytree": 0.9, "min_child_weight": 1.0, "reg_lambda": 1.0},
    "GRU": {"embed_dim": 32, "hidden_dim": 64, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-5,
            "batch_size": 256},
    "Transformer": {"dim": 64, "heads": 4, "layers": 2, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-5,
                    "batch_size": 256},
}
TRAINING_KEYS = ("lr", "weight_decay", "batch_size")  # passed to train(); the rest go to the constructor


def suggest(trial, model: str) -> dict:
    """Draw one configuration from the section 1 search space."""
    if model == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True)}
    common = {"dropout": trial.suggest_float("dropout", 0.0, 0.5),
              "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
              "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512])}
    if model == "GRU":
        return {"embed_dim": trial.suggest_categorical("embed_dim", [16, 32, 64]),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]), **common}
    if model == "Transformer":
        return {"dim": trial.suggest_categorical("dim", [32, 64, 128]),
                "heads": trial.suggest_categorical("heads", [2, 4]),
                "layers": trial.suggest_categorical("layers", [1, 2, 3]), **common}
    raise ValueError(model)


def make_study(model: str):
    import optuna

    pruner = (optuna.pruners.NopPruner() if model == "XGBoost"
              else optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP, n_warmup_steps=PRUNER_WARMUP))
    study = optuna.create_study(direction="minimize", study_name=f"task29_{model}",
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED), pruner=pruner)
    study.enqueue_trial(DEFAULTS[model])
    return study


def predictions_path(model: str, seed: int):
    return OUT_DIR / f"{model.lower()}_seed{seed}.parquet"


def best_params_path(model: str):
    return OUT_DIR / f"best_{model.lower()}.json"


def trials_path(model: str):
    return rp.TABLES_DIR / f"hparam_trials_{model.lower()}.csv"


def save_study(study, model: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials_path(model).parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_path(model), index=False)
    best = {"params": study.best_params, "val_log_loss": study.best_value, "trial": study.best_trial.number,
            "default_val_log_loss": study.trials[0].value}
    best_params_path(model).write_text(json.dumps(best, indent=2))
    return best


def load_best(model: str) -> dict:
    return json.loads(best_params_path(model).read_text())["params"]
