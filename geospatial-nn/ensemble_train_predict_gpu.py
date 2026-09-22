##########################################################
# Neural Network Ensemble for BVB Regime Inference (GPU)
##########################################################

from pathlib import Path
import random
from joblib import dump

import numpy as np
import xarray as xr
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Global Variable Configuration
SEED = 42
NUM_MEMBERS = 50

BATCH_SIZE = 4096 * 2
PREDICT_BATCH_SIZE = 65536
EPOCHS = 250
LEARNING_RATE = 1e-4
PATIENCE = 10

DATA_RES = "p25"
SCENARIO = "tm"

BVB_TERMS = ["beta_V", "BPT", "Mass_flux", "eta_dt",
             "Curl_dudt", "Curl_taus", "Curl_taub", "Curl_Adv", "Curl_diff"]


# Device Configuration
def configure_device():
    """
    Enable memory growth on visible GPUs and return the device to run on.
    Returns
    -------
    device : str
        "/GPU:0" if a GPU is available, otherwise "/CPU:0".
    """

    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"Compute device     : {device} ({len(gpus)} GPU(s) visible)")

    return device


# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Data Preparation
def prepare_ml_data(ds, bvb_terms, mask=None):
    """
    Convert xarray Dataset into ML arrays.

    Returns
    -------
    X : ndarray
        Shape (samples, features)
    y : ndarray
        Shape (samples,)
    sample_index : pandas.MultiIndex
        Grid coordinates of each retained sample.
    """

    if mask is not None:
        ds = ds.where(mask)

    # Stack all spatial/temporal dimensions
    sample_dims = [dim for dim in ("time", "lat", "lon") if dim in ds.dims]
    stacked = ds.stack(sample=sample_dims)

    # Extract data of interest
    X = stacked[bvb_terms].to_array("feature").transpose("sample", "feature")
    y = stacked["bvb_regime"]

    # Get valid mask for both X and y
    valid = np.isfinite(X).all("feature") & np.isfinite(y)

    # Get correct data
    X = X.where(valid, drop=True)
    sample_index = X.indexes["sample"]

    X = X.values.astype(np.float32)
    y = y.where(valid, drop=True).values.astype(np.int64)

    return X, y, sample_index



# Regime Label Encoding
def encode_labels(y_train, y_valid, y_test, y_total):
    """
    Convert arbitrary regime labels to contiguous
    class indices: 0, 1, ..., n_classes-1.
    """

    labels = np.unique(y_train)
    label_to_index = {label: i for i, label in enumerate(labels)}

    def encode(y):
        return np.array([label_to_index[label] for label in y], dtype=np.int64,)

    return (encode(y_train), encode(y_valid), encode(y_test), labels, label_to_index, encode(y_total))



# Model
def build_model(num_features, num_classes):
    inputs = Input(shape=(num_features,))

    x = Dense(24, activation="tanh")(inputs)
    x = Dense(24, activation="tanh")(x)
    x = Dense(16, activation="tanh")(x)
    x = Dense(16, activation="tanh")(x)

    outputs = Dense(num_classes, activation="softmax",)(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy",),],)

    return model



# Ensemble Prediction
def predict_ensemble(models, X, device):
    """
    Generate predictions from all ensemble members.

    Returns
    -------
    member_probs : ndarray
        (members, samples, classes)

    ensemble_probs : ndarray
        Mean probability across members.

    ensemble_pred : ndarray
        Final ensemble class prediction.
    """

    with tf.device(device):
        member_probs = np.stack([model.predict(X, batch_size=PREDICT_BATCH_SIZE, verbose=0) for model in models], axis=0,)

    ensemble_probs = member_probs.mean(axis=0)
    ensemble_pred = np.argmax(ensemble_probs,  axis=1,)

    return member_probs, ensemble_probs, ensemble_pred



# Ensemble Uncertainty
def prediction_entropy(probabilities):
    """Shannon entropy of categorical probabilities."""

    probabilities = np.clip(probabilities, 1e-12, 1.0,)

    return -np.sum(probabilities * np.log(probabilities), axis=-1,)


def ensemble_diagnostics(member_probs):
    """
    Calculate ensemble uncertainty and member disagreement.
    """

    ensemble_probs = member_probs.mean(axis=0)
    entropy = prediction_entropy(ensemble_probs)
    member_preds = np.argmax(member_probs, axis=-1,)

    # Fraction of members disagreeing with ensemble prediction
    ensemble_pred = np.argmax(ensemble_probs, axis=-1,)
    disagreement = np.mean(member_preds != ensemble_pred[None, :], axis=0,)

    return {"entropy": entropy,
            "disagreement": disagreement,
            "ensemble_probability": ensemble_probs.max(axis=1),}



# Evaluation
def evaluate_predictions(y_true, probabilities, class_labels, name="ensemble",):
    """
    Evaluate multiclass predictions.
    """

    y_pred = np.argmax(probabilities, axis=1)

    metrics = {"accuracy": accuracy_score(y_true, y_pred),
               "balanced_accuracy": balanced_accuracy_score(y_true, y_pred,),
               "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0,),
               "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0,),
               "top2_accuracy": top_k_accuracy_score(y_true, probabilities, k=2, labels=np.arange(len(class_labels)),),
              }

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=np.arange(len(class_labels)),
                                target_names=[str(label) for label in class_labels],
                                zero_division=0,))

    return metrics, confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels)),)



# Geographic Back-Projection
def back_project(sample_index, fields, template):
    """
    Map flat sample-space results back onto the original grid.

    Parameters
    ----------
    sample_index : pandas.MultiIndex
        Grid coordinates of each sample, from prepare_ml_data.
    fields : dict
        name -> (values, extra_dims), values shaped (samples,) + extra_dims.
    template : xarray.Dataset
        Source of the full grid coordinates to reindex onto.

    Returns
    -------
    xarray.Dataset on the original grid, NaN where samples were absent.
    """

    ds = xr.Dataset(coords=xr.Coordinates.from_pandas_multiindex(sample_index, "sample",))

    for name, (values, extra_dims) in fields.items():
        ds[name] = (("sample",) + tuple(extra_dims), values)

    ds = ds.unstack("sample")

    return ds.reindex({dim: template[dim] for dim in sample_index.names})


def build_prediction_dataset(sample_index, y_true, ensemble_probs, ensemble_pred,
                             diagnostics, class_labels, template, name,):
    """
    Assemble gridded ensemble results for one prediction case.
    """

    fields = {"regime_true": (class_labels[y_true], ()),
              "regime_predicted": (class_labels[ensemble_pred], ()),
              "ensemble_probability": (diagnostics["ensemble_probability"].astype(np.float32), ()),
              "class_probability": (ensemble_probs.astype(np.float32), ("regime",)),
              "entropy": (diagnostics["entropy"].astype(np.float32), ()),
              "disagreement": (diagnostics["disagreement"].astype(np.float32), ()),
             }

    ds = back_project(sample_index, fields, template,)
    ds = ds.assign_coords(regime=("regime", class_labels,)).transpose(*sample_index.names, ...)
    ds.attrs.update(case=name, num_members=NUM_MEMBERS, model=MODEL_NAME,)

    return ds



# Path Configuration
SLVP_DIR = Path("/group/maikesgrp/laique/SLVP")
mask_path = SLVP_DIR / f"inputs/nn_inputs_CM4X_{DATA_RES}_masks.zarr"
bvb_path = SLVP_DIR / f"inputs/global_CM4X_{DATA_RES}_BVB_tm_fields.zarr"
label_path = SLVP_DIR  / f"inputs/nn_labels_HAC25_emb_ID6_CM4X_{DATA_RES}_{SCENARIO}.zarr"

# Configure Device
DEVICE = configure_device()

# Load Data
ds_mask = xr.open_zarr(mask_path, chunks=None)
ds_bvb = xr.open_zarr(bvb_path, chunks=None)
ds_labels = xr.open_zarr(label_path, chunks=None)

# Prepare Masks
global_mask = ds_mask.global_mask
train_mask = ds_mask.train_mask
valid_mask = ds_mask.valid_mask
test_mask = ds_mask.test_mask


# Prepare input dataset
ds_inputs = xr.merge([ds_bvb, ds_labels])


# Prepare ML data
X_total, y_total, total_index = prepare_ml_data(ds_inputs, BVB_TERMS, mask=global_mask,)
X_train, y_train, train_index = prepare_ml_data(ds_inputs, BVB_TERMS, mask=train_mask,)
X_valid, y_valid, valid_index = prepare_ml_data(ds_inputs, BVB_TERMS, mask=valid_mask,)
X_test, y_test, test_index = prepare_ml_data(ds_inputs, BVB_TERMS, mask=test_mask,)


# Encode labels
(y_train, y_valid, y_test, class_labels, label_to_index, y_total,) = encode_labels(y_train, y_valid, y_test, y_total,)

num_classes = len(class_labels)
num_features = X_train.shape[1]

print(f"Number of features : {num_features}")
print(f"Number of classes  : {num_classes}")
print(f"Training samples   : {len(y_train)}")
print(f"Validation samples : {len(y_valid)}")
print(f"Testing samples    : {len(y_test)}")


# Scale Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_valid = scaler.transform(X_valid).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
X_total = scaler.transform(X_total).astype(np.float32)


# Save Scaler and Class Information
MODEL_NAME = f"model_24x2_16x2_tanh_nc{num_classes}_CM4X_{DATA_RES}_{SCENARIO}"
MODEL_DIR = SLVP_DIR / "NN4X/models" / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR = MODEL_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

dump(scaler, MODEL_DIR / "scaler.pkl",)
np.save(MODEL_DIR / "class_labels.npy", class_labels,)



# Train Ensemble
history_dict = {}

for member in range(NUM_MEMBERS):

    print(f"\nTraining Ensemble Member {member + 1}/{NUM_MEMBERS}")

    set_seed(SEED + member)
    tf.keras.backend.clear_session()

    with tf.device(DEVICE):
        model = build_model(num_features, num_classes,)
        member_name = f"model_{member + 1}"

        checkpoint_path = WEIGHTS_DIR / f"{member_name}.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0,)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0,)

        history = model.fit(X_train, y_train,
                            validation_data=(X_valid, y_valid),
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            shuffle=True,
                            verbose=0,
                            callbacks=[checkpoint, early_stopping,],)

        history_dict[member_name] = history.history

        # Individual member validation performance
        val_prob = model.predict(X_valid, batch_size=PREDICT_BATCH_SIZE, verbose=0,)

    evaluate_predictions(y_valid, val_prob, class_labels, name=member_name,)



# Save Training History

np.savez_compressed(MODEL_DIR / f"ensemble_training_history_{NUM_MEMBERS}_members.npz",
                    **{key: np.array(list(value.items()), dtype=object,) for key, value in history_dict.items()},)


# Load ensemble
with tf.device(DEVICE):
    models = [tf.keras.models.load_model(WEIGHTS_DIR / f"model_{i + 1}.keras") for i in range(NUM_MEMBERS)]

# Ensemble Validation Prediction
valid_member_probs, valid_probs, valid_pred = predict_ensemble(models, X_valid, DEVICE,)
valid_metrics, valid_cm = evaluate_predictions(y_valid, valid_probs, class_labels, name="SOFT-VOTING ENSEMBLE — VALIDATION",)

# Ensemble Test Prediction
test_member_probs, test_probs, test_pred = predict_ensemble(models, X_test, DEVICE,)
test_metrics, test_cm = evaluate_predictions(y_test, test_probs, class_labels, name="SOFT-VOTING ENSEMBLE — TEST",)

# Ensemble Test Diagnostics
test_diagnostics = ensemble_diagnostics(test_member_probs)

print("\nEnsemble Test Uncertainty:")
print(f"Mean entropy       : {test_diagnostics['entropy'].mean():.4f}")
print(f"Mean disagreement  : {test_diagnostics['disagreement'].mean():.4f}")
print(f"Mean confidence    : {test_diagnostics['ensemble_probability'].mean():.4f}")


# Save Test Predictions
np.savez_compressed(MODEL_DIR / "ensemble_test_predictions.npz",
                    y_true=y_test,
                    member_probabilities=test_member_probs,
                    ensemble_probabilities=test_probs,
                    ensemble_predictions=test_pred,
                    entropy=test_diagnostics["entropy"],
                    disagreement=test_diagnostics["disagreement"],)


# Ensemble Total Prediction
total_member_probs, total_probs, total_pred = predict_ensemble(models, X_total, DEVICE,)
total_metrics, total_cm = evaluate_predictions(y_total, total_probs, class_labels, name="SOFT-VOTING ENSEMBLE — TOTAL",)

# Ensemble Total Diagnostics
total_diagnostics = ensemble_diagnostics(total_member_probs)

print("\nEnsemble Total Uncertainty:")
print(f"Mean entropy       : {total_diagnostics['entropy'].mean():.4f}")
print(f"Mean disagreement  : {total_diagnostics['disagreement'].mean():.4f}")
print(f"Mean confidence    : {total_diagnostics['ensemble_probability'].mean():.4f}")


# Save Total predictions
np.savez_compressed(MODEL_DIR / "ensemble_total_predictions.npz",
                    y_true=y_total,
                    member_probabilities=total_member_probs,
                    ensemble_probabilities=total_probs,
                    ensemble_predictions=total_pred,
                    entropy=total_diagnostics["entropy"],
                    disagreement=total_diagnostics["disagreement"],)



# Back-Project Ensemble Results onto the Original Grid
PREDICTION_DIR = MODEL_DIR / "gridded"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

cases = {"test": (test_index, y_test, test_probs, test_pred, test_diagnostics),
         "total": (total_index, y_total, total_probs, total_pred, total_diagnostics),}

for case_name, (sample_index, y_true, probs, pred, diagnostics) in cases.items():

    ds_case = build_prediction_dataset(sample_index, y_true, probs, pred,
                                       diagnostics, class_labels, ds_inputs, case_name,)

    out_path = PREDICTION_DIR / f"ensemble_{case_name}_predictions_gridded.zarr"
    ds_case.to_zarr(out_path, mode="w", consolidated=True,)

    print(f"\nGridded {case_name} predictions written to {out_path}")
    print(ds_case)
