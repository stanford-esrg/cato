import numpy as np
import time
import random


from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

from helper import utils

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def create_nn_model(input_dim, l2_reg, learning_rate, dropout_rate, num_nodes):
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=input_dim, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_nodes, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_nodes, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="linear"))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
    return model

def train_and_evaluate(X_train, y_train, params):
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    val_mse_scores = []
    
    for i, (train_index, val_index) in enumerate(kfold.split(X_train)):
        print(utils.CYAN + f"Fold {i}" + utils.RESET)
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model = create_nn_model(
            input_dim=X_train.shape[1],
            l2_reg=params["l2_reg"],
            learning_rate=params["learning_rate"],
            dropout_rate=params["dropout_rate"],
            num_nodes=params["num_nodes"]
        )
        
        early_stopping = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
        history = model.fit(X_train_fold, y_train_fold, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(X_val_fold, y_val_fold), verbose=1, callbacks=[early_stopping])
        
        val_mse = min(history.history["val_mse"])
        val_mse_scores.append(val_mse)
    
    avg_val_mse = np.mean(val_mse_scores)
    print(f"Average validation MSE: {avg_val_mse}")
    return avg_val_mse


def train_python_dnn(X, y):
    print(utils.CYAN + "Training Python DNN" + utils.RESET)
    print(f"Training set shape: {X.shape}")
    start_ts = time.time()

    # hyperparams
    param_grid = {
        "batch_size": [16, 32, 64],
        "epochs": [100],
        "l2_reg": [0.1, 0.5],
        "learning_rate": [0.001, 0.01],
        "dropout_rate": [0.2, 0.4, 0.6, 0.8],
        "num_nodes": [4, 8, 16]
    }

    best_val_mse = float("inf")
    best_model = None
    best_params = None

    for combo in range(20):
        params = {key: random.choice(values) for key, values in param_grid.items()}
        val_mse = train_and_evaluate(X, y, params)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params

    best_model = create_nn_model(
        input_dim=X.shape[1],
        l2_reg=best_params["l2_reg"],
        learning_rate=best_params["learning_rate"],
        dropout_rate=best_params["dropout_rate"],
        num_nodes=best_params["num_nodes"]
    )

    early_stopping = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
    history = best_model.fit(X, y, epochs=best_params["epochs"], batch_size=best_params["batch_size"], verbose=2, callbacks=[early_stopping])

    print(f"training time: {time.time() - start_ts}s")
    res = {
        "reg": best_model, 
        "params": best_params, 
        "rmse": np.sqrt(best_val_mse), 
    }
    return res

def test_python_dnn(model, X_test, y_test, num_trials=1000):
    print(utils.CYAN + "Testing Python DNN" + utils.RESET)
    num_trials = min(num_trials, len(X_test))

    y_hat = []
    y_time = []
    indexes = np.random.choice(len(X_test), num_trials, replace=False)
    for idx in indexes:
        instance = X_test[idx, :].reshape(1, -1) 
        pred_start = time.perf_counter_ns()
        pred = model.predict(instance)
        pred_end = time.perf_counter_ns()
        y_hat.append(pred)
        y_time.append(pred_end - pred_start)
    
    y_hat_batch = model.predict(X_test).flatten()

    rmse = root_mean_squared_error(y_test, y_hat_batch)
    mae = mean_absolute_error(y_test, y_hat_batch)
    medae = median_absolute_error(y_test, y_hat_batch)
    res = {
        "rmse": rmse, 
        "mae": mae,
        "medae": medae,
        "indiv_pred_time": float(np.mean(y_time)),
        "p99_pred_time": float(np.percentile(y_time, 99)),
    }
    return res

