import pandas as pd
from tqdm import tqdm
from lstm import LSTM
import numpy as np
import math
import random
from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def compute_mse_loss(pred, target):
    """Compute Mean Squared Error loss."""
    return (pred - target) ** 2

def compute_mse_derivative(pred, target):
    """Compute derivative of MSE loss w.r.t. predicted value."""
    return 2 * (pred - target)

# --- Utility Functions ---

def softmax(logits):
    """Numerically stable softmax."""
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def cross_entropy_loss_from_probs(probs, target_index):
    """Compute the cross entropy loss for predicted probabilities."""
    loss = -math.log(probs[target_index] + 1e-12)
    return loss

def compute_dloss(probs, target_index):
    """Compute gradient of the softmax loss."""
    return [probs[j] - (1 if j == target_index else 0) for j in range(len(probs))]

def classify_return(r, threshold=0):
    """
    For binary classification:
      - Return 1 (Up) if computed return is nonnegative.
      - Return 0 (Down) if computed return is negative.
    """
    return 1 if r >= threshold else 0


if __name__ == '__main__':
    np.random.seed(6001)
    random.seed(6001)
    hidden_size = 10
    sequence_length = 50
    learning_rate = 0.01
    epochs = 10
    training_window_size = 2
    testing_window_size = 1

    print("Reading data file...")
    df = pd.read_csv('1min_features_withFutures.csv').fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2023]
    df = df[df["nextVwap"] != 0]

    stock_list = df['Stock'].unique()
    time_point_list = sorted(df['DateTime'].unique())
    date_list = sorted(df['date'].unique())

    df["computed_return"] = (df["nextVwap"] - df["vwap"])*100 / df["vwap"]

    drop_cols = ['DateTime', 'Stock', 'rtn', 'nextRtn', 'rtnMidQ', 'date', 'vwap', 'nextVwap', 'nextParkVol', 'computed_return']
    feature_columns = ['prevRtn', 'avgOrderImb', 'avgNormBV5', 'avgNormBV1', 'SP1', 'avgNormBV2', 'BP1', 'avgNormBV4', 'avgNormBV3', 'proc_m_20min', 'avgNormSV1', 'avgOrderImb_futures', 'avgSlope_ab_futures', 'avgSpreadRatio', 'sma_m_60min_10min', 'sma_m_30min_10min', 'avgSpreadRatio_futures', 'proc_m_30min', 'avgNormSV4_futures', 'sma_m_60min_5min', 'avgNormBV5_futures', 'avgNormSV3', 'will_m_60min', 'proc_m_15min', 'avgNormSV2', 'parkVol_20min', 'cho_m_60min_5min', 'avgNormSV5_futures', 'avgNormSV5', 'proc_m_10min', 'parkVol_30min', 'avgNormSV3_futures', 'parkVol_5min', 'parkVol_10min', 'cho_m_30min_10min', 'avgNormBV4_futures', 'avgNormSV4', 'avgNormSV2_futures']
    
    input_size = len(feature_columns)

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    train_results = []
    test_results = []
    window_metrics = []

    final_train_df = None
    final_test_df = None

    max_start = len(date_list) - (training_window_size + testing_window_size) + 1
    window_starts = range(0, max_start, testing_window_size)
    for window_start_idx in tqdm(window_starts, desc="Rolling Windows Progress"):
        train_end_idx = window_start_idx + training_window_size - 1
        test_end_idx = train_end_idx + testing_window_size
        train_dates = date_list[window_start_idx:train_end_idx + 1]
        test_dates = date_list[train_end_idx + 1:train_end_idx + testing_window_size + 1]
        train_time_points = sorted(df[df['date'].isin(train_dates)]['DateTime'].unique())
        test_time_points = sorted(df[df['date'].isin(test_dates)]['DateTime'].unique())

        window_label = df[df['date'] == test_dates[0]]['DateTime'].min() if test_dates and not df[df['date'] == test_dates[0]].empty else time_point_list[-1]

        lstm_models = {stock: LSTM(input_size, hidden_size, output_size=1) for stock in stock_list}
        loss_values = []

        y_true_train, y_pred_train, stock_train = [], [], []
        y_true_test, y_pred_test, stock_test = [], [], []

        # Training
        for time_point in train_time_points[sequence_length:]:
            current_index = time_point_list.index(time_point)
            start_index = max(0, current_index - sequence_length)
            relevant_time_points = time_point_list[start_index:current_index + 1]

            for stock in stock_list:
                stock_data = df[(df['Stock'] == stock) & (df['DateTime'].isin(relevant_time_points))]
                if len(stock_data) < sequence_length:
                    continue

                stock_features = stock_data[feature_columns].values.tolist()
                stock_target = stock_data["computed_return"].iloc[-1]
                input_sequence = stock_features[:sequence_length]

                lstm = lstm_models[stock]
                lstm.stock = stock 
                best_val_loss = np.inf
                counter = 0
                best_model_state = None

                # Train with early stopping
                for epoch in range(epochs):
                    pred, outputs, caches = lstm.forward(input_sequence, weight=1)

                    loss = compute_mse_loss(pred, stock_target)
                    d_y = compute_mse_derivative(pred, stock_target)
                    lstm.backward(d_y, outputs, caches)
                    lstm.update_parameters(learning_rate)
                    loss_values.append(loss)


                y_true_train.append(stock_target)
                y_pred_train.append(pred)
                stock_train.append(stock)

        # Compute training metrics
        mse_train = mean_squared_error(y_true_train, y_pred_train) if y_true_train else np.nan
        mae_train = mean_absolute_error(y_true_train, y_pred_train) if y_true_train else np.nan
        rmse_train = np.sqrt(mse_train) if not np.isnan(mse_train) else np.nan

        # Testing
        test_losses = []
        for time_point in test_time_points[sequence_length:]:
            current_index = time_point_list.index(time_point)
            start_index = max(0, current_index - sequence_length)
            relevant_time_points = time_point_list[start_index:current_index + 1]
            for stock in stock_list:
                stock_data = df[(df['Stock'] == stock) & (df['DateTime'].isin(relevant_time_points))]
                if len(stock_data) <= sequence_length:
                    continue

                stock_features = stock_data[feature_columns].values.tolist()
                input_sequence = stock_features[:sequence_length]
                stock_target = stock_data["computed_return"].iloc[-1]

                lstm = lstm_models[stock]
                pred, _, _ = lstm.forward(input_sequence, weight=1)
                y_true_test.append(stock_target)
                y_pred_test.append(pred)
                stock_test.append(stock)

                loss = compute_mse_loss(pred, stock_target)
                test_losses.append(loss)

        # Compute testing metrics
        mse_test = mean_squared_error(y_true_test, y_pred_test) if y_true_test else np.nan
        mae_test = mean_absolute_error(y_true_test, y_pred_test) if y_true_test else np.nan
        rmse_test = np.sqrt(mse_test) if not np.isnan(mse_test) else np.nan

        # Store results
        train_df = pd.DataFrame({
            'Window': [window_label] * len(y_true_train),
            'Stock': stock_train,
            'True_Value': y_true_train,
            'Prediction': y_pred_train
        })
        test_df = pd.DataFrame({
            'Window': [window_label] * len(y_true_test),
            'Stock': stock_test,
            'True_Value': y_true_test,
            'Prediction': y_pred_test
        })

        train_results.append(train_df)
        test_results.append(test_df)

        window_metrics.append({
            'Window_Start': date_list[window_start_idx],
            'Training_MSE': mse_train,
            'Training_MAE': mae_train,
            'Training_RMSE': rmse_train,
            'Testing_MSE': mse_test,
            'Testing_MAE': mae_test,
            'Testing_RMSE': rmse_test
        })

        print(f"\nWindow Metrics (Start: {date_list[window_start_idx]}):")
        print(f"  Training: MSE: {mse_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
        print(f"  Testing: MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")

    if train_results:
        final_train_df = pd.concat(train_results, ignore_index=True)
        final_train_df.to_csv('training_results_regression_independent.csv', index=False)
        print("\nSaved training results to training_results_regression_independent.csv")
    
    if test_results:
        final_test_df = pd.concat(test_results, ignore_index=True)
        final_test_df.to_csv('testing_results_regression_independent.csv', index=False)
        print("Saved testing results to testing_results_regression_independent.csv")

    window_metrics_df = pd.DataFrame(window_metrics)
    window_metrics_df.to_csv('window_metrics_independent.csv', index=False)
    print("Saved window metrics to window_metrics_independent.csv")

    if final_train_df.shape[0] > 0:
        overall_mse_train = mean_squared_error(final_train_df['True_Value'], final_train_df['Prediction'])
        overall_mae_train = mean_absolute_error(final_train_df['True_Value'], final_train_df['Prediction'])
        overall_rmse_train = np.sqrt(overall_mse_train)
        print(f"\nOverall Training Metrics:\n MSE: {overall_mse_train:.4f}, MAE: {overall_mae_train:.4f}, RMSE: {overall_rmse_train:.4f}")

    if final_test_df.shape[0] > 0:
        overall_mse_test = mean_squared_error(final_test_df['True_Value'], final_test_df['Prediction'])
        overall_mae_test = mean_absolute_error(final_test_df['True_Value'], final_test_df['Prediction'])
        overall_rmse_test = np.sqrt(overall_mse_test)
        print(f"\nOverall Testing Metrics:\n MSE: {overall_mse_test:.4f}, MAE: {overall_mae_test:.4f}, RMSE: {overall_rmse_test:.4f}")