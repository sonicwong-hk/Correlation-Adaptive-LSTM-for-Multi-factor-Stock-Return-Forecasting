import pandas as pd
from tqdm import tqdm
from lstm import LSTM
import numpy as np
import math
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
from copy import deepcopy
import random
import statistics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def cross_entropy_loss_from_probs(probs, target_index):
    loss = -math.log(probs[target_index] + 1e-12)
    return loss

def compute_dloss(probs, target_index):
    return [probs[j] - (1 if j == target_index else 0) for j in range(len(probs))]

def prediction_from_probs(probs, w):
    """
    Given the probability vector (for two classes) from softmax, choose the class
    with the highest probability and return [vote, weight].
    """
    vote = probs.index(max(probs))
    return [vote, w]

def ensemble_prediction_from_probs(predictions):
    """
    For binary classification (0: Down, 1: Up), this function aggregates
    weighted votes and returns the class with the highest total weight.
    """
    weighted_counts = {0: 0, 1: 0}  # Only two classes: 0 (Down) and 1 (Up)
    for vote, weight in predictions:
        weighted_counts[vote] += weight  # Accumulate weight for each class
    return max(weighted_counts, key=weighted_counts.get)


def majority_voting(predictions):
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]

def compute_mse_loss(pred, target):
    """Compute Mean Squared Error loss."""
    return (pred - target) ** 2

def compute_mse_derivative(pred, target):
    """Compute derivative of MSE loss w.r.t. predicted value."""
    return 2 * (pred - target)

def prediction_from_output(output, w):
    """
    For regression, return the predicted value and its weight.
    """
    return [output, w]

def ensemble_prediction_from_outputs(predictions):
    """
    For regression, compute the weighted average of predictions.
    """
    total_weight = sum(w for _, w in predictions)
    if total_weight == 0:
        return 0.0  # Avoid division by zero
    weighted_sum = sum(v * w for v, w in predictions)
    return weighted_sum / total_weight

def compute_stock_correlation(df, stock_list, time_points):
    """Compute correlations and save to CSV."""
    correlations = {}
    correlation_data = []
    for stock_a in stock_list:
        for stock_b in stock_list:
            if stock_a == stock_b:
                correlations[(stock_a, stock_b)] = 1.0
                correlation_data.append([stock_a, stock_b, 1.0])
                continue
            data_a = df[(df['Stock'] == stock_a) & (df['DateTime'].isin(time_points))]['computed_return'].reset_index(drop=True)
            data_b = df[(df['Stock'] == stock_b) & (df['DateTime'].isin(time_points))]['computed_return'].reset_index(drop=True)
            if data_a.empty or data_b.empty:
                correlations[(stock_a, stock_b)] = np.nan
                correlation_data.append([stock_a, stock_b, np.nan])
            else:
                corr = data_a.corr(data_b)
                correlations[(stock_a, stock_b)] = corr
                correlation_data.append([stock_a, stock_b, corr])
    
    # Save correlations to CSV
    corr_df = pd.DataFrame(correlation_data, columns=['Stock_A', 'Stock_B', 'Correlation'])
    corr_df.to_csv('stock_correlations.csv', index=False)
    return correlations

def read_stock_correlations(csv_path='stock_correlations.csv'):
    """
    Read stock correlations from CSV and return as a dictionary.
    
    Args:
        csv_path (str): Path to the correlations CSV file.
    
    Returns:
        dict: Dictionary of correlations {(stock_a, stock_b): corr_value}.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is missing required columns or has invalid data.
    """
    try:
        # Read CSV
        corr_df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['Stock_A', 'Stock_B', 'Correlation']
        if not all(col in corr_df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Convert to dictionary
        correlations = {}
        for _, row in corr_df.iterrows():
            stock_a = int(row['Stock_A'])
            stock_b = int(row['Stock_B'])
            corr = row['Correlation']
            # Handle NaN or invalid values
            corr_value = np.nan if pd.isna(corr) else float(corr)
            correlations[(stock_a, stock_b)] = corr_value
        
        return correlations
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Correlation CSV file not found at {csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading correlations CSV: {str(e)}")
 


def get_correlation(correlations, stock_target, stock_model):
    return correlations.get((stock_target, stock_model), 0.0)

# ---------- Main Script ----------
if __name__ == '__main__':
    np.random.seed(6001)
    random.seed(6001)
    hidden_size = 1
    sequence_length = 10
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

    pretrain_size = int(len(time_point_list) * 0.1)
    pretrain_time_points = time_point_list[:pretrain_size]


    print("Getting correlations...")
    #correlations = read_stock_correlations('stock_correlations.csv')
    correlations  = compute_stock_correlation(df, stock_list, pretrain_time_points)

    lstm_models = {stock: LSTM(input_size, hidden_size, output_size=1) for stock in stock_list}

    final_train_df = None
    final_test_df = None

    # Pretraining 
    print("\nPretraining Individual LSTM Models on minute-level data...")
    for time_point in tqdm(pretrain_time_points[sequence_length:], desc="Pretraining Progress"):
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

            for epoch in range(epochs):
                pred, outputs, caches = lstm.forward(input_sequence, weight=1)
                loss = compute_mse_loss(pred, stock_target)
                d_y = compute_mse_derivative(pred, stock_target)
                lstm.backward(d_y, outputs, caches)
                lstm.update_parameters(learning_rate)


    train_results = []
    test_results = []
    window_metrics = []

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

        lstm_models_window = {stock: deepcopy(lstm) for stock, lstm in lstm_models.items()}
        loss_values = []

        y_true_train, y_pred_train, stock_train, ensemble_models_train = [], [], [], []
        y_true_test, y_pred_test, stock_test, ensemble_models_test = [], [], [], []

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

                pred_other = []
                ensemble_model_preds = []
                for other_stock, model in lstm_models_window.items():
                    corr_val = get_correlation(correlations, stock, other_stock)
                    if corr_val >= 0.1:
                        w = corr_val
                        h_backup = model.h.copy()
                        c_backup = model.c.copy()
                        pred_other_val, _, _ = model.forward(input_sequence, weight=w)
                        pred_other.append(prediction_from_output(pred_other_val, w))
                        ensemble_model_preds.append({
                            'Model_Stock': other_stock,
                            'Prediction': pred_other_val,
                            'Weight': w
                        })
                        model.h, model.c = h_backup, c_backup

                ensemble_pred = ensemble_prediction_from_outputs(pred_other)
                y_true_train.append(stock_target)
                y_pred_train.append(ensemble_pred)
                stock_train.append(stock)
                ensemble_models_train.append(str(ensemble_model_preds))

                lstm = lstm_models_window[stock]
                lstm.stock = stock
                best_val_loss = np.inf
                counter = 0
                best_model_state = None

                for epoch in range(epochs):
                    pred, outputs, caches = lstm.forward(input_sequence, weight=1)
                    loss = compute_mse_loss(pred, stock_target)
                    d_y = compute_mse_derivative(pred, stock_target)
                    lstm.backward(d_y, outputs, caches)
                    lstm.update_parameters(learning_rate)
                    loss_values.append(loss)


        # Compute training metrics for this window
        mse_train = mean_squared_error(y_true_train, y_pred_train) if y_true_train else np.nan
        mae_train = mean_absolute_error(y_true_train, y_pred_train) if y_true_train else np.nan
        rmse_train = np.sqrt(mse_train) if not np.isnan(mse_train) else np.nan

        # Testing on minute-level DateTime
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

                pred_other = []
                ensemble_model_preds = []
                for other_stock, model in lstm_models_window.items():
                    corr_val = get_correlation(correlations, stock, other_stock)
                    if corr_val >= 0.10:  # Correlation threshold
                        w = corr_val
                        h_backup = model.h.copy()
                        c_backup = model.c.copy()
                        pred_other_val, _, _ = model.forward(input_sequence, weight=w)
                        pred_other.append(prediction_from_output(pred_other_val, w))
                        ensemble_model_preds.append({
                            'Model_Stock': other_stock,
                            'Prediction': pred_other_val,
                            'Weight': w
                        })
                        model.h, model.c = h_backup, c_backup

                ensemble_pred = ensemble_prediction_from_outputs(pred_other)
                y_true_test.append(stock_target)
                y_pred_test.append(ensemble_pred)
                stock_test.append(stock)
                ensemble_models_test.append(str(ensemble_model_preds))

                lstm = lstm_models_window[stock]
                pred, _, _ = lstm.forward(input_sequence, weight=1)
                loss = compute_mse_loss(pred, stock_target)
                test_losses.append(loss)

        # Compute testing metrics for this window
        mse_test = mean_squared_error(y_true_test, y_pred_test) if y_true_test else np.nan
        mae_test = mean_absolute_error(y_true_test, y_pred_test) if y_true_test else np.nan
        rmse_test = np.sqrt(mse_test) if not np.isnan(mse_test) else np.nan

        # Store results for this window
        train_df = pd.DataFrame({
            'Window': [window_label] * len(y_true_train),
            'Stock': stock_train,
            'True_Value': y_true_train,
            'Prediction': y_pred_train,
            'Ensemble_Model_Predictions': ensemble_models_train
        })
        test_df = pd.DataFrame({
            'Window': [window_label] * len(y_true_test),
            'Stock': stock_test,
            'True_Value': y_true_test,
            'Prediction': y_pred_test,
            'Ensemble_Model_Predictions': ensemble_models_test
        })

        train_results.append(train_df)
        test_results.append(test_df)

        # Store window metrics
        window_metrics.append({
            'Window_Start': date_list[window_start_idx],
            'Training_MSE': mse_train,
            'Training_MAE': mae_train,
            'Training_RMSE': rmse_train,
            'Testing_MSE': mse_test,
            'Testing_MAE': mae_test,
            'Testing_RMSE': rmse_test
        })

        # Print window metrics
        print(f"\nWindow Metrics (Start: {date_list[window_start_idx]}):")
        print(f"  Training: MSE: {mse_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
        print(f"  Testing: MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")

    # Combine results from all windows
    if train_results:
        final_train_df = pd.concat(train_results, ignore_index=True)
        final_train_df.to_csv('training_results_regression - ensemble.csv', index=False)
        print("\nSaved training results to training_results_regression - ensemble.csv")
    
    if test_results:
        final_test_df = pd.concat(test_results, ignore_index=True)
        final_test_df.to_csv('testing_results_regression - ensemble.csv', index=False)
        print("Saved testing results to testing_results_regression - ensemble.csv")

    # Save window metrics
    window_metrics_df = pd.DataFrame(window_metrics)
    window_metrics_df.to_csv('window_metrics - ensemble.csv', index=False)
    print("Saved window metrics to window_metrics - ensemble.csv")

    # Compute and print overall metrics
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