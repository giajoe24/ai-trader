import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def hyperparameter_tuning(X_train, y_train):
    print("Starting Hyperparameter Tuning (Grid Search)... this may take a minute.")
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    
    # Grid of parameters to test
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # TimeSeriesSplit prevents future data leakage during Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1, # Parallel processing
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_rolling_window(input_file):
    print(f"Loading features from {input_file}...")
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    feature_cols = [
        'sma_20', 'sma_50', 'rsi', 
        'macd', 'macd_signal', 'macd_diff',
        'bb_width',
        'return', 'return_lag1', 'return_lag2'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    # Configuration for Walk-Forward
    # Train initially on first 2 years, then roll forward 3 months at a time
    initial_train_size = int(len(df) * 0.6) # 60% initial data
    test_window_size = 60 # Re-train every ~60 trading days (3 months)
    
    print(f"Total Data Points: {len(df)}")
    print(f"Initial Training Size: {initial_train_size}")
    
    predictions = []
    actuals = []
    dates = []
    probs = []
    
    current_train_end = initial_train_size
    
    # 1. First, Tune Parameters on the INITIAL chunk
    print("\n--- Tuning Phase ---")
    X_init = X.iloc[:current_train_end]
    y_init = y.iloc[:current_train_end]
    best_model = hyperparameter_tuning(X_init, y_init)
    best_params = best_model.get_params()
    
    print("\n--- Rolling Execution Phase ---")
    while current_train_end < len(df):
        # Define Test Window
        test_end = min(current_train_end + test_window_size, len(df))
        
        # Training Data (Expanding Window: All history up to now)
        # Check: Is Expanding Window better? Or Sliding Window (Fixed 2 years)?
        # Expanding catches long term trends but might get confused by regime changes.
        # Let's use Sliding (last 500 days) to keep it adaptive
        start_index = max(0, current_train_end - 500)
        X_train = X.iloc[start_index:current_train_end]
        y_train = y.iloc[start_index:current_train_end]
        
        # Test Data (Next 3 months)
        X_test = X.iloc[current_train_end:test_end]
        y_test = y.iloc[current_train_end:test_end]
        
        if len(X_test) == 0:
            break
            
        print(f"Training on period: {X_train.index[0].date()} -> {X_train.index[-1].date()}")
        print(f"Predicting period: {X_test.index[0].date()} -> {X_test.index[-1].date()}")
        
        # Refit with Best Params
        # We create a new instance to reset weights, but keep 'warm_start=False' (default)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        predictions.extend(preds)
        actuals.extend(y_test)
        dates.extend(X_test.index)
        probs.extend(proba)
        
        # Move Forward
        current_train_end = test_end

    # Aggregate Results
    results_df = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions,
        'prob_up': probs
    }, index=dates)
    
    # Merge back original features for backtest analysis (returns etc)
    # We join on index
    results_df = results_df.join(df[['return', 'close']], how='left')
    
    # Save
    results_df.to_csv("data/predictions_rolling.csv")
    print("\nRolling predictions saved to data/predictions_rolling.csv")
    
    # Evaluate
    acc = accuracy_score(results_df['actual'], results_df['predicted'])
    print(f"\nOverall Rolling Accuracy: {acc:.4f}")
    print(classification_report(results_df['actual'], results_df['predicted']))

if __name__ == "__main__":
    INPUT_FILE = "data/stock_data_features.csv"
    train_rolling_window(INPUT_FILE)
