import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def train_model(input_file):
    print(f"Loading features from {input_file}...")
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Run 02_feature_engineering.py first.")
        return

    # Extended Features
    feature_cols = [
        'sma_20', 'sma_50', 'rsi', 
        'macd', 'macd_signal', 'macd_diff',
        'bb_width',
        'return', 'return_lag1', 'return_lag2'
    ]
    
    X = df[feature_cols]
    y = df['target']

    # Time-based Split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # Initialize and Train XGBoost
    # Using small depth to avoid overfitting on limited data
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] # Probability of Class 1 (Up)

    # Evaluate
    acc = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Feature Importance Plot
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nFeature Importances:")
    print(importances.sort_values(ascending=False))
    
    plt.figure()
    importances.sort_values().plot(kind='barh')
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    
    # Save predictions
    result_df = X_test.copy()
    result_df['actual'] = y_test
    result_df['predicted'] = predictions
    result_df['prob_up'] = probs
    result_df.to_csv("data/predictions.csv")
    print("\nPredictions saved to data/predictions.csv")

if __name__ == "__main__":
    INPUT_FILE = "data/stock_data_features.csv"
    train_model(INPUT_FILE)
