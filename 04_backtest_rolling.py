import pandas as pd
import matplotlib.pyplot as plt

def backtest_rolling(predictions_file):
    print(f"Loading rolling predictions from {predictions_file}...")
    try:
        df = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {predictions_file} not found.")
        return

    # Initial Capital
    initial_capital = 10000

    # Strategy: Buy if Predicted=1
    # df already has 'return' from original data attached
    
    # Logic:
    # If we predicted UP (1) for Tomorrow, we buy Tomorrow Open/Close, catching Tomorrow's return.
    # The 'return' column in data usually tracks 'Today Close vs Yesterday Close'.
    # Our Target was Shift(-1), so 'predicted' on Day T matches 'return' on Day T+1.
    # But in `results_df`, we joined on index.
    # The `predictions_rolling.csv` has index T. 
    # 'predicted' at T is prediction for T+1.
    # 'return' at T is return of T.
    # So we need to shift Position by 1 to align.
    
    df['position'] = df['predicted']
    df['strategy_return'] = df['position'].shift(1) * df['return']
    df['benchmark_return'] = df['return']
    
    # Filter: removing first row NaN due to shift
    df.dropna(inplace=True)

    # Cumulative Returns
    df['strategy_equity'] = initial_capital * (1 + df['strategy_return']).cumprod()
    df['benchmark_equity'] = initial_capital * (1 + df['benchmark_return']).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['strategy_equity'], label='Rolling Strategy (XGBoost)')
    plt.plot(df.index, df['benchmark_equity'], label='Benchmark (Buy & Hold)', alpha=0.6)
    plt.title('Walk-Forward Validation: Rolling Model vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    output_img = "data/backtest_rolling_result.png"
    plt.savefig(output_img)
    print(f"Rolling Backtest plot saved to {output_img}")
    
    # Metrics
    total_return = (df['strategy_equity'].iloc[-1] / initial_capital) - 1
    print(f"Total Rolling Strategy Return: {total_return:.2%}")
    print(f"Total Benchmark Return: {((df['benchmark_equity'].iloc[-1] / initial_capital) - 1):.2%}")

if __name__ == "__main__":
    PREDICTIONS_FILE = "data/predictions_rolling.csv"
    backtest_rolling(PREDICTIONS_FILE)
