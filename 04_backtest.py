import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(predictions_file):
    print(f"Loading predictions from {predictions_file}...")
    try:
        df = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {predictions_file} not found. Run 03_train_model.py first.")
        return

    # Initial Capital
    initial_capital = 10000 # USD

    # Strategy Returns
    # If predicted UP (1), we take the return. If DOWN (0), we take 0 return (Cash).
    # Shift(-1) was used for target, but here 'return' column is the return of 'today' based on yesterday close?
    # Wait, in feature_engineering: df['return'] = df['close'].pct_change()
    # This return is "Return from Yesterday to Today".
    # And we predicted "Target" which is valid for "Tomorrow".
    # So we need to match Prediction(Day T) with Return(Day T+1).
    
    # Let's check 03_train_model.py output logic.
    # result_df includes 'return' which is the return of the current row date.
    # The 'target' was shift(-1).
    # So if we predict for row T, we are predicting the return of T+1.
    # We need to align signal with next day's return.
    
    # We will simply recalculate strategy return properly.
    # Signal (Position for tomorrow)
    df['position'] = df['predicted']
    
    # Strategy Return = Position(Yesterday) * Return(Today)
    # We need to shift position forward by 1 day to match it with the return it generates.
    df['strategy_return'] = df['position'].shift(1) * df['return']
    
    # Benchmark (Buy and Hold)
    df['benchmark_return'] = df['return']

    # Cumulative Returns
    df['strategy_equity'] = initial_capital * (1 + df['strategy_return']).cumprod()
    df['benchmark_equity'] = initial_capital * (1 + df['benchmark_return']).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['strategy_equity'], label='Strategy')
    plt.plot(df.index, df['benchmark_equity'], label='Buy & Hold (Benchmark)', alpha=0.7)
    plt.title('Backtest Results: Model vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    output_img = "data/backtest_result.png"
    plt.savefig(output_img)
    print(f"Backtest plot saved to {output_img}")
    
    # Metrics
    total_return = (df['strategy_equity'].iloc[-1] / initial_capital) - 1
    print(f"Total Strategy Return: {total_return:.2%}")
    print(df[['return', 'predicted', 'strategy_return']].tail())

if __name__ == "__main__":
    PREDICTIONS_FILE = "data/predictions.csv"
    backtest_strategy(PREDICTIONS_FILE)
