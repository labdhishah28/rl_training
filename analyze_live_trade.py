import argparse, glob, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def find_latest_csv():
    files = sorted(glob.glob('live_trade_*.csv'))
    if not files:
        raise FileNotFoundError('No live_trade_*.csv files found.')
    return files[-1]

def analyze(csv_path):
    print(f'Loading: {csv_path}')
    df = pd.read_csv(csv_path)
    df['cum_reward'] = df['reward'].cumsum()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Live Trade Analysis - ' + os.path.basename(csv_path), fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    steps = df['step']

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, df['inventory_remaining'], color='steelblue', linewidth=2, marker='o', markersize=4)
    ax1.fill_between(steps, df['inventory_remaining'], alpha=0.15, color='steelblue')
    ax1.set_title('Inventory Remaining Over Time')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Inventory (base units)'); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(steps, df['action'], color='coral', alpha=0.8, edgecolor='darkred', linewidth=0.5)
    ax2.set_title('Action Size Per Step')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Fraction to Trade [0-1]'); ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, df['mid_price'], color='purple', linewidth=2, label='Mid Price')
    ax3.plot(steps, df['execution_price'], color='orange', linewidth=1.5, linestyle='--', label='Execution Price')
    ax3.set_title('Mid Price vs Execution Price'); ax3.set_xlabel('Step'); ax3.set_ylabel('Price (USDT)')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, df['cum_reward'] * 10000, color='green', linewidth=2)
    ax4.fill_between(steps, df['cum_reward'] * 10000, alpha=0.15, color='green')
    ax4.set_title('Cumulative Cash Flow'); ax4.set_xlabel('Step'); ax4.set_ylabel('Cash Flow (USD)'); ax4.grid(True, alpha=0.3)

    plt.savefig('live_trade_analysis.png', dpi=150, bbox_inches='tight')
    print('Saved: live_trade_analysis.png')
    total_cash = df['reward'].sum() * 10000
    print(f'\nSteps={len(df)}, Avg action={df["action"].mean():.4f}, Cash flow=')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', default=None)
    args = p.parse_args()
    csv_path = args.file if args.file else find_latest_csv()
    analyze(csv_path)
