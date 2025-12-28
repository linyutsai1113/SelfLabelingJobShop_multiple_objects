import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_curves(csv_path):
    """
    Reads the CSV log and plots training/validation loss.
    """
    if not os.path.exists(csv_path):
        print(f"[Plot] File not found: {csv_path}")
        return

    print(f"[Plot] Generating loss plot for {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Plot] Error reading CSV: {e}")
        return

    # Check columns
    if not {'epoch', 'phase', 'loss'}.issubset(df.columns):
        print("[Plot] CSV missing required columns (epoch, phase, loss).")
        return

    # Group by epoch to get average loss
    train_loss = df[df['phase'] == 'train'].groupby('epoch')['loss'].mean()
    val_loss = df[df['phase'] == 'val'].groupby('epoch')['loss'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss.index, train_loss.values, label='Train Loss', marker='o')
    
    if not val_loss.empty:
        plt.plot(val_loss.index, val_loss.values, label='Validation Loss', marker='x')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    out_file = os.path.splitext(csv_path)[0] + '.png'
    plt.savefig(out_file)
    print(f"[Plot] Saved plot to {out_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_curves(sys.argv[1])
    else:
        print("Usage: python plot_loss.py <path_to_csv>")