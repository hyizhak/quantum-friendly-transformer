import sys
import re
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <log_file.txt>")
        sys.exit(1)

    log_file = sys.argv[1]
    # Extract a base name to use for saving plots, e.g. 'my_log' from 'my_log.txt'
    base_name = os.path.splitext(os.path.basename(log_file))[0]

    # Storage for metrics
    model1_epochs = []
    model1_f1 = []
    model1_acc = []

    model2_epochs = []
    model2_f1 = []
    model2_acc = []

    current_model = None
    plot_type = None

    # Regex to extract: epoch, f1, accuracy
    # Example lines: "epoch: 2, metrics: {'f1': 0.475297678386248, 'accuracy': 0.8359342550287783}"
    pattern = re.compile(
        r"epoch:\s*(\d+),\s*metrics:\s*\{.*'f1':\s*([\d\.]+).+'accuracy':\s*([\d\.]+).*"
    )

    # Read the log file
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect if we just hit a "vanilla" or "sn_model" line
            if line.startswith("vanilla"):
                current_model = "vanilla"
                plot_type = "train"
                continue
            elif line.startswith("sn_model"):
                current_model = "sn_model"
                plot_type = "train"
                continue
            elif line.startswith("attn_normalized_model"):
                current_model = "attn_normalized_model"
                plot_type = "fine-tune"
                continue
            elif line.startswith("ffn_normalized_model"):
                current_model = "ffn_normalized_model"
                plot_type = "fine-tune"
                continue

            # Attempt to parse lines with epoch/f1/accuracy
            match = pattern.match(line)
            if match:
                epoch_val = int(match.group(1))
                f1_val = float(match.group(2))
                acc_val = float(match.group(3))

                if current_model == "vanilla" or current_model == "attn_normalized_model":
                    model1_epochs.append(epoch_val)
                    model1_f1.append(f1_val)
                    model1_acc.append(acc_val)
                elif current_model == "sn_model" or current_model == "ffn_normalized_model":
                    model2_epochs.append(epoch_val)
                    model2_f1.append(f1_val)
                    model2_acc.append(acc_val)

    label1 = "vanilla model" if plot_type == "train" else "attn-normalized"
    label2 = "spectral-normalized" if plot_type == "train" else "ffn-normalzied"
    # --- Plot F1 ---
    plt.figure()
    plt.plot(model1_epochs, model1_f1, label=label1)
    plt.plot(model2_epochs, model2_f1, label=label2)
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 over Epochs")
    plt.legend()

    # Force x-axis to only use integer ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save figure based on the input filename
    f1_filename = f"./demo/{base_name}_f1.png"
    plt.savefig(f1_filename)
    print(f"Saved F1 plot to {f1_filename}")
    plt.close()

    # --- Plot Accuracy ---
    plt.figure()
    plt.plot(model1_epochs, model1_acc, label=label1)
    plt.plot(model2_epochs, model2_acc, label=label2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    # Force x-axis to only use integer ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save figure based on the input filename
    acc_filename = f"./demo/{base_name}_accuracy.png"
    plt.savefig(acc_filename)
    print(f"Saved Accuracy plot to {acc_filename}")
    plt.close()

if __name__ == "__main__":
    main()