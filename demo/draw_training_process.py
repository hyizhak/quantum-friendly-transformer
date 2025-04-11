import sys
import re
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MODEL_NAME = ['vanilla', 'sn_model', 'fn_model', 'fn_layernorm', 'subnorm_model',  'attn_sn_model', 'ffn_sn_model', 'attn_fn_model', 'ffn_fn_model']

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <log_file.txt>")
        sys.exit(1)

    log_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(log_file))[0]

    # Regex to extract: epoch, f1, accuracy
    # Matches lines like:
    #  "epoch: 2, metrics: {'f1': 0.475297678386248, 'accuracy': 0.8359342550287783}"
    pattern = re.compile(
        r"epoch:\s*(\d+),\s*metrics:\s*\{.*'f1':\s*([\d\.]+).+'accuracy':\s*([\d\.]+).*"
    )

    # Dictionary to store data for *any* number of models
    models_data = {}

    current_model = None

    # Read the log file
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Try to match our epoch-f1-accuracy pattern
            match = pattern.match(line)
            if match:
                # This line has epoch/metrics
                epoch_val = int(match.group(1))
                f1_val = float(match.group(2))
                acc_val = float(match.group(3))

                # Only record metrics if we have a current_model set
                if current_model is not None:
                    models_data[current_model]['epochs'].append(epoch_val)
                    models_data[current_model]['f1'].append(f1_val)
                    models_data[current_model]['accuracy'].append(acc_val)
            else:
                if line in MODEL_NAME:
                    current_model = line
                    # If this model hasn't been seen before, initialize its data
                    if current_model not in models_data:
                        models_data[current_model] = {
                            'epochs': [],
                            'f1': [],
                            'accuracy': []
                        }

    # --- Plot F1 for each model ---
    plt.figure()
    for model_name, data in models_data.items():
        plt.plot(data['epochs'], data['f1'], label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 over Epochs")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    f1_filename = f"./demo/{base_name}_f1.png"
    plt.savefig(f1_filename)
    print(f"Saved F1 plot to {f1_filename}")
    plt.close()

    # --- Plot Accuracy for each model ---
    plt.figure()
    for model_name, data in models_data.items():
        plt.plot(data['epochs'], data['accuracy'], label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    acc_filename = f"./demo/{base_name}_accuracy.png"
    plt.savefig(acc_filename)
    print(f"Saved Accuracy plot to {acc_filename}")
    plt.close()

if __name__ == "__main__":
    main()