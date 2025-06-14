import os
import torch
from tqdm import tqdm

def evaluate(model, data_loader, device, metric_fns, is_sequence=True):
    """
    General evaluation function for sequence or classification tasks.

    Args:
        model: the model to evaluate
        data_loader: DataLoader for validation or test
        device: torch device
        metric_fns: dict mapping metric name to function(fn(y_true, y_pred) -> scalar)
        is_sequence: if True, treats outputs as sequence labeling and flattens all tokens

    Returns:
        results: dict of computed metric values
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            # Move inputs to device
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch['input_ids']
            y = batch['labels']
            attn_mask = batch['attention_mask']

            # Forward pass
            logits = model(x, key_padding_mask=(attn_mask == 0))
            preds = torch.argmax(logits, dim=-1)

            if is_sequence:
                # Flatten token-level predictions and labels, ignore padding index -100
                preds_flat = preds[y != -100]
                labels_flat = y[y != -100]
                all_preds.extend(preds_flat.cpu().tolist())
                all_labels.extend(labels_flat.cpu().tolist())
            else:
                # Flatten batch-level classification
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

    # Compute all metrics
    results = {}
    for name, fn in metric_fns.items():
        results[name] = fn(all_labels, all_preds)
    return results


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    save_dir,
    save_prefix,
    metric_fns,
    num_epochs=200,
    lr=1e-5,
    freeze_transformer=False,
    is_sequence=True,
    early_stopping=True,
    early_stopping_patience=25,
    metric_name="accuracy",
    greater_is_better=True
):
    """
    Train and evaluate a model with optional Early Stopping.

    Args:
        model: the model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: torch device
        save_dir: directory to save checkpoints
        save_prefix: filename prefix for checkpoints
        metric_fns: dict of evaluation metrics
        num_epochs: maximum number of epochs to train
        lr: learning rate
        freeze_transformer: whether to freeze transformer backbone
        is_sequence: True for sequence labeling, False for classification
        early_stopping: whether to enable Early Stopping
        early_stopping_patience: number of epochs with no improvement before stopping
        metric_name: name of metric in metric_fns to monitor
        greater_is_better: whether a higher metric value indicates improvement

    Returns:
        history: dict with 'val_metrics' list and 'test_metrics'
    """
    # Optionally freeze transformer backbone
    if freeze_transformer and hasattr(model, 'transformer'):
        for param in model.transformer.parameters():
            param.requires_grad = False

    # Choose loss function
    if is_sequence:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler()

    history = {'val_metrics': []}

    # --- Early Stopping variables ---
    best_metric = None
    patience_counter = 0
    best_model_state = None

    # Initial evaluation before any training
    init_metrics = evaluate(model, val_loader, device, metric_fns, is_sequence)
    print(f"Initial validation metrics: {init_metrics}")
    history['val_metrics'].append({'epoch': 0, **init_metrics})

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch['input_ids']
            y = batch['labels']
            attn_mask = batch['attention_mask']

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                logits = model(x, key_padding_mask=(attn_mask == 0))
                if is_sequence:
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    loss = criterion(logits, y.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation step
        val_metrics = evaluate(model, val_loader, device, metric_fns, is_sequence)
        print(f"Epoch {epoch}: {val_metrics}")
        history['val_metrics'].append({'epoch': epoch, **val_metrics})

        # --- Early Stopping check ---
        current_metric = val_metrics.get(metric_name)
        if current_metric is None:
            raise KeyError(f"Monitored metric '{metric_name}' not found in validation results.")

        improved = (
            best_metric is None or
            (greater_is_better and current_metric > best_metric) or
            (not greater_is_better and current_metric < best_metric)
        )

        if improved:
            best_metric = current_metric
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save a checkpoint of the current best model
            os.makedirs(save_dir, exist_ok=True)
            best_path = os.path.join(save_dir, f"{save_prefix}_best.pth")
            torch.save(best_model_state, best_path)
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

        # Optional periodic checkpointing
        if (epoch <= 30 and epoch % 10 == 0) or (epoch > 30 and epoch % 40 == 0):
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{save_prefix}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)

    # Load the best model state if early stopping was used
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, device, metric_fns, is_sequence)
    print(f"{save_prefix} final test metrics: {test_metrics}")
    history['test_metrics'] = test_metrics

    return history