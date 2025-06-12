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
        metrics: dict of computed metric values
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch['input_ids']
            y = batch['labels']
            attn_mask = batch['attention_mask']

            logits = model(x, key_padding_mask=(attn_mask == 0))
            preds = torch.argmax(logits, dim=-1)

            if is_sequence:
                # Flatten token-level preds and labels, ignore padding index -100
                preds_flat = preds[y != -100]
                labels_flat = y[y != -100]
                all_preds.extend(preds_flat.cpu().tolist())
                all_labels.extend(labels_flat.cpu().tolist())
            else:
                # Flatten batch-level classification
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

    # Compute metrics
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
    freeze_transformer=True,
    is_sequence=True
):
    """
    Train and evaluate a model for sequence labeling or classification.

    Args:
        model: the model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: torch device
        save_dir: directory to save checkpoints
        save_prefix: filename prefix for checkpoints
        metric_fns: dict of evaluation metrics
        num_epochs: max epochs
        lr: learning rate
        freeze_transformer: whether to freeze transformer backbone
        is_sequence: True for seq labeling, False for classification

    Returns:
        history: dict with 'val_metrics' list and 'test_metrics'
    """
    # Freeze transformer if needed
    if freeze_transformer and hasattr(model, 'transformer'):
        for param in model.transformer.parameters():
            param.requires_grad = False

    if is_sequence:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler()

    history = {'val_metrics': []}

    # Initial evaluation
    init_metrics = evaluate(model, val_loader, device, metric_fns, is_sequence)
    print(f"Initial validation metrics: {init_metrics}")
    history['val_metrics'].append({'epoch': 0, **init_metrics})

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        for batch in tqdm(train_loader):
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

        # Validation
        val_metrics = evaluate(model, val_loader, device, metric_fns, is_sequence)
        print(f"Epoch {epoch}: {val_metrics}")
        history['val_metrics'].append({'epoch': epoch, **val_metrics})

        # Save checkpoint
        if (epoch <= 30 and epoch % 10 == 0) or (epoch > 30 and epoch % 40 == 0):
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{save_prefix}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, device, metric_fns, is_sequence)
    print(f"{save_prefix} final metrics: {test_metrics}")
    history['test_metrics'] = test_metrics

    return history
