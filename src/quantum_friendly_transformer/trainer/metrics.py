"""Classification metrics for optional research training workflows.

Install the ``experiments`` extra for NumPy and scikit-learn.
"""


def compute_metrics(eval_preds):
    """Binary classification metrics for (model_outputs, labels)."""
    import numpy as np

    all_model_outputs, labels = eval_preds  # all_model_outputs is a tuple
    # all_model_outputs[0] is the real logits,
    # and all_model_outputs[1] is hidden_states
        # Check if hidden states are included. If so, logits should be the first element.
    if isinstance(all_model_outputs, (list, tuple)):
        logits = all_model_outputs[0]
    else:
        logits = all_model_outputs

    predictions = np.argmax(logits, axis=-1)

    from sklearn.metrics import f1_score, accuracy_score
    return {
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions)
    }
