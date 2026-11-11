"""Optional Hugging Face Trainer callback; install the ``hf`` extra."""

try:
    from transformers import TrainingArguments, TrainerCallback, TrainerControl, TrainerState
except ModuleNotFoundError as exc:
    if exc.name == "transformers":
        raise ImportError(
            "Trainer callbacks require 'quantum-friendly-transformer[hf]'."
        ) from exc
    raise


class PrintMetricsCallback(TrainerCallback):
    """
    A callback to automatically run after each evaluation and
    print the metrics in your custom format, e.g.:
        epoch: X, metrics: {'f1': ..., 'accuracy': ...}
    """
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs
    ):
        # The `metrics` dict has keys like "eval_loss", "eval_f1", "eval_accuracy", "epoch", etc.
        # We'll grab `f1` and `accuracy` if they exist.
        epoch = metrics.get("epoch", state.epoch)  # Typically stored in 'epoch'
        if epoch is None:
            epoch = 0.0
        f1 = metrics.get("eval_f1", None)
        acc = metrics.get("eval_accuracy", None)

        if f1 is not None and acc is not None:
            print(f"epoch: {int(epoch)}, metrics: {{'f1': {f1}, 'accuracy': {acc}}}")
        else:
            # If your metric keys differ or you want to see all metric keys:
            print(f"epoch: {int(epoch)}, all metrics: {metrics}")

        return control
