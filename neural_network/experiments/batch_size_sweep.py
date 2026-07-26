from dataclasses import dataclass
from time import perf_counter
import numpy as np

@dataclass
class SweepConfig:
    batch_sizes: list[int]      # e.g. [1, 4, 16, 64, 256, 1024]
    seeds: list[int]            # e.g. [0, 1, 2]
    budget_type: str            # "epochs" or "steps"
    budget_value: int           # e.g. 30 epochs
    base_learning_rate: float
    learning_rate_rule: str     # "fixed" initially
    shuffle: bool = True


@dataclass
class RunResult:
    seed: int
    batch_size: int
    learning_rate: float
    budget_type: str
    budget_value: int

    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float

    optimiser_steps: int
    examples_seen: int
    elapsed_seconds: float

def _validate_dataset(X, y, name):
    if len(X) != len(y):
        raise ValueError(
            f"{name}: X and y must have the same number of examples."
        )
    if len(X) == 0:
        raise ValueError(f"{name}: dataset must not be empty.")

def _loss_forward(loss, prediction, target):
    if hasattr(loss, "forward"):
        value = loss.forward(prediction, target)
    else:
        value = loss(prediction, target)

    return float(np.mean(value))

def _classification_accuracy(prediction, target):
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    is_binary = prediction.ndim == 1 or prediction.shape[-1] == 1

    if is_binary:
        prediction_labels = (prediction.reshape(-1) >= .5).astype(int)
        true_labels = target.reshape(-1).astype(int)
    else:
        prediction_labels = np.argmax(prediction, axis=1)
        if target.ndim == 2:
            true_labels = np.argmax(target, axis=1)
        else:
            true_labels = target.astype(int)

    return float(np.mean(prediction_labels == true_labels))

def _iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    """Yield every example exactly once, in batches, for one epoch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    n_examples = len(X)
    indices = np.arange(n_examples)

    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n_examples, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]


def run_batch_size_sweep(
    X_train,
    y_train,
    X_validation,
    y_validation,
    model_factory,
    optimiser_factory,
    loss,
    config: SweepConfig,
) -> list[RunResult]:
    if config.learning_rate_rule != "fixed":
        raise ValueError(
            "Only learning_rate_rule='fixed' is implemented for now."
        )

    if not config.batch_sizes:
        raise ValueError("batch_sizes must not be empty.")

    if not config.seeds:
        raise ValueError("seeds must not be empty.")

    results = []

    for batch_size in config.batch_sizes:
        learning_rate = config.base_learning_rate

        for seed in config.seeds:
            result = run_one_training(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                model_factory=model_factory,
                optimiser_factory=optimiser_factory,
                loss=loss,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                budget_type=config.budget_type,
                budget_value=config.budget_value,
                shuffle=config.shuffle,
            )
            results.append(result)

    return results

def run_one_training(
    X_train,
    y_train,
    X_validation,
    y_validation,
    model_factory,
    optimiser_factory,
    loss,
    batch_size,
    learning_rate,
    seed,
    budget_type,
    budget_value,
    shuffle,
) -> RunResult:
    _validate_dataset(X_train, y_train, "training set")
    _validate_dataset(X_validation, y_validation, "validation set")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if budget_type not in {"epochs", "steps"}:
        raise ValueError("budget_type must be either 'epochs' or 'steps'.")

    if budget_value <= 0:
        raise ValueError("budget_value must be positive.")

    # Same seed means the same initial model for every batch-size comparison.
    model = model_factory(seed)
    optimiser = optimiser_factory(learning_rate)

    # Separate RNG for data order, avoiding hidden global randomness.
    rng = np.random.default_rng(seed)

    optimiser_steps = 0
    examples_seen = 0

    start_time = perf_counter()

    if budget_type == "epochs":
        for _ in range(budget_value):
            for X_batch, y_batch in _iterate_minibatches(
                X_train, y_train, batch_size, rng, shuffle
            ):
                train_one_batch(model, optimiser, loss, X_batch, y_batch)

                optimiser_steps += 1
                examples_seen += len(X_batch)

    elif budget_type == "steps":
        batch_iterator = iter(
            _iterate_minibatches(
                X_train, y_train, batch_size, rng, shuffle
            )
        )

        while optimiser_steps < budget_value:
            try:
                X_batch, y_batch = next(batch_iterator)
            except StopIteration:
                # One epoch ended; create a newly shuffled epoch.
                batch_iterator = iter(
                    _iterate_minibatches(
                        X_train, y_train, batch_size, rng, shuffle
                    )
                )
                X_batch, y_batch = next(batch_iterator)

            train_one_batch(model, optimiser, loss, X_batch, y_batch)

            optimiser_steps += 1
            examples_seen += len(X_batch)

    elapsed_seconds = perf_counter() - start_time

    train_loss, train_accuracy = evaluate(model, X_train, y_train, loss)
    validation_loss, validation_accuracy = evaluate(
        model, X_validation, y_validation, loss
    )

    return RunResult(
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        budget_type=budget_type,
        budget_value=budget_value,
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        optimiser_steps=optimiser_steps,
        examples_seen=examples_seen,
        elapsed_seconds=elapsed_seconds,
    )

def iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    """Yield each example once per epoch, in batches."""
    ...

def train_one_batch(model, optimiser, loss, X_batch, y_batch):
    """
    Run one complete learning update.

    Assumes:
        model.backward(d_loss_d_prediction)
        model.parameters()
        model.gradients()
        optimiser.step(parameters, gradients)
    """
    prediction = model.forward(X_batch)

    batch_loss = _loss_forward(loss, prediction, y_batch)
    d_loss_d_prediction = loss.backward(prediction, y_batch)

    model.backward(d_loss_d_prediction)

    optimiser.step(
        model.parameters(),
        model.gradients(),
    )

    return batch_loss

def evaluate(model, X, y, loss):
    """Evaluate the whole dataset without changing parameters."""
    prediction = model.forward(X)
    dataset_loss = _loss_forward(loss, prediction, y)
    accuracy = _classification_accuracy(prediction, y)

    return dataset_loss, accuracy

def summarise_results(results):
    """
    Return a dictionary grouped by batch size.

    For each numerical metric:
        {"mean": ..., "std": ...}
    """
    if not results:
        raise ValueError("results must not be empty.")

    metrics = [
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "optimiser_steps",
        "examples_seen",
        "elapsed_seconds",
    ]

    grouped = {}

    for result in results:
        grouped.setdefault(result.batch_size, []).append(result)

    summary = {}

    for batch_size, runs in grouped.items():
        batch_summary = {
            "n_runs": len(runs),
            "learning_rate": runs[0].learning_rate,
        }

        for metric in metrics:
            values = np.array(
                [getattr(run, metric) for run in runs],
                dtype=float,
            )

            # ddof=1 gives sample standard deviation.
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            batch_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": std,
            }

        summary[batch_size] = batch_summary

    return summary