from pathlib import Path
from typing import Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
from IPython.display import display


def visualization(checkpoints_path: Union[Path, str], tests_path: Union[Path, str]):
    def get_test_files(dir_path):
        p = Path(dir_path)
        if not p.exists():
            return []
        else:
            files = list(p.glob("*.pt"))
            return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

    def plot_train_val(metric):
        with out_train_val:
            out_train_val.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            metric_str = str(metric)
            train_vals = train_metrics_dict.get(metric_str, [])
            val_vals = val_metrics_dict.get(metric_str, [])
            if not train_vals or not val_vals:
                print(f"Metric {metric_str} not found")
                return
            epochs = range(1, len(train_vals) + 1)
            ax.plot(epochs, train_vals, "b-", label="Train", linewidth=2)
            ax.plot(epochs, val_vals, "r-", label="Validation", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_str.capitalize())
            ax.set_title(f"{metric_str.capitalize()} over epochs")
            ax.legend()
            ax.grid(True, alpha=0.3)
            display(fig)
            plt.close(fig)

    def show_test(file_path):
        try:
            metrics = torch.load(file_path, weights_only=False)
            with out_test:
                out_test.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                names = list(metrics.keys())
                values = [
                    metrics[k] if isinstance(metrics[k], (int, float)) else 0
                    for k in names
                ]
                ax.bar(names, values, color="skyblue", edgecolor="navy")
                ax.set_ylabel("Value")
                ax.set_title(f"Test metrics for file: {file_path.name}")
                ax.grid(axis="y", alpha=0.3)
                plt.xticks(rotation=45, ha="right")
                display(fig)
                plt.close(fig)
        except Exception as ex:
            print(
                f"Error processing test metrics from file {file_path}: {ex}. "
                "File skipped."
            )

    try:
        checkpoint = torch.load(checkpoints_path, weights_only=False)
        history = checkpoint["metrics_history"]
        train_metrics_dict = dict(history["train"])
        val_metrics_dict = dict(history["val"])
    except Exception as ex:
        print(f"Error loading checkpoint {checkpoints_path}: {ex}")
        return

    out_train_val = widgets.Output()

    train_val_metrics_lst = list(train_metrics_dict.keys())
    if not train_val_metrics_lst:
        print("No metrics to display")
        return

    train_val_metric_selector = widgets.Dropdown(
        options=train_val_metrics_lst,
        value=train_val_metrics_lst[0],
        description="Metric:",
        disabled=False,
    )
    train_val_metric_selector.observe(
        lambda change: plot_train_val(change["new"]), names="value"
    )

    tab1 = widgets.VBox([train_val_metric_selector, out_train_val])

    test_files_lst = get_test_files(tests_path)
    if len(test_files_lst) > 0:
        out_test = widgets.Output()

        test_metric_selector = widgets.Dropdown(
            options=test_files_lst,
            value=test_files_lst[0],
            description="File:",
            disabled=False,
        )
        test_metric_selector.observe(
            lambda change: show_test(change["new"]), names="value"
        )

        tab2 = widgets.VBox([test_metric_selector, out_test])

        tab = widgets.Tab([tab1, tab2])
        tab.set_title(0, "Train / Val")
        tab.set_title(1, "Test")
        show_test(test_files_lst[0])
    else:
        print(f"Test files not found in directory: {tests_path}")
        tab = widgets.Tab([tab1])
        tab.set_title(0, "Train / Val")

    plot_train_val(train_val_metrics_lst[0])

    display(tab)
