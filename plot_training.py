"""
CLI to plot training and eval results.

Usage:
    python plot_training.py
    python plot_training.py training_results.json -o ./figures
"""
import argparse
import json
import os

from graphs import plot_training, plot_eval


def main():
    parser = argparse.ArgumentParser(description="Plot training and eval results")
    parser.add_argument("file", nargs="?", default="training_results.json", help="JSON file with results")
    parser.add_argument("--no-save", action="store_true", help="Show only, don't save")
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        print("Run 'python run_train.py' first to generate training_results.json")
        return

    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)

    save = not args.no_save
    show = not args.no_show
    os.makedirs(args.out_dir, exist_ok=True)

    if "history" in data:
        plot_training(data["history"], args.out_dir, save, show)

    if "eval" in data:
        eval_data = data["eval"]
        plot_eval(
            eval_data["all_wins"],
            eval_data["n_rounds"],
            eval_data.get("agent_heuristic", {}),
            args.out_dir,
            save,
            show,
        )


if __name__ == "__main__":
    main()
