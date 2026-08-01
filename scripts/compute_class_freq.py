import argparse
import json
import math
from collections import Counter
from pathlib import Path

DEFAULT_LABEL_DIR = Path("data/eightclass/datasets/RFT/labels/train")
DEFAULT_OUTPUT_PATH = Path("results/tables/class_weights.json")

CLASS_NAMES = [
    'bottle', 'grass', 'branch', 'milk-box',
    'plastic-bag', 'plastic-garbage', 'ball', 'leaf'
]

def compute_class_frequencies(label_dir: Path):
    counts = Counter()
    
    label_files = list(label_dir.glob("*.txt"))
    for filepath in label_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] += 1

    results = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = counts.get(class_id, 0)
        if count > 0:
            inv_freq = 1.0 / count
            inv_sqrt_freq = 1.0 / math.sqrt(count)
            inv_log_freq = 1.0 / math.log(count + 1)
        else:
            inv_freq = 0.0
            inv_sqrt_freq = 0.0
            inv_log_freq = 0.0

        results[class_name] = {
            "count": count,
            "inv_freq": inv_freq,
            "inv_sqrt_freq": inv_sqrt_freq,
            "inv_log_freq": inv_log_freq
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Compute class frequencies and weighting schemes from YOLO labels.")
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Path to directory containing YOLO format .txt label files."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON file."
    )
    args = parser.parse_args()

    results = compute_class_frequencies(args.label_dir)

    # Print clean table
    print(f"\n{'Class Name':<18} | {'Count':<8} | {'Inv Freq':<12} | {'Inv Sqrt Freq':<14} | {'Inv Log Freq':<12}")
    print("-" * 75)
    for class_name, data in results.items():
        print(
            f"{class_name:<18} | "
            f"{data['count']:<8} | "
            f"{data['inv_freq']:<12.6e} | "
            f"{data['inv_sqrt_freq']:<14.6f} | "
            f"{data['inv_log_freq']:<12.6f}"
        )
    print("-" * 75)

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved class weights to {args.output}")

if __name__ == "__main__":
    main()
