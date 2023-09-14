import argparse
import json
import os

import numpy as np
from pytablewriter import MarkdownTableWriter


# Creates markdown table for the given directory of lm-eval results


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory to list", default="./results")
args = parser.parse_args()


def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def main(args):
    file_paths = [fp for fp in os.listdir(args.dir) if fp.endswith(".json")]
    print(file_paths)

    task2name = {
        "arc_challenge": "ARC Challenge✱",
        "arc_easy": "ARC Easy✱",
        "lambada_openai": "LAMBADA OpenAI",
        "boolq": "BoolQ",
        "hellaswag": "HellaSwag✱",
        "openbookqa": "OpenBookQA",
        "piqa": "PIQA",
        # NOTE: SocialIQA IS VERY NOISY! If a model performs > 50% accuracy, assume it's overfitting
        # "siqa": "SocialIQA",
        "sciq": "SciQ",
        # "truthfulqa_mc": "TruthfulQA (MC)",
        "winogrande": "Winogrande",
    }
    task2metric = {
        "arc_challenge": "acc_norm",
        "arc_easy": "acc_norm",
        "lambada_openai": "acc",
        "boolq": "acc",
        "hellaswag": "acc_norm",
        "openbookqa": "acc",
        "piqa": "acc",
        # "siqa": "acc",
        "sciq": "acc",
        # "truthfulqa_mc": "mc2",
        "winogrande": "acc",
    }
    task_headers = []
    for task in sorted(task2name.keys()):
        # task_headers.append(f"{task2name[task]}\n({task2metric[task]})")
        task_headers.append(f"{task2name[task]}")
    headers = ["Model", "Average", *task_headers]
    rows = []
    for result in file_paths:
        data = read_json(os.path.join(args.dir, result))
        model_name = [k for k in data["config"]["model_args"].split(",") if "pretrained" in k][0].split("=")[1]
        row = [model_name]

        # Compute mean accuracy
        sum = 0
        for task in sorted(task2name.keys()):
            sum += data["results"][task][task2metric[task]]
        row.append(f"{(sum / len(task2name)) * 100.0:.2f}")

        for task in sorted(task2name.keys()):
            # score = f"{data['results'][task]['acc'] * 100:.2f} ± {data['results'][task]['acc_stderr'] * 100:.2f}"
            score = f"{data['results'][task][task2metric[task]] * 100:.2f}"
            row.append(score)

        rows.append(row)

    # Sort by average accuracy
    rows = sorted(rows, key=lambda x: float(x[1]), reverse=True)
    print(rows)

    # Print table in markdown
    writer = MarkdownTableWriter(
        table_name="Results",
        headers=headers,
        value_matrix=rows,
        margin=1,
        flavor="github",
    )
    writer.dump("guac0_results.md")


if __name__ == "__main__":
    main(args)
