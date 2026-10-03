"""Print the bounded experiment commands; never launch jobs or cloud resources."""

import argparse
import json
from pathlib import Path
import shlex


def commands(plan, output):
    result = []
    for stage in plan["stages"]:
        for lane in stage["lanes"]:
            settings = {**plan["common"], **stage["overrides"]}
            cmd = ["python", "-m", "src.models.research", "--lane", lane,
                   "--output", str(output / stage["name"] / lane)]
            for name, value in settings.items():
                cmd.extend(["--" + name.replace("_", "-"), str(value)])
            result.append(cmd)
    if len(result) > plan["max_screen_runs"]:
        raise ValueError("Plan exceeds its screening budget")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("configs/local_tuning_plan.json"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/mlp-tuning"))
    args = p.parse_args()
    for cmd in commands(json.loads(args.plan.read_text()), args.output_root):
        print(shlex.join(cmd))


if __name__ == "__main__":
    main()
