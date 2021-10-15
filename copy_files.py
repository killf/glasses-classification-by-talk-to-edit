import random
import shutil
import click
import os


def random_choices(ls, k):
    indices = set()
    while len(indices) < k:
        idx = random.randint(0, len(ls) - 1)
        indices.add(idx)

    return [ls[idx] for idx in indices]


@click.command()
@click.option("--input-dir", help='The images folder.')
@click.option("--output-dir", help='The images folder.')
@click.option("--result-file", default="result.txt", help='The result file.')
def main(input_dir, output_dir, result_file):
    lines = open(result_file).readlines()
    lines = [line.strip().split(',') for line in lines if line.strip()]

    no_glasses, glasses = [], []
    for step, line in enumerate(lines):
        print(f"Progress: {step + 1}/{len(lines)}", end="\r", flush=True)

        file_name, label = line[0], line[2]  # glasses
        if label == "0":
            no_glasses.append(file_name)
            continue
        else:
            glasses.append(file_name)

        src_file = os.path.join(input_dir, file_name)
        dst_file = os.path.join(output_dir, label, file_name)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy(src_file, dst_file)
    print()

    if len(glasses) < len(no_glasses):
        no_glasses = random_choices(no_glasses, len(glasses))

    for step, file_name in enumerate(no_glasses):
        print(f"Progress: {step + 1}/{len(no_glasses)}", end="\r", flush=True)

        src_file = os.path.join(input_dir, file_name)
        dst_file = os.path.join(output_dir, "0", file_name)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy(src_file, dst_file)

    print("\nComplete!")


if __name__ == '__main__':
    main()
