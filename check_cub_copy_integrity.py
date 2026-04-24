from pathlib import Path
import argparse
from collections import Counter


def load_image_paths(images_file: Path) -> dict[int, str]:
    image_id_to_rel_path: dict[int, str] = {}
    with images_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_id_str, rel_path = line.strip().split(" ", 1)
            image_id_to_rel_path[int(image_id_str)] = rel_path
    return image_id_to_rel_path


def load_expected_locations(cub_dir: Path, copied_root: Path) -> dict[str, Path]:
    image_id_to_rel_path = load_image_paths(cub_dir / "images.txt")
    expected: dict[str, Path] = {}

    with (cub_dir / "train_test_split.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            image_id_str, is_train_str = line.strip().split()
            rel_path = Path(image_id_to_rel_path[int(image_id_str)])
            split_dir = "train_cropped_augmented" if int(is_train_str) == 1 else "test_cropped"
            expected[rel_path.as_posix()] = copied_root / split_dir / rel_path

    return expected


def collect_actual_files(split_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not split_dir.exists():
        return files

    for file_path in split_dir.rglob("*.jpg"):
        rel_path = file_path.relative_to(split_dir).as_posix()
        files[rel_path] = file_path

    return files


def format_examples(items: list[str], limit: int) -> str:
    if not items:
        return "  none"
    return "\n".join(f"  - {item}" for item in items[:limit])


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that copied CUB images are in the expected split/class folder.")
    parser.add_argument("--dataset-root", default="./dataset-root")
    parser.add_argument("--show", type=int, default=10, help="Number of example paths to print per category")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cub_dir = dataset_root / "CUB_200_2011"
    copied_root = dataset_root / "cub200_cropped"
    train_dir = copied_root / "train_cropped_augmented"
    test_dir = copied_root / "test_cropped"

    expected_locations = load_expected_locations(cub_dir, copied_root)
    actual_train = collect_actual_files(train_dir)
    actual_test = collect_actual_files(test_dir)

    duplicate_rel_paths = sorted(set(actual_train) & set(actual_test))

    missing_expected: list[str] = []
    wrong_split: list[str] = []
    present_expected = 0

    for rel_path, expected_path in expected_locations.items():
        in_train = rel_path in actual_train
        in_test = rel_path in actual_test

        if expected_path.parent.parent.name == "train_cropped_augmented":
            should_be_train = True
        else:
            should_be_train = False

        if should_be_train:
            if in_train:
                present_expected += 1
            elif in_test:
                wrong_split.append(f"{rel_path} -> found in test, expected train")
            else:
                missing_expected.append(rel_path)
        else:
            if in_test:
                present_expected += 1
            elif in_train:
                wrong_split.append(f"{rel_path} -> found in train, expected test")
            else:
                missing_expected.append(rel_path)

    expected_set = set(expected_locations)
    unexpected_train = sorted(rel_path for rel_path in actual_train if rel_path not in expected_set)
    unexpected_test = sorted(rel_path for rel_path in actual_test if rel_path not in expected_set)

    class_counts = Counter(path.split("/", 1)[0] for path in expected_locations)

    print("CUB copy integrity check")
    print(f"dataset_root: {dataset_root}")
    print(f"expected images from metadata: {len(expected_locations)}")
    print(f"expected classes from metadata: {len(class_counts)}")
    print(f"actual train jpg files: {len(actual_train)}")
    print(f"actual test jpg files: {len(actual_test)}")
    print(f"expected images present in correct split: {present_expected}")
    print(f"missing expected images: {len(missing_expected)}")
    print(f"wrong-split images: {len(wrong_split)}")
    print(f"duplicate relpaths across train/test: {len(duplicate_rel_paths)}")
    print(f"unexpected train files: {len(unexpected_train)}")
    print(f"unexpected test files: {len(unexpected_test)}")

    print("\nExamples: missing expected")
    print(format_examples(missing_expected, args.show))
    print("\nExamples: wrong split")
    print(format_examples(wrong_split, args.show))
    print("\nExamples: duplicates across splits")
    print(format_examples(duplicate_rel_paths, args.show))
    print("\nExamples: unexpected train files")
    print(format_examples(unexpected_train, args.show))
    print("\nExamples: unexpected test files")
    print(format_examples(unexpected_test, args.show))

    failures = any([
        missing_expected,
        wrong_split,
        duplicate_rel_paths,
        unexpected_train,
        unexpected_test,
    ])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
