from pathlib import Path
import argparse
import shutil

import pandas as pd
from PIL import Image
from tqdm import tqdm


def clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description='Build bbox-cropped CUB train/test image folders.')
    parser.add_argument('--dataset-root', default='./dataset-root')
    parser.add_argument('--clear', action='store_true', help='Clear existing generated crop folders before writing new crops.')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cub_root = dataset_root / 'CUB_200_2011'
    source_images_root = cub_root / 'images'
    output_root = dataset_root / 'cub200_cropped'
    train_root = output_root / 'train_cropped_augmented'
    test_root = output_root / 'test_cropped'

    if args.clear:
        clear_directory_contents(train_root)
        clear_directory_contents(test_root)
    else:
        train_root.mkdir(parents=True, exist_ok=True)
        test_root.mkdir(parents=True, exist_ok=True)

    images = pd.read_csv(cub_root / 'images.txt', sep=' ', names=['image_id', 'image_path'])
    splits = pd.read_csv(cub_root / 'train_test_split.txt', sep=' ', names=['image_id', 'is_training_img'])
    bboxes = pd.read_csv(cub_root / 'bounding_boxes.txt', sep=' ', names=['image_id', 'x', 'y', 'w', 'h'])

    metadata = images.merge(splits, on='image_id').merge(bboxes, on='image_id')

    train_count = 0
    test_count = 0

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc='Cropping CUB images'):
        rel_path = Path(row.image_path)
        src_path = source_images_root / rel_path
        dst_root = train_root if int(row.is_training_img) == 1 else test_root
        dst_path = dst_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        left = int(row.x)
        top = int(row.y)
        right = left + int(row.w)
        bottom = top + int(row.h)

        with Image.open(src_path).convert('RGB') as image:
            cropped = image.crop((left, top, right, bottom))
            cropped.save(dst_path, quality=95)

        if int(row.is_training_img) == 1:
            train_count += 1
        else:
            test_count += 1

    print('Finished building cropped CUB dataset')
    print(f'dataset_root: {dataset_root}')
    print(f'train images written: {train_count}')
    print(f'test images written: {test_count}')
    print(f'train output: {train_root}')
    print(f'test output: {test_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
