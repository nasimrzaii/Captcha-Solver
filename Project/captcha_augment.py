import argparse
from pathlib import Path
import random

import cv2
import numpy as np


DEFAULT_CLASSES = [str(number) for number in range(1, 10)]


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def augment_image(image):
    augmented_images = []

    for angle in [-45, -30, -15, 0, 15, 30, 45]:
        augmented_images.append(rotate_image(image, angle))

    augmented_images.append(cv2.flip(image, 1))
    augmented_images.append(cv2.flip(image, 0))

    for angle in [-20, 20, 30]:
        augmented_images.append(cv2.flip(rotate_image(image, angle), 1))

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 2] *= np.random.uniform(0.6, 1.4)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
    augmented_images.append(cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR))

    noise = np.random.normal(0, 0.03, image.shape) * 255
    augmented_images.append(np.clip(image + noise, 0, 255).astype(np.uint8))

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmented_images.append(rotate_image(blurred, random.choice([-25, -10, 10, 25])))

    return augmented_images


def augment_dataset(input_dir, output_dir, target_per_class):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for class_name in DEFAULT_CLASSES:
        class_input = input_path / class_name
        class_output = output_path / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        images = []
        for image_file in sorted(class_input.glob('*')):
            image = cv2.imread(str(image_file))
            if image is not None:
                images.append(cv2.resize(image, (32, 32)))

        if not images:
            print(f'{class_name}: no source images found')
            continue

        final_images = list(images)
        image_index = 0
        while len(final_images) < target_per_class:
            for augmented in augment_image(images[image_index % len(images)]):
                final_images.append(augmented)
                if len(final_images) >= target_per_class:
                    break
            image_index += 1

        for index, image in enumerate(final_images[:target_per_class]):
            cv2.imwrite(str(class_output / f'{class_name}_{index:04d}.png'), image)

        print(f'{class_name}: saved {min(len(final_images), target_per_class)} images')


def main():
    parser = argparse.ArgumentParser(description='Create augmented single-digit captcha images.')
    parser.add_argument('--input', default='kaptcha', help='Source dataset folder.')
    parser.add_argument('--output', default='augmented_captcha', help='Output folder.')
    parser.add_argument('--target-per-class', type=int, default=1000, help='Images to save for each class.')
    args = parser.parse_args()
    augment_dataset(args.input, args.output, args.target_per_class)


if __name__ == '__main__':
    main()
