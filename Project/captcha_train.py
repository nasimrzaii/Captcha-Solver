import argparse
from pathlib import Path

import cv2
import numpy as np
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


IMAGE_SIZE = (32, 32)


def load_dataset(dataset_dir):
    images = []
    labels = []

    for image_file in sorted(Path(dataset_dir).glob('*/*')):
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        images.append(cv2.resize(image, IMAGE_SIZE) / 255.0)
        labels.append(image_file.parent.name)

    if not images:
        raise ValueError(f'No images found in {dataset_dir}')

    label_binarizer = LabelBinarizer()
    encoded_labels = label_binarizer.fit_transform(labels)
    return np.array(images), encoded_labels, label_binarizer.classes_


def build_model(number_of_classes):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(number_of_classes, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train the captcha digit classifier.')
    parser.add_argument('--data', default='kaptcha', help='Dataset folder organized as class/image files.')
    parser.add_argument('--model', default='FinalModel_1.keras', help='Output Keras model path.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    images, labels, classes = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = build_model(len(classes))
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')
    model.save(args.model)


if __name__ == '__main__':
    main()
