import argparse
from pathlib import Path

import cv2
import numpy as np
from keras import models


IMAGE_SIZE = (32, 32)


def find_digit_regions(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    return sorted(boxes, key=lambda box: box[0])


def predict_captcha(model_path, image_path, output_path=None):
    model = models.load_model(model_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Could not read image: {image_path}')

    predictions = []
    for x, y, width, height in find_digit_regions(image):
        x1 = max(x - 4, 0)
        y1 = max(y - 4, 0)
        x2 = min(x + width + 5, image.shape[1])
        y2 = min(y + height + 5, image.shape[0])

        roi = cv2.resize(image[y1:y2, x1:x2], IMAGE_SIZE) / 255.0
        output = model.predict(np.array([roi]), verbose=0)
        digit = str(np.argmax(output) + 1)
        predictions.append(digit)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, digit, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if output_path:
        cv2.imwrite(str(output_path), image)

    return ''.join(predictions)


def main():
    parser = argparse.ArgumentParser(description='Predict digits in a captcha image.')
    parser.add_argument('image', nargs='?', default='testCaptcha_2.png', help='Captcha image path.')
    parser.add_argument('--model', default='FinalModel_1.keras', help='Keras model path.')
    parser.add_argument('--output', default=None, help='Optional annotated output image path.')
    args = parser.parse_args()

    prediction = predict_captcha(args.model, args.image, args.output)
    print(f'Prediction: {prediction}')


if __name__ == '__main__':
    main()
