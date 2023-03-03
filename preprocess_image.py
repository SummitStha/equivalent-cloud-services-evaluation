import os
import cv2
import numpy as np
from PIL import Image


def scale_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a new directory for the scaled images
    output_directory = "scaled_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Apply scaling steps from 10% to 90% with steps of 10%
    for scale_percent in range(10, 100, 10):
        # Calculate the new dimensions of the image based on the scaling percentage
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        # Resize the image using the calculated dimensions
        scaled_image = cv2.resize(image, (width, height))

        # Save the scaled image to a file
        output_filename = f"scaled_{scale_percent}percent.jpg"
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, scaled_image)


def blur_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a new directory for the blurred images
    output_directory = "blurred_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Apply blurring steps from 10 to 90 with steps of 10
    for blur_step in range(10, 100, 10):
        # Apply the normalized box filter with kernel size equal to blur_step
        blurred_image = cv2.boxFilter(image, -1, (blur_step, blur_step), normalize=True)

        # Save the blurred image to a file
        output_filename = f"blurred_{blur_step}percent.jpg"
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, blurred_image)


def gamma_correction(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a new directory for the gamma-corrected images
    output_directory = "gamma_corrected_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define gamma correction values to use
    gamma_values = [0.25, 1.5, 3.0]

    # Apply gamma correction using each value and save the resulting images
    for gamma in gamma_values:
        # Calculate the gamma correction lookup table
        gamma_inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma_inv) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # Apply the gamma correction to the image
        gamma_corrected = cv2.LUT(image, table)

        # Save the gamma-corrected image to a file
        output_filename = f"gamma_{gamma}.jpg"
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, gamma_corrected)


def salt_and_pepper_noise(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a new directory for the noisy images
    output_directory = "salt_and_pepper_noise_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Apply salt and pepper noise to the image with different probabilities
    for i in range(1, 10):
        noise_probability = i * 0.1
        noisy_image = np.copy(image)
        height, width, channels = image.shape
        noise_mask = np.random.choice([0, 1, 2], size=(height, width),
                                       p=[1 - noise_probability, noise_probability / 2, noise_probability / 2])
        noisy_image[noise_mask == 1] = [255, 255, 255]  # Salt noise
        noisy_image[noise_mask == 2] = [0, 0, 0]  # Pepper noise

        # Save the noisy image to a file
        output_path = os.path.join(output_directory, f"noisy_image_{i}.jpg")
        cv2.imwrite(output_path, noisy_image)


def preprocess_image(image):
    scale_image(image)
    blur_image(image)
    gamma_correction(image)
    salt_and_pepper_noise(image)


if __name__ == "__main__":
	preprocess_image("test.jpg")

