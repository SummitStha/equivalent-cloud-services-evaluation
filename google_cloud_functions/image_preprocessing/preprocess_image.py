import random
import string
import tempfile
import os
import cv2
import requests
import numpy as np
from timeit import default_timer as timer

from google.cloud import storage, firestore
# from google.oauth2 import id_token
# from google.auth.transport import requests as g_requests

# Create a Google Cloud Storage and Firestore client
storage_client = storage.Client()
firestore_client = firestore.Client()
output_bucket_name = os.getenv('BUCKET_NAME')
output_bucket = storage_client.bucket(output_bucket_name)


def generate_random_string(length):
    # Define the possible characters to use in the string
    characters = string.ascii_letters + string.digits
    
    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for i in range(length))
    
    return random_string


def upload_preprocessed_image(preprocessed_image, file_name, temp_local_filename, random_string, operation_performed, steps):
    ''' Upload the preprocessed image to another bucket'''
    filename, ext = file_name.split('.')
    output_filename = f"{filename}_{random_string}/{operation_performed}_{steps}.{ext}"
    filepath = temp_local_filename + ".jpg"
    print(f"Filepath: {filepath}")
    cv2.imwrite(filepath, preprocessed_image)

    new_blob = output_bucket.blob(output_filename)
    new_blob.upload_from_filename(filepath)
    preprocessed_image_path = f'https://storage.googleapis.com/{output_bucket}/{output_filename}'
    print(f'{operation_performed} image uploaded to: gs://{output_bucket_name}/{output_filename}')

    os.remove(filepath)

    return str(preprocessed_image_path)


def scale_image(image, file_name, temp_local_filename, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing

    # Apply scaling steps from 10 to 100 with steps of 20
    for scale_percent in range(10, 100, 20):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        scaled_image = cv2.resize(image, (width, height))
        preprocessed_image_path = upload_preprocessed_image(scaled_image, file_name, temp_local_filename, random_string, 'scaled', scale_percent)

        preprocessing_metrics[f'scaled_{scale_percent}_path'] = preprocessed_image_path

    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['scaling_operation_time'] = processing_time


def blur_image(image, file_name, temp_local_filename, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing

    # Apply blurring steps from 10 to 100 with steps of 20
    for blur_step in range(1, 20, 4):
        # Apply the normalized box filter with kernel size equal to blur_step
        blurred_image = cv2.boxFilter(image, -1, (blur_step, blur_step), normalize=True)
        preprocessed_image_path = upload_preprocessed_image(blurred_image, file_name, temp_local_filename, random_string, 'blurred', blur_step)

        preprocessing_metrics[f'blurred_{blur_step}_path'] = preprocessed_image_path
    
    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['bluring_operation_time'] = processing_time


def adjust_brightness(image, file_name, temp_local_filename, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing
   
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
        preprocessed_image_path = upload_preprocessed_image(gamma_corrected, file_name, temp_local_filename, random_string, 'brightness', gamma)

        preprocessing_metrics[f'brightness_{gamma}_path'] = preprocessed_image_path
    
    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['brightness_adjustment_time'] = processing_time


def adjust_salt_and_pepper_noise(image, file_name, temp_local_filename, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing

    # Apply salt and pepper noise to the image with different probabilities
    for value in range(1, 10, 2):
        noise_probability = value * 0.1
        noisy_image = np.copy(image)
        height, width, channels = image.shape
        noise_mask = np.random.choice([0, 1, 2], size=(height, width),
                                       p=[1 - noise_probability, noise_probability / 2, noise_probability / 2])
        noisy_image[noise_mask == 1] = [255, 255, 255]  # Salt noise
        noisy_image[noise_mask == 2] = [0, 0, 0]  # Pepper noise

        preprocessed_image_path = upload_preprocessed_image(noisy_image, file_name, temp_local_filename, random_string, 'noise', value)

        preprocessing_metrics[f'noise_{value}_path'] = preprocessed_image_path
    
    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['noise_adjustment_time'] = processing_time


def preprocess_image(data, context):
    start_time = timer()  # Record the start time of the processing

    preprocessing_metrics = {}

    file_data = data
    file_name = file_data['name']
    bucket_name = file_data['bucket']

    blob = storage_client.bucket(bucket_name).get_blob(file_name)
    file_name = blob.name
    _, temp_local_filename = tempfile.mkstemp()
    print(f"local filenmame: {temp_local_filename}")

    blob.download_to_filename(temp_local_filename)
    print(f'Image {file_name} was downloaded to {temp_local_filename}.')

    image = cv2.imread(temp_local_filename)

    # Generate random string to be used for unique identity for images
    random_string = generate_random_string(8)

    # Scaling operations
    scale_image(image, file_name, temp_local_filename, random_string, preprocessing_metrics)

    # Bluring Operations
    blur_image(image, file_name, temp_local_filename, random_string, preprocessing_metrics)

    # Brightness Adjustment Operations
    adjust_brightness(image, file_name, temp_local_filename, random_string, preprocessing_metrics)

    # Noise Adjustment Operations
    adjust_salt_and_pepper_noise(image, file_name, temp_local_filename, random_string, preprocessing_metrics)

    os.remove(temp_local_filename)

    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['total_preprocessing_time'] = processing_time

    db_id = f"{file_name.split('.')[0]}_{random_string}"
    preprocessing_metrics['id'] = db_id
    doc_ref = firestore_client.collection('preprocessing_metrics').document(db_id)
    doc_ref.set(preprocessing_metrics)

