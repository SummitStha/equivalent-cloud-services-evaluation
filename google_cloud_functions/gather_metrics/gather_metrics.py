import json
import os
from google.cloud import storage, firestore

storage_client = storage.Client()
metrics_bucket_name = os.getenv('METRICS_BUCKET_NAME')
metrics_bucket = storage_client.bucket(metrics_bucket_name)

operations = [
        "noise_3",
        "brightness_1.5",
        "blurred_13",
        "scaled_10",
        "blurred_9",
        "scaled_70",
        "noise_5",
        "noise_9",
        "noise_7",
        "scaled_30",
        "noise_1",
        "scaled_90",
        "brightness_3.0",
        "blurred_5",
        "scaled_50",
        "blurred_1",
        "brightness_0.25",
        "blurred_17"
    ]


def gather_metrics(request):
    final_metrics = {}

    firestore_client = firestore.Client()

    collection_ref = firestore_client.collection('cloud_vision_results')

    preprocessing_metrics_collection_ref = firestore_client.collection('preprocessing_metrics')

     # image_ids = ['Image_Success_bo4IiJBg', 'Image_Success_iwtmyh4C', 'Test_1_3eVnnoiz', 'Test_1_Bq42WUNC', 'Test_1_EMZeVi6B', 'Test_1_KD0HHLe3']

    # # Define the query
    # for i in image_ids:
    #     query = collection_ref.where('image_id', '==', i)

    #     # Get the documents that match the query
    #     docs = query.stream()

    #     # Delete the documents
    #     for doc in docs:
    #         doc.reference.delete()
    
    # return "Done", 200

    # Get the total count of documents in the collection
    total_count = len(collection_ref.get())
    original_image_count = total_count / 18
    print(f'Total count: {total_count}')
    final_metrics['total_count'] = total_count
    final_metrics['original_image_count'] = original_image_count

    query = collection_ref.where('detected', '==', False)

     # Get the total count of documents that match the query
    undetected_count = len(query.get())
    print(f'Undetected count: {undetected_count}')
    final_metrics['undetected_count'] = undetected_count

    detected_count = total_count - undetected_count
    final_metrics['detected_count'] = detected_count
    accuracy = (detected_count/total_count) * 100
    print(f'Accuracy: {accuracy}')
    final_metrics['accuracy'] = accuracy

    # Loop through all the documents in the collection
    for doc in query.stream():
        # Print out the document ID and its data
        print(f'Document ID: {doc.id}')
        print(f'Data: {doc.to_dict()}')
        doc_dict = doc.to_dict()
        main_image_name, performed_operation = doc_dict['id'].split('/')
        op_undetected_count_key = f"{'.'.join(performed_operation.split('.')[:-1])}_undetected_count"
        if final_metrics.get(main_image_name):
            final_metrics[main_image_name].append(performed_operation)
            if final_metrics.get(op_undetected_count_key):
                final_metrics[op_undetected_count_key] += 1
            else:
                final_metrics[op_undetected_count_key] = 1
        else:
            final_metrics[main_image_name] = [performed_operation]
            if final_metrics.get(op_undetected_count_key):
                final_metrics[op_undetected_count_key] += 1
            else:
                final_metrics[op_undetected_count_key] = 1
    
    for operation in operations:
        final_metrics[f'{operation}_detected_count'] = original_image_count - final_metrics[f'{operation}_undetected_count']
        final_metrics[f'{operation}_accuracy'] = (final_metrics[f'{operation}_detected_count'] / original_image_count) * 100
    
    preprocessing_metrics_dict = {'results': []}
    for doc in preprocessing_metrics_collection_ref.get():
        preprocessing_metrics_dict['results'].append(doc.to_dict())

    # Save metrics to a file
    metrics_data = json.dumps(final_metrics, indent=4)
    metrics_filename = "metrics.json"

    preprocessing_metrics_data = json.dumps(preprocessing_metrics_dict, indent=4)
    preprocessing_metrics_filename = "preprocessing_metrics.json"

    # Upload the metrics file to the Cloud Storage bucket
    metrics_blob = metrics_bucket.blob(metrics_filename)
    metrics_blob.upload_from_string(metrics_data, content_type="application/json")

    preprocessing_metrics_blob = metrics_bucket.blob(preprocessing_metrics_filename)
    preprocessing_metrics_blob.upload_from_string(preprocessing_metrics_data, content_type="application/json")

    return f"Metrics saved to gs://{metrics_bucket_name}/{metrics_filename}", 200

