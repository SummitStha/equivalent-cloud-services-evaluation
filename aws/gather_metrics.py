import boto3
import json
import os
from decimal import Decimal

from boto3.dynamodb.conditions import Attr
from boto3.dynamodb.conditions import Key


s3_client = boto3.client('s3')
metrics_bucket = os.getenv('METRICS_BUCKET_NAME')

dynamodb_client = boto3.resource('dynamodb')
table = dynamodb_client.Table('amazon_rekognition_results')

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


# Convert Decimal to float
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def gather_metrics(event, context):
    final_metrics = {}

    # Get the total count of documents in the collection
    total_count = table.scan(Select='COUNT')['Count']
    original_image_count = total_count / 18
    print(f'Total count: {total_count}')
    final_metrics['total_count'] = total_count
    final_metrics['original_image_count'] = original_image_count

    undetected_response = table.scan(
        FilterExpression=Attr('detected').eq(False)
    )

     # Get the total count of documents that match the query
    undetected_count = undetected_response['Count']
    print(f'Undetected count: {undetected_count}')
    final_metrics['undetected_count'] = undetected_count

    detected_count = total_count - undetected_count
    final_metrics['detected_count'] = detected_count
    accuracy = (detected_count/total_count) * 100
    print(f'Accuracy: {accuracy}')
    final_metrics['accuracy'] = accuracy

    # Loop through all the items in the result
    items = undetected_response['Items']
    for item in items:
        # Print out the ID and its data
        print(f"ID: {item['id']}")
        print(f'Item: {item}')
        main_image_name, performed_operation = item['id'].split('/')
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

    # Save metrics to a file
    metrics_data = json.dumps(final_metrics, indent=4)
    metrics_filename = "metrics.json"

    # Upload the metrics file to the Cloud Storage bucket
    s3_client.put_object(Body=metrics_data, Bucket=metrics_bucket, Key=metrics_filename)

    return {
        'statusCode': 200,
        'body': f"Metrics saved to https://{metrics_bucket}.s3.amazonaws.com/{metrics_filename}"
    }


