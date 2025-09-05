import json
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to archive DynamoDB items older than 30 days to S3
    """
    try:
        table_name = os.environ['DYNAMODB_TABLE_NAME']
        bucket_name = os.environ['S3_BUCKET_NAME']
        
        # Calculate cutoff timestamp (30 days ago)
        cutoff_date = datetime.now() - timedelta(days=30)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        table = dynamodb.Table(table_name)
        
        # Scan for items older than 30 days
        items_to_archive = []
        
        # Scan the table for old items
        response = table.scan(
            FilterExpression='#ts < :cutoff',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={':cutoff': cutoff_timestamp}
        )
        
        items_to_archive.extend(response['Items'])
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='#ts < :cutoff',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':cutoff': cutoff_timestamp},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items_to_archive.extend(response['Items'])
        
        if not items_to_archive:
            logger.info("No items found for archiving")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No items to archive',
                    'archived_count': 0
                })
            }
        
        # Archive items to S3
        archived_count = 0
        batch_size = 100
        
        for i in range(0, len(items_to_archive), batch_size):
            batch = items_to_archive[i:i + batch_size]
            
            # Create S3 key with date partition
            archive_date = datetime.now().strftime('%Y/%m/%d')
            s3_key = f"archived-data/{archive_date}/batch_{i//batch_size}_{int(datetime.now().timestamp())}.json"
            
            # Convert batch to JSON and upload to S3
            s3_data = {
                'archived_at': datetime.now().isoformat(),
                'items': batch
            }
            
            s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json.dumps(s3_data, default=str),
                ContentType='application/json'
            )
            
            archived_count += len(batch)
            logger.info(f"Archived {len(batch)} items to s3://{bucket_name}/{s3_key}")
        
        logger.info(f"Successfully archived {archived_count} items to S3")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully archived {archived_count} items to S3',
                'archived_count': archived_count,
                's3_bucket': bucket_name
            })
        }
        
    except Exception as e:
        logger.error(f"Error archiving data: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Failed to archive data: {str(e)}'
            })
        }

def clean_item_for_json(item):
    """Clean DynamoDB item for JSON serialization"""
    if isinstance(item, dict):
        return {k: clean_item_for_json(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [clean_item_for_json(i) for i in item]
    else:
        return item