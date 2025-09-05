import json
import boto3
import os
from typing import List, Dict, Any
from datetime import datetime
import hashlib
from rag_engine import DocumentProcessor
from langchain_aws import BedrockEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection

class DataIngestionPipeline:
    """Advanced data ingestion pipeline with automatic parsing and chunking"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bedrock_client = boto3.client('bedrock-runtime')
        
        # Initialize embeddings (Titan)
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v1"
        )
        
        # Initialize OpenSearch
        opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
        self.opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_endpoint.replace('https://', ''), 'port': 443}],
            http_auth=('admin', 'admin'),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(self.embeddings, self.opensearch_client)
    
    def ingest_slack_message(self, message_data: Dict[str, Any]):
        """Ingest and process Slack message for future retrieval"""
        try:
            content = message_data.get('text', '')
            channel_id = message_data.get('channel', '')
            user_id = message_data.get('user', '')
            timestamp = message_data.get('ts', str(time.time()))
            
            if not content or len(content.strip()) < 10:
                return False
            
            # Create metadata
            metadata = {
                'source': 'slack',
                'channel_id': channel_id,
                'user_id': user_id,
                'timestamp': timestamp,
                'message_type': 'user_message',
                'doc_id': hashlib.md5(f"{channel_id}_{timestamp}_{content}".encode()).hexdigest()
            }
            
            # Process and store
            self.doc_processor.process_and_store_document(
                content=content,
                metadata=metadata,
                channel_id=channel_id
            )
            
            return True
            
        except Exception as e:
            print(f"Error ingesting Slack message: {str(e)}")
            return False
    
    def ingest_file_from_s3(self, bucket: str, key: str, channel_id: str):
        """Ingest and process file from S3"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            # Advanced file parsing based on extension
            file_extension = key.split('.')[-1].lower()
            parsed_content = self._parse_file_content(content, file_extension)
            
            metadata = {
                'source': 's3',
                'bucket': bucket,
                'key': key,
                'file_type': file_extension,
                'doc_id': hashlib.md5(f"{bucket}_{key}".encode()).hexdigest(),
                'ingestion_timestamp': datetime.now().isoformat()
            }
            
            self.doc_processor.process_and_store_document(
                content=parsed_content,
                metadata=metadata,
                channel_id=channel_id
            )
            
            return True
            
        except Exception as e:
            print(f"Error ingesting file from S3: {str(e)}")
            return False
    
    def _parse_file_content(self, content: str, file_type: str) -> str:
        """Advanced parsing based on file type"""
        if file_type in ['json']:
            try:
                data = json.loads(content)
                return self._extract_text_from_json(data)
            except:
                return content
        
        elif file_type in ['csv']:
            return self._parse_csv_content(content)
        
        elif file_type in ['md', 'markdown']:
            return self._parse_markdown_content(content)
        
        else:
            return content
    
    def _extract_text_from_json(self, data: Any) -> str:
        """Extract meaningful text from JSON data"""
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 5:
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    text_parts.append(self._extract_text_from_json(value))
            return "\n".join(text_parts)
        
        elif isinstance(data, list):
            return "\n".join([self._extract_text_from_json(item) for item in data])
        
        else:
            return str(data)
    
    def _parse_csv_content(self, content: str) -> str:
        """Parse CSV content into readable format"""
        lines = content.split('\n')
        if len(lines) < 2:
            return content
        
        header = lines[0]
        rows = lines[1:10]  # Sample first 10 rows
        
        return f"CSV Data:\nHeaders: {header}\nSample rows:\n" + "\n".join(rows)
    
    def _parse_markdown_content(self, content: str) -> str:
        """Parse markdown content, preserving structure"""
        import re
        
        # Remove markdown syntax while preserving content
        content = re.sub(r'[#*`_]', '', content)
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        return content

def lambda_handler_ingestion(event, context):
    """Lambda handler for data ingestion triggered by S3 events"""
    try:
        pipeline = DataIngestionPipeline()
        
        results = []
        for record in event.get('Records', []):
            if record.get('eventSource') == 'aws:s3':
                bucket = record['s3']['bucket']['name']
                key = record['s3']['object']['key']
                
                # Extract channel_id from key path or use default
                channel_id = key.split('/')[0] if '/' in key else 'general'
                
                success = pipeline.ingest_file_from_s3(bucket, key, channel_id)
                results.append({
                    'bucket': bucket,
                    'key': key,
                    'success': success
                })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data ingestion completed',
                'results': results
            })
        }
        
    except Exception as e:
        print(f"Error in data ingestion: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Data ingestion failed',
                'details': str(e)
            })
        }