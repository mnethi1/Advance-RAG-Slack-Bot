import json
import os
import time
import boto3
import hashlib
import hmac
import base64
import urllib3
from urllib.parse import parse_qs
from rag_engine import AdvancedRAGEngine, create_opensearch_index

# Initialize RAG engine globally for reuse across invocations
rag_engine = None

def get_rag_engine():
    """Lazy initialization of RAG engine"""
    global rag_engine
    if rag_engine is None:
        rag_engine = AdvancedRAGEngine()
        # Ensure OpenSearch index exists
        create_opensearch_index()
    return rag_engine

def verify_slack_request(body, timestamp, signature, signing_secret):
    """Verify Slack request signature for security"""
    if abs(int(timestamp) - int(time.time())) > 60 * 5:
        return False
    
    sig_basestring = f'v0:{timestamp}:{body}'
    my_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(my_signature, signature)

def send_slack_message(channel, text, bot_token, thread_ts=None):
    """Send enhanced message to Slack with formatting"""
    http = urllib3.PoolManager()
    
    slack_url = "https://slack.com/api/chat.postMessage"
    
    # Enhanced formatting for better presentation
    formatted_text = f"🤖 *AI Assistant*\n\n{text}"
    
    payload = {
        "channel": channel,
        "text": formatted_text,
        "as_user": False,
        "username": "Advanced AI Assistant",
        "parse": "mrkdwn"  # Enable markdown parsing
    }
    
    if thread_ts:
        payload["thread_ts"] = thread_ts
    
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = http.request(
            'POST',
            slack_url,
            body=json.dumps(payload),
            headers=headers
        )
        
        result = json.loads(response.data.decode('utf-8'))
        
        if not result.get('ok'):
            print(f"Slack API error: {result.get('error', 'Unknown error')}")
            return False
            
        print(f"Enhanced message sent successfully to channel {channel}")
        return True
        
    except Exception as e:
        print(f"Error sending message to Slack: {str(e)}")
        return False

def clean_bot_mention(text, bot_user_id=None):
    """Enhanced bot mention cleaning with context preservation"""
    import re
    
    # Remove bot mentions while preserving context
    cleaned_text = re.sub(r'<@[UW][A-Z0-9]+>', '', text)
    
    if bot_user_id:
        cleaned_text = re.sub(f'<@{bot_user_id}>', '', cleaned_text)
    
    # Clean up extra whitespace but preserve paragraph structure
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text.strip()

def extract_channel_context(event_data):
    """Extract enhanced channel context for privacy and awareness"""
    return {
        "channel_id": event_data.get('channel', ''),
        "channel_type": event_data.get('channel_type', 'unknown'),
        "user_id": event_data.get('user', ''),
        "timestamp": event_data.get('ts', ''),
        "thread_ts": event_data.get('thread_ts'),
        "message_type": event_data.get('type', ''),
        "subtype": event_data.get('subtype')
    }

def lambda_handler(event, context):
    """Enhanced Lambda handler with advanced RAG capabilities"""
    try:
        # Initialize services
        secrets_client = boto3.client('secretsmanager')
        
        # Get environment variables
        agent_id = os.environ.get('BEDROCK_AGENT_ID')
        agent_alias = os.environ.get('BEDROCK_AGENT_ALIAS', 'TSTALIASID')
        signing_secret_arn = os.environ.get('SLACK_SIGNING_SECRET_ARN')
        bot_token_arn = os.environ.get('SLACK_BOT_TOKEN_ARN')
        bot_user_id = os.environ.get('SLACK_BOT_USER_ID')
        
        # Retrieve secrets
        signing_secret = secrets_client.get_secret_value(SecretId=signing_secret_arn)['SecretString']
        bot_token = secrets_client.get_secret_value(SecretId=bot_token_arn)['SecretString']
        
        if not all([agent_id, signing_secret, bot_token]):
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Missing required configuration'})
            }

        headers = event.get('headers', {})
        body = event.get('body', '')

        # Verify Slack signature
        slack_signature = headers.get('x-slack-signature', '')
        slack_timestamp = headers.get('x-slack-request-timestamp', '')
        
        if slack_signature and slack_timestamp:
            if not verify_slack_request(body, slack_timestamp, slack_signature, signing_secret):
                return {
                    'statusCode': 401,
                    'body': json.dumps({'error': 'Invalid signature'})
                }

        if event.get('isBase64Encoded', False):
            body = base64.b64decode(body).decode('utf-8')

        # Enhanced payload parsing
        try:
            content_type = headers.get('content-type', '').lower()
            
            if content_type.startswith('application/x-www-form-urlencoded'):
                parsed_body = parse_qs(body)
                payload = json.loads(parsed_body.get('payload', ['{}'])[0])
            elif content_type.startswith('application/json') or not content_type:
                payload = json.loads(body) if body else {}
            else:
                try:
                    payload = json.loads(body)
                except:
                    payload = {}
                    
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing payload: {str(e)}")
            payload = {}

        print(f"Enhanced payload processing: {json.dumps(payload, indent=2)}")

        # Handle URL verification
        if payload.get('type') == 'url_verification':
            return {
                'statusCode': 200,
                'body': payload.get('challenge', '')
            }

        event_data = payload.get('event', {})
        message_text = event_data.get('text', '')
        
        # Extract enhanced channel context
        channel_context = extract_channel_context(event_data)
        
        # Enhanced message filtering
        if (event_data.get('bot_id') or 
            event_data.get('subtype') in ['bot_message', 'message_changed', 'message_deleted'] or
            (bot_user_id and channel_context['user_id'] == bot_user_id) or
            not channel_context['user_id'] or
            channel_context['message_type'] != 'message'):
            
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Message filtered out'})
            }
        
        # Clean and prepare message
        cleaned_message_text = clean_bot_mention(message_text, bot_user_id)
        
        if not cleaned_message_text or not channel_context['channel_id']:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No content to process'})
            }

        print(f"Processing enhanced message from user {channel_context['user_id']} in channel {channel_context['channel_id']}: '{cleaned_message_text}'")

        # Process through advanced RAG engine
        rag = get_rag_engine()
        ai_response = rag.process_message(
            user_query=cleaned_message_text,
            channel_id=channel_context['channel_id'],
            user_id=channel_context['user_id']
        )

        print(f"Enhanced RAG response: {ai_response}")

        # Determine threading behavior
        response_thread_ts = channel_context['thread_ts'] if channel_context['thread_ts'] else channel_context['timestamp']
        
        # Send enhanced response to Slack
        slack_success = send_slack_message(
            channel=channel_context['channel_id'],
            text=ai_response,
            bot_token=bot_token,
            thread_ts=response_thread_ts
        )
        
        if not slack_success:
            print("Failed to send response to Slack")

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'message': 'Enhanced message processed successfully',
                'slack_success': slack_success,
                'response_length': len(ai_response),
                'processing_method': 'advanced_rag'
            })
        }
        
    except Exception as e:
        print(f"Error in enhanced processing: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Enhanced processing error',
                'details': str(e)
            })
        }