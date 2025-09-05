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
    """Send a message to Slack using the Web API"""
    http = urllib3.PoolManager()
    
    slack_url = "https://slack.com/api/chat.postMessage"
    
    payload = {
        "channel": channel,
        "text": text,
        "as_user": False,
        "username": "AI Assistant"
    }
    
    # Add threading support if thread_ts is provided
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
            
        print(f"Message sent successfully to channel {channel}")
        return True
        
    except Exception as e:
        print(f"Error sending message to Slack: {str(e)}")
        return False

def clean_bot_mention(text, bot_user_id=None):
    """Remove bot mentions from message text"""
    import re
    
    # Remove bot mentions like <@U06D5B8AR8R>
    cleaned_text = re.sub(r'<@[UW][A-Z0-9]+>', '', text)
    
    # Also remove any specific bot user ID if provided
    if bot_user_id:
        cleaned_text = re.sub(f'<@{bot_user_id}>', '', cleaned_text)
    
    # Clean up extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text.strip()

def format_agent_response(response_text):
    """Format the agent response for Slack display"""
    # Limit response length to avoid Slack message limits
    max_length = 3000
    if len(response_text) > max_length:
        response_text = response_text[:max_length] + "...\n\n_Response truncated due to length._"
    
    # Add some formatting for better readability
    if response_text.strip():
        return f"🤖 *AI Assistant Response:*\n{response_text}"
    else:
        return "🤖 I'm processing your request, but didn't generate a response. Please try rephrasing your question."

def lambda_handler(event, context):

    try:
        secrets_client = boto3.client('secretsmanager')
        agent_id = os.environ.get('BEDROCK_AGENT_ID')
        agent_alias = os.environ.get('BEDROCK_AGENT_ALIAS', 'TSTALIASID')
        signing_secret_arn = os.environ.get('SLACK_SIGNING_SECRET_ARN')
        bot_token_arn = os.environ.get('SLACK_BOT_TOKEN_ARN')
        bot_user_id = os.environ.get('SLACK_BOT_USER_ID')
        guardrail_id = os.environ.get('GUARDRAIL_ID')
        guardrail_version = os.environ.get('GUARDRAIL_VERSION')
        
        # Retrieve secrets
        signing_secret = secrets_client.get_secret_value(SecretId=signing_secret_arn)['SecretString']
        bot_token = secrets_client.get_secret_value(SecretId=bot_token_arn)['SecretString']
        
        if not agent_id or not signing_secret or not bot_token:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Missing required environment variables'})
            }

        headers = event.get('headers', {})
        body = event.get('body', '')

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

        # Parse the payload - handle both JSON and form-encoded formats
        try:
            content_type = headers.get('content-type', '').lower()
            
            if content_type.startswith('application/x-www-form-urlencoded'):
                # Slack interactive components and some webhooks use form encoding
                parsed_body = parse_qs(body)
                payload = json.loads(parsed_body.get('payload', ['{}'])[0])
            elif content_type.startswith('application/json') or not content_type:
                # Direct JSON payload (most common for Events API)
                payload = json.loads(body) if body else {}
            else:
                # Try to parse as JSON first, fallback to empty dict
                try:
                    payload = json.loads(body)
                except:
                    payload = {}
                    
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing payload: {str(e)}")
            payload = {}

        print(f"Parsed payload: {json.dumps(payload, indent=2)}")

        if payload.get('type') == 'url_verification':
            return {
                'statusCode': 200,
                'body': payload.get('challenge', '')
            }

        event_data = payload.get('event', {})
        message_text = event_data.get('text', '')
        user_id = event_data.get('user', '')
        channel_id = event_data.get('channel', '')
        thread_ts = event_data.get('thread_ts')  # For threaded responses
        original_ts = event_data.get('ts')  # Message timestamp
        message_type = event_data.get('type', '')
        
        # Only process 'message' type events
        if message_type != 'message':
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignored event type: {message_type}'})
            }
        
        # Ignore bot messages, messages from our own bot, and certain subtypes
        if (event_data.get('bot_id') or 
            event_data.get('subtype') == 'bot_message' or 
            event_data.get('subtype') == 'message_changed' or
            event_data.get('subtype') == 'message_deleted' or
            (bot_user_id and user_id == bot_user_id) or
            not user_id):
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Bot message or system message ignored'})
            }
        
        # Clean bot mentions from the message text
        cleaned_message_text = clean_bot_mention(message_text, bot_user_id)
        
        print(f"Original text: '{message_text}'")
        print(f"Cleaned text: '{cleaned_message_text}'")
        
        # Validate we have content to process after cleaning
        if not cleaned_message_text or not channel_id:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No meaningful text or channel to process after cleaning'})
            }
        
        # Generate session ID for conversation continuity
        session_id = f"slack-{user_id}-{channel_id}"

        print(f"Processing message from user {user_id} in channel {channel_id}: '{cleaned_message_text}'")

        # Process through advanced RAG engine
        rag = get_rag_engine()
        agent_response = rag.process_message(
            user_query=cleaned_message_text,
            channel_id=channel_id,
            user_id=user_id
        )

        print(f"User {user_id} in channel {channel_id}: {message_text}")
        print(f"Bedrock agent response: {agent_response}")

        # Format and send response back to Slack
        formatted_response = format_agent_response(agent_response)
        
        # Determine if this should be a threaded response
        # Use thread_ts if the original message was in a thread, otherwise use original_ts to start a new thread
        response_thread_ts = thread_ts if thread_ts else original_ts
        
        # Send the response back to Slack
        slack_success = send_slack_message(
            channel=channel_id,
            text=formatted_response,
            bot_token=bot_token,
            thread_ts=response_thread_ts
        )
        
        if not slack_success:
            print("Failed to send response to Slack, but processing was successful")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'message': 'Message processed and sent to Slack successfully',
                'slack_success': slack_success,
                'agent_response_length': len(agent_response)
            })
        }
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e)
            })
        }