variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "slack-bedrock-integration"
}

variable "api_gateway_name" {
  description = "Name of the API Gateway"
  type        = string
  default     = "slack-bedrock-api"
}

variable "bedrock_agent_name" {
  description = "Name of the Bedrock agent"
  type        = string
  default     = "slack-assistant"
}

variable "bedrock_agent_description" {
  description = "Description of the Bedrock agent"
  type        = string
  default     = "AI assistant for Slack integration"
}

variable "bedrock_agent_instruction" {
  description = "Instructions for the Bedrock agent"
  type        = string
  default     = "You are a helpful AI assistant that responds to questions from Slack users. Be concise, friendly, and professional in your responses."
}

variable "bedrock_foundation_model" {
  description = "Foundation model for the Bedrock agent"
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "enable_knowledge_base" {
  description = "Enable knowledge base for the agent"
  type        = bool
  default     = false
}

variable "s3_bucket_name" {
  description = "S3 bucket name for knowledge base documents"
  type        = string
  default     = null
}

variable "create_s3_bucket" {
  description = "Create S3 bucket for knowledge base"
  type        = bool
  default     = false
}

variable "enable_prompt_override" {
  description = "Enable custom prompt templates"
  type        = bool
  default     = false
}

variable "temperature" {
  description = "Temperature for model inference"
  type        = number
  default     = 0.7
}

variable "top_p" {
  description = "Top-p for model inference"
  type        = number
  default     = 0.9
}

variable "max_tokens" {
  description = "Maximum tokens for model response"
  type        = number
  default     = 2048
}

variable "slack_signing_secret_arn" {
  description = "AWS Secrets Manager ARN for Slack app signing secret"
  type        = string
  sensitive   = true
}

variable "slack_bot_token_arn" {
  description = "AWS Secrets Manager ARN for Slack bot token"
  type        = string
  sensitive   = true
}

variable "slack_bot_user_id" {
  description = "Slack bot user ID for filtering bot mentions and messages"
  type        = string
  default     = null
}

# Guardrails Configuration
variable "enable_guardrails" {
  description = "Enable Bedrock Guardrails integration"
  type        = bool
  default     = false
}

variable "guardrail_name" {
  description = "Name of the Bedrock guardrail"
  type        = string
  default     = "bedrock-guardrail"
}

variable "guardrail_description" {
  description = "Description of the Bedrock guardrail"
  type        = string
  default     = "Guardrail for Bedrock agent safety"
}

variable "blocked_input_messaging" {
  description = "Message to return when input is blocked by guardrails"
  type        = string
  default     = "Sorry, I can't process that request as it violates our content policy."
}

variable "blocked_outputs_messaging" {
  description = "Message to return when output is blocked by guardrails"
  type        = string
  default     = "I apologize, but I can't provide that information as it goes against our content guidelines."
}

variable "create_guardrail_version" {
  description = "Create a version of the guardrail"
  type        = bool
  default     = true
}

variable "guardrail_version_description" {
  description = "Description for the guardrail version"
  type        = string
  default     = "Initial version of the guardrail"
}


variable "content_policy_config" {
  description = "Content policy configuration for guardrails"
  type = object({
    filters_config = list(object({
      input_strength  = string
      output_strength = string
      type           = string
    }))
  })
  default = null
}

variable "sensitive_information_policy_config" {
  description = "Sensitive information policy configuration"
  type = object({
    pii_entities_config = list(object({
      action = string
      type   = string
    }))
    regexes_config = list(object({
      action      = string
      description = string
      name        = string
      pattern     = string
    }))
  })
  default = null
}

variable "topic_policy_config" {
  description = "Topic policy configuration for guardrails"
  type = object({
    topics_config = list(object({
      definition = string
      name       = string
      type       = string
      examples   = list(string)
    }))
  })
  default = null
}

variable "word_policy_config" {
  description = "Word policy configuration for guardrails"
  type = object({
    managed_word_lists_config = list(object({
      type = string
    }))
    words_config = list(object({
      text = string
    }))
  })
  default = null
}

variable "contextual_grounding_policy_config" {
  description = "Contextual grounding policy configuration"
  type = object({
    filters_config = list(object({
      threshold = number
      type      = string
    }))
  })
  default = null
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# New variables for RAG chatbot
variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for short-term storage"
  type        = string
  default     = "slack-chatbot-context"
}

variable "opensearch_domain_name" {
  description = "Name of the OpenSearch domain for vector database"
  type        = string
  default     = "slack-chatbot-vectors"
}

variable "opensearch_instance_type" {
  description = "Instance type for OpenSearch"
  type        = string
  default     = "t3.small.search"
}

variable "opensearch_instance_count" {
  description = "Number of instances in OpenSearch cluster"
  type        = number
  default     = 1
}