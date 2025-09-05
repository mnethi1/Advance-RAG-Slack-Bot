terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# DynamoDB for short-term storage (30 days)
module "dynamodb" {
  source = "./modules/dynamodb"
  
  table_name = var.dynamodb_table_name
  ttl_attribute = "expires_at"
}

# S3 bucket for long-term storage and vector embeddings
module "s3_storage" {
  source = "./modules/s3"
  
  bucket_name = var.s3_bucket_name
  versioning_enabled = true
}

# Vector database (OpenSearch)
module "opensearch" {
  source = "./modules/opensearch"
  
  domain_name = var.opensearch_domain_name
  instance_type = var.opensearch_instance_type
  instance_count = var.opensearch_instance_count
}

module "bedrock_agent" {
  source = "./modules/bedrock-agent"
  
  agent_name               = var.bedrock_agent_name
  agent_description        = var.bedrock_agent_description
  agent_instruction        = var.bedrock_agent_instruction
  foundation_model         = var.bedrock_foundation_model
  enable_knowledge_base    = var.enable_knowledge_base
  s3_bucket_name          = module.s3_storage.bucket_name
  create_s3_bucket        = false
  enable_prompt_override  = var.enable_prompt_override
  temperature             = var.temperature
  top_p                   = var.top_p
  max_tokens              = var.max_tokens
  
  # Guardrails integration
  guardrail_configuration = var.enable_guardrails ? {
    guardrail_identifier = module.bedrock_guardrails.guardrail_id
    guardrail_version    = var.create_guardrail_version ? module.bedrock_guardrails.guardrail_version_number : "DRAFT"
  } : null
  
  depends_on = [module.bedrock_guardrails, module.s3_storage]
}

module "iam" {
  source = "./modules/iam"
  
  lambda_function_name = var.lambda_function_name
}

module "lambda" {
  source = "./modules/lambda"
  
  function_name     = var.lambda_function_name
  lambda_role_arn   = module.iam.lambda_role_arn
  source_code_path  = "./lambda-code"
  
  bedrock_agent_id        = module.bedrock_agent.agent_id
  bedrock_agent_alias     = module.bedrock_agent.agent_alias_id
  slack_signing_secret_arn = var.slack_signing_secret_arn
  slack_bot_token_arn     = var.slack_bot_token_arn
  slack_bot_user_id       = var.slack_bot_user_id
  
  # Storage configuration
  dynamodb_table_name = module.dynamodb.table_name
  s3_bucket_name      = module.s3_storage.bucket_name
  opensearch_endpoint = module.opensearch.domain_endpoint
  
  # Guardrails configuration for Lambda
  guardrail_id      = var.enable_guardrails ? module.bedrock_guardrails.guardrail_id : null
  guardrail_version = var.enable_guardrails ? (var.create_guardrail_version ? module.bedrock_guardrails.guardrail_version_number : "DRAFT") : null
}

module "bedrock_guardrails" {
  source = "./modules/BedRockGuardRails"
  
  guardrail_name               = var.guardrail_name
  description                  = var.guardrail_description
  blocked_input_messaging      = var.blocked_input_messaging
  blocked_outputs_messaging    = var.blocked_outputs_messaging
  content_policy_config        = var.content_policy_config
  sensitive_information_policy_config = var.sensitive_information_policy_config
  topic_policy_config          = var.topic_policy_config
  word_policy_config          = var.word_policy_config
  contextual_grounding_policy_config = var.contextual_grounding_policy_config
  create_version              = var.create_guardrail_version
  version_description         = var.guardrail_version_description
  tags                        = var.tags
}

module "api_gateway" {
  source = "./modules/api-gateway"
  
  lambda_function_arn = module.lambda.lambda_function_arn
  lambda_function_name = var.lambda_function_name
  api_gateway_name = var.api_gateway_name
}