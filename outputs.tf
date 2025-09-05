output "api_gateway_url" {
  description = "API Gateway invoke URL"
  value       = module.api_gateway.api_gateway_url
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = module.lambda.lambda_function_arn
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = module.lambda.lambda_function_name
}

output "bedrock_agent_id" {
  description = "Bedrock agent ID"
  value       = module.bedrock_agent.agent_id
}

output "bedrock_agent_arn" {
  description = "Bedrock agent ARN"
  value       = module.bedrock_agent.agent_arn
}

output "bedrock_agent_alias_id" {
  description = "Bedrock agent alias ID"
  value       = module.bedrock_agent.agent_alias_id
}

output "knowledge_base_id" {
  description = "Knowledge base ID"
  value       = module.bedrock_agent.knowledge_base_id
}

output "guardrail_id" {
  description = "Bedrock guardrail ID"
  value       = var.enable_guardrails ? module.bedrock_guardrails.guardrail_id : null
}

output "guardrail_arn" {
  description = "Bedrock guardrail ARN"
  value       = var.enable_guardrails ? module.bedrock_guardrails.guardrail_arn : null
}

output "guardrail_version" {
  description = "Bedrock guardrail version"
  value       = var.enable_guardrails ? module.bedrock_guardrails.guardrail_version : null
}

