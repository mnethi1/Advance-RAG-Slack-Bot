#!/bin/bash

set -e

echo "🚀 Deploying Advanced Slack RAG Chatbot..."

# Check if required environment variables are set
if [ -z "$AWS_REGION" ]; then
    export AWS_REGION="us-east-1"
    echo "⚠️  AWS_REGION not set, defaulting to us-east-1"
fi

# Check Terraform installation
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is not installed. Please install Terraform first."
    exit 1
fi

# Check AWS CLI installation and authentication
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install AWS CLI first."
    exit 1
fi

echo "🔍 Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS credentials verified"

# Initialize Terraform
echo "📦 Initializing Terraform..."
terraform init

# Validate Terraform configuration
echo "🔍 Validating Terraform configuration..."
terraform validate

# Plan deployment
echo "📋 Planning deployment..."
terraform plan -var-file="terraform.tfvars.dev" -out=tfplan

# Ask for confirmation
read -p "🤔 Do you want to apply these changes? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

# Apply changes
echo "🚀 Applying Terraform changes..."
terraform apply tfplan

# Get outputs
echo "📊 Deployment completed! Getting outputs..."
terraform output

echo "✅ Advanced Slack RAG Chatbot deployed successfully!"
echo ""
echo "🔧 Next steps:"
echo "1. Configure your Slack app with the API Gateway endpoint"
echo "2. Add the bot to your Slack channels"
echo "3. Upload documents to S3 for knowledge base population"
echo "4. Test the bot by mentioning it in a channel"