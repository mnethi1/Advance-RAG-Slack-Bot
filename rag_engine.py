import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import tiktoken

class RAGState(TypedDict):
    messages: List[Dict[str, Any]]
    context: List[str]
    user_query: str
    channel_id: str
    user_id: str
    retrieved_docs: List[Dict[str, Any]]
    final_response: str
    processed_query: str

class AdvancedRAGEngine:
    def __init__(self):
        self.bedrock_client = boto3.client('bedrock-runtime')
        self.dynamodb = boto3.resource('dynamodb')
        self.s3_client = boto3.client('s3')
        
        self.dynamodb_table_name = os.environ.get('DYNAMODB_TABLE_NAME')
        self.s3_bucket_name = os.environ.get('S3_BUCKET_NAME')
        self.opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
        
        # Initialize OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{'host': self.opensearch_endpoint.replace('https://', ''), 'port': 443}],
            http_auth=('admin', 'admin'),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        
        # Initialize embeddings model (Titan)
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v1"
        )
        
        # Initialize Claude Haiku
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("store_interaction", self._store_interaction)
        
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "process_query")
        workflow.add_edge("process_query", "generate_response")
        workflow.add_edge("generate_response", "store_interaction")
        workflow.add_edge("store_interaction", END)
        
        return workflow.compile()
    
    def _retrieve_context(self, state: RAGState) -> RAGState:
        """Retrieve relevant context from vector database and conversation history with enhanced semantic search"""
        try:
            # Get channel-specific conversation history from DynamoDB
            table = self.dynamodb.Table(self.dynamodb_table_name)
            channel_history = self._get_channel_history(state["channel_id"])
            
            # Enhanced query preprocessing
            processed_query = self._preprocess_query(state["user_query"])
            
            # Generate embedding for processed query
            query_embedding = self.embeddings.embed_query(processed_query)
            
            # Hybrid search: Combine semantic vector search with keyword matching
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Semantic vector search (primary)
                            {
                                "knn": {
                                    "content_vector": {
                                        "vector": query_embedding,
                                        "k": 8,
                                        "boost": 2.0
                                    }
                                }
                            },
                            # Keyword search (supplementary)
                            {
                                "multi_match": {
                                    "query": processed_query,
                                    "fields": ["content^1.5", "metadata.query"],
                                    "type": "best_fields",
                                    "boost": 1.0,
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "_source": ["content", "metadata", "channel_id", "timestamp"],
                "size": 10
            }
            
            # Filter by channel for privacy
            if state["channel_id"]:
                search_body["query"]["bool"]["filter"] = [
                    {"term": {"channel_id": state["channel_id"]}}
                ]
            
            # Add recency boost for more recent documents
            search_body["query"]["bool"]["should"].append({
                "function_score": {
                    "query": {"match_all": {}},
                    "functions": [{
                        "exp": {
                            "timestamp": {
                                "origin": "now",
                                "scale": "7d",
                                "decay": 0.5
                            }
                        }
                    }],
                    "boost": 0.5
                }
            })
            
            response = self.opensearch_client.search(
                index="chat_vectors",
                body=search_body
            )
            
            # Process and rank results with enhanced scoring
            retrieved_docs = []
            for hit in response["hits"]["hits"]:
                # Calculate relevance score combining semantic and keyword matching
                base_score = hit["_score"]
                content = hit["_source"]["content"]
                
                # Add semantic relevance boost
                semantic_boost = self._calculate_semantic_relevance(content, processed_query)
                final_score = base_score * (1 + semantic_boost)
                
                retrieved_docs.append({
                    "content": content,
                    "score": final_score,
                    "base_score": base_score,
                    "semantic_score": semantic_boost,
                    "metadata": hit["_source"].get("metadata", {}),
                    "timestamp": hit["_source"].get("timestamp", 0)
                })
            
            # Sort by final score and take top 5
            retrieved_docs.sort(key=lambda x: x["score"], reverse=True)
            retrieved_docs = retrieved_docs[:5]
            
            # Combine channel history and retrieved docs for context
            context = []
            for msg in channel_history[-3:]:  # Last 3 messages for context
                context.append(f"Previous: {msg}")
            
            for doc in retrieved_docs:
                context.append(f"Reference (score: {doc['score']:.2f}): {doc['content']}")
            
            state["context"] = context
            state["retrieved_docs"] = retrieved_docs
            state["processed_query"] = processed_query
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            state["context"] = []
            state["retrieved_docs"] = []
            state["processed_query"] = state["user_query"]
            
        return state
    
    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing for better semantic understanding"""
        import re
        
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Expand common abbreviations and acronyms
        abbreviations = {
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "when's": "when is",
            "who's": "who is",
            "why's": "why is",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        
        query_lower = query.lower()
        for abbrev, full_form in abbreviations.items():
            query_lower = query_lower.replace(abbrev, full_form)
        
        # Preserve original casing for proper nouns while using expanded contractions
        words = query.split()
        expanded_words = query_lower.split()
        
        result_words = []
        for i, word in enumerate(words):
            if i < len(expanded_words):
                # Keep original casing for words that start with uppercase (likely proper nouns)
                if word[0].isupper() and expanded_words[i] != word.lower():
                    result_words.append(word)
                else:
                    result_words.append(expanded_words[i])
            else:
                result_words.append(word)
        
        processed_query = ' '.join(result_words)
        
        # Add contextual keywords for domain-specific queries
        domain_keywords = {
            "error": ["problem", "issue", "bug", "failure"],
            "setup": ["configuration", "install", "deploy", "configure"],
            "performance": ["speed", "slow", "optimization", "efficiency"],
            "security": ["authentication", "authorization", "permissions", "access"]
        }
        
        query_words = processed_query.lower().split()
        for domain, keywords in domain_keywords.items():
            if domain in query_words:
                # Add semantic context without changing the original query structure
                processed_query += f" {' '.join(keywords)}"
                break
        
        return processed_query
    
    def _calculate_semantic_relevance(self, content: str, query: str) -> float:
        """Calculate additional semantic relevance score between content and query"""
        try:
            # Simple keyword overlap scoring
            content_words = set(content.lower().split())
            query_words = set(query.lower().split())
            
            if not query_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = content_words & query_words
            union = content_words | query_words
            
            if not union:
                return 0.0
            
            jaccard_score = len(intersection) / len(union)
            
            # Boost score for exact phrase matches
            phrase_boost = 0.0
            if len(query.split()) > 1 and query.lower() in content.lower():
                phrase_boost = 0.3
            
            # Boost for key term matches (longer words are more important)
            key_term_boost = 0.0
            for word in query_words:
                if len(word) > 4 and word in content_words:
                    key_term_boost += 0.1
            
            final_score = min(jaccard_score + phrase_boost + key_term_boost, 1.0)
            return final_score
            
        except Exception as e:
            print(f"Error calculating semantic relevance: {str(e)}")
            return 0.0
    
    def _process_query(self, state: RAGState) -> RAGState:
        """Process and analyze the user query for intent and context"""
        try:
            # Extract key information from query
            query_analysis_prompt = f"""
            Analyze this user query for intent and extract key topics:
            Query: {state["user_query"]}
            
            Provide a brief analysis of:
            1. Main intent
            2. Key topics/entities
            3. Context requirements
            
            Keep response under 100 words.
            """
            
            analysis = self.llm.invoke(query_analysis_prompt)
            state["query_analysis"] = analysis.content
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            state["query_analysis"] = "General query"
            
        return state
    
    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using RAG with enhanced context awareness and semantic understanding"""
        try:
            context_str = "\n".join(state["context"]) if state["context"] else "No previous context available."
            processed_query = state.get("processed_query", state["user_query"])
            
            # Analyze the quality and relevance of retrieved context
            context_quality = self._assess_context_quality(state["retrieved_docs"], processed_query)
            
            rag_prompt = f"""
            You are an advanced AI assistant with semantic search capabilities integrated with Slack. Use the provided context to give accurate, helpful responses.
            
            Original Query: {state["user_query"]}
            Processed Query: {processed_query}
            Context Quality Score: {context_quality:.2f}/1.0
            
            Context from conversation history and semantic search results:
            {context_str}
            
            Instructions:
            - Use the semantically retrieved context to provide accurate, relevant responses
            - The context includes relevance scores - prioritize higher-scored references
            - If context quality is low (< 0.3), acknowledge limited relevant information and ask for clarification
            - If context quality is moderate (0.3-0.7), provide what you can and suggest areas that need more detail
            - If context quality is high (> 0.7), provide comprehensive answers based on the retrieved information
            - Maintain conversation continuity using the chat history
            - Be concise but comprehensive
            - If discussing sensitive topics, be cautious and professional
            - Format response appropriately for Slack (use markdown if helpful)
            - Reference specific sources when available with their relevance scores
            
            Response:
            """
            
            response = self.llm.invoke(rag_prompt)
            state["final_response"] = response.content
            
            # Add context quality metadata to the response for debugging
            if context_quality < 0.3:
                debug_note = f"\n\n_Debug: Low context relevance ({context_quality:.2f}). Consider rephrasing your question for better results._"
                state["final_response"] += debug_note
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            state["final_response"] = "I apologize, but I encountered an error processing your request. Please try again."
            
        return state
    
    def _assess_context_quality(self, retrieved_docs: List[Dict[str, Any]], query: str) -> float:
        """Assess the quality and relevance of retrieved context"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # Calculate average relevance score
            total_score = sum(doc.get("score", 0.0) for doc in retrieved_docs)
            avg_score = total_score / len(retrieved_docs)
            
            # Normalize score to 0-1 range (assuming max score is around 5.0 for hybrid search)
            normalized_score = min(avg_score / 5.0, 1.0)
            
            # Boost score if we have multiple relevant documents
            diversity_boost = min(len(retrieved_docs) / 5.0, 0.2)
            
            # Penalize if all scores are very low
            min_score = min(doc.get("score", 0.0) for doc in retrieved_docs)
            if min_score < 0.1:
                penalty = 0.1
            else:
                penalty = 0.0
            
            final_quality = max(0.0, normalized_score + diversity_boost - penalty)
            return min(final_quality, 1.0)
            
        except Exception as e:
            print(f"Error assessing context quality: {str(e)}")
            return 0.5  # Return moderate quality if assessment fails
    
    def _store_interaction(self, state: RAGState) -> RAGState:
        """Store interaction in both short-term (DynamoDB) and long-term (S3) storage"""
        try:
            timestamp = int(time.time())
            expires_at = timestamp + (30 * 24 * 60 * 60)  # 30 days TTL
            
            # Store in DynamoDB for short-term access
            table = self.dynamodb.Table(self.dynamodb_table_name)
            table.put_item(
                Item={
                    'channel_id': state["channel_id"],
                    'timestamp': timestamp,
                    'user_id': state["user_id"],
                    'user_query': state["user_query"],
                    'ai_response': state["final_response"],
                    'context_used': state["context"],
                    'expires_at': expires_at
                }
            )
            
            # Create embeddings for the interaction and store in vector DB
            self._store_in_vector_db(state)
            
            # Archive to S3 for long-term storage
            self._archive_to_s3(state, timestamp)
            
        except Exception as e:
            print(f"Error storing interaction: {str(e)}")
            
        return state
    
    def _store_in_vector_db(self, state: RAGState):
        """Store interaction embeddings in OpenSearch for future retrieval with enhanced metadata"""
        try:
            # Create enhanced document for vectorization using processed query
            processed_query = state.get("processed_query", state["user_query"])
            content = f"Q: {processed_query}\nA: {state['final_response']}"
            
            # Generate embedding from the enhanced content
            embedding = self.embeddings.embed_query(content)
            
            # Create document with enhanced metadata
            doc = {
                "content": content,
                "content_vector": embedding,
                "channel_id": state["channel_id"],
                "user_id": state["user_id"],
                "timestamp": int(time.time()),
                "metadata": {
                    "query": state["user_query"],
                    "processed_query": processed_query,
                    "response": state["final_response"],
                    "context_length": len(state["context"]),
                    "retrieved_docs_count": len(state["retrieved_docs"]),
                    "context_quality": self._assess_context_quality(state["retrieved_docs"], processed_query),
                    "semantic_processing": True
                }
            }
            
            # Index document
            self.opensearch_client.index(
                index="chat_vectors",
                body=doc,
                refresh=True
            )
            
        except Exception as e:
            print(f"Error storing in vector DB: {str(e)}")
    
    def _archive_to_s3(self, state: RAGState, timestamp: int):
        """Archive interaction to S3 for long-term storage"""
        try:
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d")
            key = f"chat_archive/{date_str}/{state['channel_id']}/{timestamp}.json"
            
            archive_data = {
                "channel_id": state["channel_id"],
                "user_id": state["user_id"],
                "timestamp": timestamp,
                "user_query": state["user_query"],
                "ai_response": state["final_response"],
                "context_used": state["context"],
                "retrieved_docs": state["retrieved_docs"]
            }
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket_name,
                Key=key,
                Body=json.dumps(archive_data, indent=2),
                ContentType='application/json'
            )
            
        except Exception as e:
            print(f"Error archiving to S3: {str(e)}")
    
    def _get_channel_history(self, channel_id: str, limit: int = 10) -> List[str]:
        """Retrieve recent channel history from DynamoDB"""
        try:
            table = self.dynamodb.Table(self.dynamodb_table_name)
            
            response = table.query(
                KeyConditionExpression='channel_id = :channel_id',
                ExpressionAttributeValues={':channel_id': channel_id},
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )
            
            history = []
            for item in response['Items']:
                history.append(f"User: {item['user_query']} | AI: {item['ai_response']}")
            
            return list(reversed(history))  # Chronological order
            
        except Exception as e:
            print(f"Error retrieving channel history: {str(e)}")
            return []
    
    def process_message(self, user_query: str, channel_id: str, user_id: str) -> str:
        """Main entry point for processing messages through RAG pipeline"""
        try:
            initial_state = RAGState(
                messages=[],
                context=[],
                user_query=user_query,
                channel_id=channel_id,
                user_id=user_id,
                retrieved_docs=[],
                final_response="",
                processed_query=""
            )
            
            # Run through LangGraph workflow
            final_state = self.workflow.invoke(initial_state)
            
            return final_state["final_response"]
            
        except Exception as e:
            print(f"Error in RAG processing: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again."

class DocumentProcessor:
    """Advanced document processing with chunking and parsing"""
    
    def __init__(self, embeddings_model, opensearch_client):
        self.embeddings = embeddings_model
        self.opensearch_client = opensearch_client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len
        )
    
    def process_and_store_document(self, content: str, metadata: Dict[str, Any], channel_id: str):
        """Process document with advanced parsing and chunking"""
        try:
            # Advanced text cleaning and preprocessing
            cleaned_content = self._clean_text(content)
            
            # Smart chunking based on content structure
            chunks = self._smart_chunk(cleaned_content)
            
            # Generate embeddings and store
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                
                doc = {
                    "content": chunk,
                    "content_vector": embedding,
                    "channel_id": channel_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": int(time.time()),
                    "metadata": {
                        **metadata,
                        "chunk_id": f"{metadata.get('doc_id', 'unknown')}_{i}",
                        "processing_method": "advanced_rag"
                    }
                }
                
                self.opensearch_client.index(
                    index="chat_vectors",
                    body=doc,
                    refresh=True
                )
                
        except Exception as e:
            print(f"Error processing document: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def _smart_chunk(self, text: str) -> List[str]:
        """Smart chunking that preserves semantic meaning"""
        # Use tiktoken for token counting
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # If text is short enough, return as single chunk
        if len(encoding.encode(text)) <= 800:
            return [text]
        
        # Use recursive character splitter for longer texts
        chunks = self.text_splitter.split_text(text)
        
        # Post-process chunks to ensure quality
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Filter out very short chunks
                processed_chunks.append(chunk.strip())
        
        return processed_chunks

def create_opensearch_index():
    """Create the vector index in OpenSearch"""
    try:
        opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
        client = OpenSearch(
            hosts=[{'host': opensearch_endpoint.replace('https://', ''), 'port': 443}],
            http_auth=('admin', 'admin'),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        
        index_mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "content_vector": {
                        "type": "knn_vector",
                        "dimension": 1536,  # Titan embedding dimension
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "channel_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "timestamp": {"type": "long"},
                    "metadata": {"type": "object"}
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            }
        }
        
        if not client.indices.exists(index="chat_vectors"):
            client.indices.create(index="chat_vectors", body=index_mapping)
            print("Created chat_vectors index")
        
    except Exception as e:
        print(f"Error creating OpenSearch index: {str(e)}")