"""
Integration tests for ScrollAIEngineer agent
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from scrollintel.agents.scroll_ai_engineer import ScrollAIEngineer
from scrollintel.core.interfaces import AgentRequest, AgentType, ResponseStatus


class TestScrollAIEngineerIntegration:
    """Integration test suite for ScrollAIEngineer agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a ScrollAIEngineer instance for integration testing"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'PINECONE_API_KEY': 'test-pinecone-key',
            'PINECONE_ENVIRONMENT': 'test-env'
        }):
            with patch('scrollintel.agents.scroll_ai_engineer.PineconeClient') as mock_pinecone_client:
                mock_client = Mock()
                mock_client.list_indexes.return_value = []
                mock_client.create_index.return_value = None
                mock_client.Index.return_value = Mock()
                mock_pinecone_client.return_value = mock_client
                
                with patch('scrollintel.agents.scroll_ai_engineer.Pinecone') as mock_pinecone_class:
                    mock_vector_store = Mock()
                    mock_pinecone_class.return_value = mock_vector_store
                    
                    agent = ScrollAIEngineer()
                    agent.vector_store = mock_vector_store
                    return agent
    
    @pytest.mark.asyncio
    async def test_rag_workflow_integration(self, agent):
        """Test complete RAG workflow integration"""
        # Prepare test data
        documents = [
            "ScrollIntel is an advanced AI platform for data analysis and machine learning.",
            "The platform includes multiple AI agents for different tasks like CTO decisions and data science.",
            "RAG (Retrieval Augmented Generation) allows combining document retrieval with language generation."
        ]
        
        query = "What is ScrollIntel and how does it use RAG?"
        
        request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="implement rag system for document search",
            context={
                "documents": documents,
                "query": query
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock the vector store operations
        agent.vector_store.add_documents = Mock()
        
        # Mock similarity search results
        mock_doc = Mock()
        mock_doc.page_content = documents[0]
        mock_doc.metadata = {}
        agent.vector_store.similarity_search_with_score = Mock(
            return_value=[(mock_doc, 0.9)]
        )
        
        # Mock LangChain components
        with patch('scrollintel.agents.scroll_ai_engineer.RetrievalQA') as mock_qa:
            mock_chain = Mock()
            mock_chain.return_value = {
                "result": "ScrollIntel is an advanced AI platform that uses RAG for enhanced document search.",
                "source_documents": [mock_doc]
            }
            mock_qa.from_chain_type.return_value = mock_chain
            
            with patch('scrollintel.agents.scroll_ai_engineer.ChatOpenAI'):
                response = await agent.process_request(request)
        
        # Verify response
        assert response.status == ResponseStatus.SUCCESS
        assert "RAG Implementation Result" in response.content
        assert "ScrollIntel" in response.content
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_vector_operations_integration(self, agent):
        """Test vector operations integration"""
        request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="generate embeddings for similarity search",
            context={
                "operation": "embed",
                "text": "Sample text for embedding generation"
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock OpenAI embeddings API
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        with patch.object(agent.openai_client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_embedding_response
            
            response = await agent.process_request(request)
        
        # Verify response
        assert response.status == ResponseStatus.SUCCESS
        assert "Vector Embedding Generation" in response.content
        assert "Dimension: 1536" in response.content
        assert "text-embedding-ada-002" in response.content
    
    @pytest.mark.asyncio
    async def test_langchain_workflow_integration(self, agent):
        """Test LangChain workflow integration"""
        request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="create langchain workflow for document processing",
            context={
                "workflow_type": "conversational_retrieval"
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock LangChain components
        with patch('scrollintel.agents.scroll_ai_engineer.ConversationalRetrievalChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.return_value = {
                "answer": "This is a conversational retrieval response.",
                "source_documents": [Mock(), Mock()]
            }
            mock_chain_class.from_llm.return_value = mock_chain
            
            with patch('scrollintel.agents.scroll_ai_engineer.ChatOpenAI'):
                response = await agent.process_request(request)
        
        # Verify response
        assert response.status == ResponseStatus.SUCCESS
        assert "LangChain Conversational Retrieval Workflow" in response.content
        assert "conversational retrieval response" in response.content
        assert "2 documents retrieved" in response.content
    
    @pytest.mark.asyncio
    async def test_memory_management_integration(self, agent):
        """Test memory management integration"""
        # Test adding to memory
        add_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="manage conversation memory",
            context={
                "operation": "add_to_memory",
                "user_input": "What is machine learning?",
                "ai_response": "Machine learning is a subset of AI that enables computers to learn from data."
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await agent.process_request(add_request)
        
        # Verify memory addition
        assert response.status == ResponseStatus.SUCCESS
        assert "Memory Operation: Add to Memory" in response.content
        assert "What is machine learning?" in response.content
        
        # Test retrieving memory
        get_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="retrieve conversation memory",
            context={
                "operation": "get_memory"
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await agent.process_request(get_request)
        
        # Verify memory retrieval
        assert response.status == ResponseStatus.SUCCESS
        assert "Memory Operation: Retrieve Memory" in response.content
    
    @pytest.mark.asyncio
    async def test_audio_processing_integration(self, agent):
        """Test audio processing integration with Whisper"""
        request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="transcribe audio using whisper",
            context={
                "audio_file_path": "/path/to/test_audio.mp3"
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock Whisper transcription
        mock_transcript = Mock()
        mock_transcript.text = "This is a test transcription from Whisper."
        
        with patch('builtins.open', mock_open=True):
            with patch.object(agent.openai_client.audio.transcriptions, 'create', new_callable=AsyncMock) as mock_transcribe:
                mock_transcribe.return_value = mock_transcript
                
                with patch.object(agent, '_analyze_transcription', new_callable=AsyncMock) as mock_analyze:
                    mock_analyze.return_value = "Key topics: AI, transcription. Sentiment: neutral."
                    
                    response = await agent.process_request(request)
        
        # Verify response
        assert response.status == ResponseStatus.SUCCESS
        assert "Audio Processing with Whisper" in response.content
        assert "This is a test transcription from Whisper." in response.content
        assert "Key topics: AI, transcription" in response.content
    
    @pytest.mark.asyncio
    async def test_multi_model_llm_integration(self, agent):
        """Test integration with multiple LLM models"""
        # Test GPT-4
        gpt_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="Explain artificial intelligence",
            context={
                "model_type": "gpt-4",
                "temperature": 0.7
            },
            priority=1,
            created_at=datetime.now()
        )
        
        mock_gpt_response = Mock()
        mock_gpt_response.choices = [Mock()]
        mock_gpt_response.choices[0].message.content = "AI is the simulation of human intelligence in machines."
        
        with patch.object(agent.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_gpt_response
            
            response = await agent.process_request(gpt_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "gpt-4" in response.content
        assert "AI is the simulation of human intelligence" in response.content
        
        # Test Claude
        claude_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="Explain machine learning",
            context={
                "model_type": "claude-3-sonnet-20240229",
                "temperature": 0.5
            },
            priority=1,
            created_at=datetime.now()
        )
        
        mock_claude_response = Mock()
        mock_claude_response.content = [Mock()]
        mock_claude_response.content[0].text = "Machine learning enables computers to learn from data."
        
        with patch.object(agent.anthropic_client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_claude_response
            
            response = await agent.process_request(claude_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "claude-3-sonnet-20240229" in response.content
        assert "Machine learning enables computers" in response.content
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, agent):
        """Test error handling in integration scenarios"""
        request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="test error handling",
            context={},
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock API failure
        with patch.object(agent.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API rate limit exceeded")
            
            response = await agent.process_request(request)
        
        # Verify error handling
        assert response.status == ResponseStatus.ERROR
        assert "Error processing AI engineering request" in response.content
        assert response.error_message == "API rate limit exceeded"
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, agent):
        """Test configuration management integration"""
        # Test RAG configuration
        rag_config = agent.get_rag_configuration()
        assert rag_config.vector_store_type.value == "pinecone"
        assert rag_config.embedding_model == "text-embedding-ada-002"
        
        # Test LLM configuration
        llm_config = agent.get_llm_configuration()
        assert llm_config.model_type.value == "gpt-4"
        assert llm_config.temperature == 0.7
        
        # Test vector store stats
        # Mock the pinecone index that's already set up in the agent
        agent.pinecone_index = Mock()
        agent.pinecone_index.describe_index_stats.return_value = {
            'total_vector_count': 500,
            'dimension': 1536,
            'index_fullness': 0.3
        }
        
        stats = await agent.get_vector_store_stats()
        
        assert stats["status"] == "active"
        assert stats["total_vectors"] == 500
        assert stats["dimension"] == 1536
    
    @pytest.mark.asyncio
    async def test_end_to_end_rag_workflow(self, agent):
        """Test complete end-to-end RAG workflow"""
        # Step 1: Add documents to vector store
        documents = [
            "ScrollIntel provides advanced AI capabilities for enterprise data analysis.",
            "The platform includes RAG functionality for enhanced document search and retrieval.",
            "Vector databases enable semantic search across large document collections."
        ]
        
        add_docs_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="add documents to vector store",
            context={
                "operation": "add_documents",
                "documents": documents
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock vector store operations
        agent.vector_store.add_documents = Mock()
        
        with patch.object(agent.text_splitter, 'split_text') as mock_split:
            mock_split.side_effect = [["chunk1"], ["chunk2"], ["chunk3"]]
            
            response = await agent.process_request(add_docs_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Documents Added to Vector Store" in response.content
        
        # Step 2: Perform similarity search
        search_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="search for similar documents",
            context={
                "operation": "similarity_search",
                "text": "AI platform for data analysis",
                "top_k": 2
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock search results
        mock_doc1 = Mock()
        mock_doc1.page_content = documents[0]
        mock_doc1.metadata = {}
        mock_doc2 = Mock()
        mock_doc2.page_content = documents[1]
        mock_doc2.metadata = {}
        
        agent.vector_store.similarity_search_with_score = Mock(
            return_value=[(mock_doc1, 0.95), (mock_doc2, 0.87)]
        )
        
        response = await agent.process_request(search_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Similarity Search Results" in response.content
        assert "ScrollIntel provides advanced AI" in response.content
        
        # Step 3: Perform RAG query
        rag_request = AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="implement rag for question answering",
            context={
                "query": "What are the key features of ScrollIntel?"
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock RAG chain
        with patch('scrollintel.agents.scroll_ai_engineer.RetrievalQA') as mock_qa:
            mock_chain = Mock()
            mock_chain.return_value = {
                "result": "ScrollIntel's key features include advanced AI capabilities, RAG functionality, and vector database integration.",
                "source_documents": [mock_doc1, mock_doc2]
            }
            mock_qa.from_chain_type.return_value = mock_chain
            
            with patch('scrollintel.agents.scroll_ai_engineer.ChatOpenAI'):
                response = await agent.process_request(rag_request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "RAG Implementation Result" in response.content
        assert "key features include advanced AI" in response.content


if __name__ == "__main__":
    pytest.main([__file__])