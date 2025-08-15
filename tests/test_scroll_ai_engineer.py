"""
Unit tests for ScrollAIEngineer agent
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from uuid import uuid4

from scrollintel.agents.scroll_ai_engineer import (
    ScrollAIEngineer, 
    AIModelType, 
    VectorStoreType,
    RAGConfiguration,
    LLMConfiguration,
    MemoryConfiguration
)
from scrollintel.core.interfaces import (
    AgentRequest, 
    AgentResponse, 
    AgentType, 
    ResponseStatus,
    AgentCapability
)


class TestScrollAIEngineer:
    """Test suite for ScrollAIEngineer agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a ScrollAIEngineer instance for testing"""
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
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample agent request"""
        return AgentRequest(
            id=str(uuid4()),
            user_id="test-user",
            agent_id="scroll-ai-engineer",
            prompt="Test prompt for AI engineering",
            context={"test": "context"},
            priority=1,
            created_at=datetime.now()
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.agent_id == "scroll-ai-engineer"
        assert agent.name == "ScrollAI Engineer"
        assert agent.agent_type == AgentType.AI_ENGINEER
        assert len(agent.capabilities) == 5
        
        # Check capabilities
        capability_names = [cap.name for cap in agent.capabilities]
        expected_capabilities = [
            "rag_implementation",
            "vector_operations", 
            "llm_integration",
            "langchain_workflows",
            "memory_management"
        ]
        for expected in expected_capabilities:
            assert expected in capability_names
    
    def test_get_capabilities(self, agent):
        """Test get_capabilities method"""
        capabilities = agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert len(capabilities) == 5
        assert all(isinstance(cap, AgentCapability) for cap in capabilities)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, agent):
        """Test successful health check"""
        with patch.object(agent.openai_client.models, 'list', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = Mock()
            
            # Mock vector store similarity search
            agent.vector_store.similarity_search.return_value = [Mock()]
            
            result = await agent.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, agent):
        """Test health check failure"""
        with patch.object(agent.openai_client.models, 'list', new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("API Error")
            
            result = await agent.health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_process_request_rag(self, agent, sample_request):
        """Test processing RAG request"""
        sample_request.prompt = "implement rag system for document search"
        sample_request.context = {
            "documents": ["Test document 1", "Test document 2"],
            "query": "What is the main topic?"
        }
        
        with patch.object(agent, '_handle_rag_request', new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = "RAG implementation successful"
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.SUCCESS
            assert "RAG implementation successful" in response.content
            mock_rag.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_vector_operations(self, agent, sample_request):
        """Test processing vector operations request"""
        sample_request.prompt = "generate embeddings for similarity search"
        sample_request.context = {
            "operation": "embed",
            "text": "Sample text for embedding"
        }
        
        with patch.object(agent, '_handle_vector_operations', new_callable=AsyncMock) as mock_vector:
            mock_vector.return_value = "Vector operations completed"
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.SUCCESS
            assert "Vector operations completed" in response.content
            mock_vector.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_langchain_workflow(self, agent, sample_request):
        """Test processing LangChain workflow request"""
        sample_request.prompt = "create langchain workflow for document processing"
        sample_request.context = {
            "workflow_type": "conversational_retrieval"
        }
        
        with patch.object(agent, '_handle_langchain_workflow', new_callable=AsyncMock) as mock_workflow:
            mock_workflow.return_value = "LangChain workflow created"
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.SUCCESS
            assert "LangChain workflow created" in response.content
            mock_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_memory_operations(self, agent, sample_request):
        """Test processing memory operations request"""
        sample_request.prompt = "manage conversation memory and context"
        sample_request.context = {
            "operation": "add_to_memory",
            "user_input": "Hello",
            "ai_response": "Hi there!"
        }
        
        with patch.object(agent, '_handle_memory_operations', new_callable=AsyncMock) as mock_memory:
            mock_memory.return_value = "Memory operations completed"
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.SUCCESS
            assert "Memory operations completed" in response.content
            mock_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_audio_processing(self, agent, sample_request):
        """Test processing audio/Whisper request"""
        sample_request.prompt = "transcribe audio using whisper"
        sample_request.context = {
            "audio_file_path": "/path/to/audio.mp3"
        }
        
        with patch.object(agent, '_handle_audio_processing', new_callable=AsyncMock) as mock_audio:
            mock_audio.return_value = "Audio transcription completed"
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.SUCCESS
            assert "Audio transcription completed" in response.content
            mock_audio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_general_llm(self, agent, sample_request):
        """Test processing general LLM request"""
        sample_request.prompt = "general AI question"
        sample_request.context = {
            "model_type": "gpt-4",
            "temperature": 0.7
        }
        
        with patch.object(agent, '_handle_general_llm_request', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "LLM response generated"
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.SUCCESS
            assert "LLM response generated" in response.content
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_error_handling(self, agent, sample_request):
        """Test error handling in process_request"""
        with patch.object(agent, '_handle_general_llm_request', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Test error")
            
            response = await agent.process_request(sample_request)
            
            assert isinstance(response, AgentResponse)
            assert response.status == ResponseStatus.ERROR
            assert "Error processing AI engineering request" in response.content
            assert response.error_message == "Test error"
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, agent):
        """Test embedding generation"""
        test_text = "Sample text for embedding"
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = mock_embedding
        
        with patch.object(agent.openai_client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await agent._generate_embeddings(test_text)
            
            assert result == mock_embedding
            mock_create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=test_text
            )
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, agent):
        """Test embedding generation error handling"""
        with patch.object(agent.openai_client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            with pytest.raises(Exception) as exc_info:
                await agent._generate_embeddings("test text")
            
            assert "Failed to generate embeddings" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_add_documents_to_vector_store(self, agent):
        """Test adding documents to vector store"""
        documents = ["Document 1 content", "Document 2 content"]
        
        # Mock text splitter
        with patch.object(agent.text_splitter, 'split_text') as mock_split:
            mock_split.side_effect = [["chunk1", "chunk2"], ["chunk3", "chunk4"]]
            
            # Mock vector store add_documents
            agent.vector_store.add_documents = Mock()
            
            result = await agent._add_documents_to_vector_store(documents)
            
            assert result["success"] is True
            assert result["chunks_created"] == 4
            assert result["documents_processed"] == 2
            agent.vector_store.add_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_documents_to_vector_store_no_store(self, agent):
        """Test adding documents when vector store is not initialized"""
        agent.vector_store = None
        
        with pytest.raises(Exception) as exc_info:
            await agent._add_documents_to_vector_store(["test doc"])
        
        assert "Vector store not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_perform_similarity_search(self, agent):
        """Test similarity search"""
        query = "test query"
        top_k = 3
        
        # Mock search results
        mock_doc1 = Mock()
        mock_doc1.page_content = "Document 1 content"
        mock_doc1.metadata = {"source": "doc1"}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Document 2 content"
        mock_doc2.metadata = {"source": "doc2"}
        
        mock_results = [(mock_doc1, 0.9), (mock_doc2, 0.8)]
        
        agent.vector_store.similarity_search_with_score = Mock(return_value=mock_results)
        
        result = await agent._perform_similarity_search(query, top_k)
        
        assert len(result) == 2
        assert result[0]["content"] == "Document 1 content"
        assert result[0]["score"] == 0.9
        assert result[1]["content"] == "Document 2 content"
        assert result[1]["score"] == 0.8
        
        agent.vector_store.similarity_search_with_score.assert_called_once_with(query, k=top_k)
    
    @pytest.mark.asyncio
    async def test_call_openai_model(self, agent):
        """Test OpenAI model call"""
        prompt = "Test prompt"
        model = "gpt-4"
        temperature = 0.7
        max_tokens = 1000
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        
        with patch.object(agent.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await agent._call_openai_model(prompt, model, temperature, max_tokens)
            
            assert result == "Test response"
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_anthropic_model(self, agent):
        """Test Anthropic model call"""
        prompt = "Test prompt"
        model = "claude-3-sonnet-20240229"
        temperature = 0.7
        max_tokens = 1000
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        
        with patch.object(agent.anthropic_client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await agent._call_anthropic_model(prompt, model, temperature, max_tokens)
            
            assert result == "Test response"
            mock_create.assert_called_once()
    
    def test_rag_configuration_management(self, agent):
        """Test RAG configuration get/set methods"""
        # Test getting current configuration
        config = agent.get_rag_configuration()
        assert isinstance(config, RAGConfiguration)
        assert config.vector_store_type == VectorStoreType.PINECONE
        
        # Test updating configuration
        new_config = RAGConfiguration(
            vector_store_type=VectorStoreType.SUPABASE,
            embedding_model="text-embedding-ada-002",
            chunk_size=500,
            chunk_overlap=100,
            top_k_results=3,
            similarity_threshold=0.8,
            index_name="test-index"
        )
        
        agent.update_rag_configuration(new_config)
        updated_config = agent.get_rag_configuration()
        assert updated_config.vector_store_type == VectorStoreType.SUPABASE
        assert updated_config.chunk_size == 500
    
    def test_llm_configuration_management(self, agent):
        """Test LLM configuration get/set methods"""
        # Test getting current configuration
        config = agent.get_llm_configuration()
        assert isinstance(config, LLMConfiguration)
        assert config.model_type == AIModelType.GPT4
        
        # Test updating configuration
        new_config = LLMConfiguration(
            model_type=AIModelType.CLAUDE3_SONNET,
            temperature=0.5,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        agent.update_llm_configuration(new_config)
        updated_config = agent.get_llm_configuration()
        assert updated_config.model_type == AIModelType.CLAUDE3_SONNET
        assert updated_config.temperature == 0.5
    
    @pytest.mark.asyncio
    async def test_get_vector_store_stats(self, agent):
        """Test getting vector store statistics"""
        # Mock Pinecone index stats
        mock_stats = Mock()
        mock_stats.total_vector_count = 1000
        mock_stats.dimension = 1536
        mock_stats.index_fullness = 0.5
        
        # Mock the pinecone index that's already set up in the agent
        agent.pinecone_index = Mock()
        agent.pinecone_index.describe_index_stats.return_value = {
            'total_vector_count': mock_stats.total_vector_count,
            'dimension': mock_stats.dimension,
            'index_fullness': mock_stats.index_fullness
        }
        
        result = await agent.get_vector_store_stats()
        
        assert result["status"] == "active"
        assert result["total_vectors"] == 1000
        assert result["dimension"] == 1536
        assert result["index_fullness"] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_vector_store_stats_not_initialized(self, agent):
        """Test getting vector store stats when not initialized"""
        agent.vector_store = None
        
        result = await agent.get_vector_store_stats()
        
        assert result["status"] == "not_initialized"
    
    @pytest.mark.asyncio
    async def test_clear_vector_store(self, agent):
        """Test clearing vector store"""
        # Mock the pinecone index that's already set up in the agent
        agent.pinecone_index = Mock()
        agent.pinecone_index.delete = Mock()
        
        result = await agent.clear_vector_store()
        
        assert result["success"] is True
        assert "cleared" in result["message"]
        agent.pinecone_index.delete.assert_called_once_with(delete_all=True)
    
    @pytest.mark.asyncio
    async def test_clear_vector_store_not_initialized(self, agent):
        """Test clearing vector store when not initialized"""
        agent.vector_store = None
        
        result = await agent.clear_vector_store()
        
        assert result["success"] is False
        assert "not initialized" in result["error"]
    
    def test_format_search_results(self, agent):
        """Test formatting search results"""
        results = [
            {
                "content": "First document content for testing",
                "score": 0.95,
                "metadata": {"source": "doc1"}
            },
            {
                "content": "Second document content for testing",
                "score": 0.87,
                "metadata": {"source": "doc2"}
            }
        ]
        
        formatted = agent._format_search_results(results)
        
        assert "Result 1 (Score: 0.9500)" in formatted
        assert "Result 2 (Score: 0.8700)" in formatted
        assert "First document content" in formatted
        assert "Second document content" in formatted
    
    def test_format_search_results_empty(self, agent):
        """Test formatting empty search results"""
        results = []
        
        formatted = agent._format_search_results(results)
        
        assert formatted == "No results found."


class TestRAGConfiguration:
    """Test RAG configuration dataclass"""
    
    def test_rag_configuration_creation(self):
        """Test creating RAG configuration"""
        config = RAGConfiguration(
            vector_store_type=VectorStoreType.PINECONE,
            embedding_model="text-embedding-ada-002",
            chunk_size=1000,
            chunk_overlap=200,
            top_k_results=5,
            similarity_threshold=0.7,
            index_name="test-index"
        )
        
        assert config.vector_store_type == VectorStoreType.PINECONE
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.top_k_results == 5
        assert config.similarity_threshold == 0.7
        assert config.index_name == "test-index"


class TestLLMConfiguration:
    """Test LLM configuration dataclass"""
    
    def test_llm_configuration_creation(self):
        """Test creating LLM configuration"""
        config = LLMConfiguration(
            model_type=AIModelType.GPT4,
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        assert config.model_type == AIModelType.GPT4
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0


class TestAIModelType:
    """Test AI model type enum"""
    
    def test_ai_model_types(self):
        """Test AI model type values"""
        assert AIModelType.GPT4.value == "gpt-4"
        assert AIModelType.GPT35_TURBO.value == "gpt-3.5-turbo"
        assert AIModelType.CLAUDE3_OPUS.value == "claude-3-opus-20240229"
        assert AIModelType.CLAUDE3_SONNET.value == "claude-3-sonnet-20240229"
        assert AIModelType.CLAUDE3_HAIKU.value == "claude-3-haiku-20240307"
        assert AIModelType.WHISPER.value == "whisper-1"


class TestVectorStoreType:
    """Test vector store type enum"""
    
    def test_vector_store_types(self):
        """Test vector store type values"""
        assert VectorStoreType.PINECONE.value == "pinecone"
        assert VectorStoreType.SUPABASE.value == "supabase"
        assert VectorStoreType.LOCAL.value == "local"


if __name__ == "__main__":
    pytest.main([__file__])