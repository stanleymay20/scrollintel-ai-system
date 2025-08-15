"""
Tests for Automated Research Engine

Tests all components:
- Topic generation
- Literature analysis
- Hypothesis formation
- Research planning
- Autonomous research coordination
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.automated_research_engine import (
    AutomatedResearchEngine,
    TopicGenerator,
    LiteratureAnalyzer,
    HypothesisFormer,
    ResearchPlanner,
    ResearchDomain,
    ResearchTopic,
    LiteratureAnalysis,
    Hypothesis,
    ResearchPlan,
    ResearchMethodology
)


class TestTopicGenerator:
    """Test topic generation functionality"""
    
    @pytest.fixture
    def topic_generator(self):
        return TopicGenerator()
    
    @pytest.mark.asyncio
    async def test_generate_topics_success(self, topic_generator):
        """Test successful topic generation"""
        domain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
        count = 3
        
        topics = await topic_generator.generate_topics(domain, count)
        
        assert len(topics) == count
        assert all(isinstance(topic, ResearchTopic) for topic in topics)
        assert all(topic.domain == domain for topic in topics)
        assert all(topic.novelty_score > 0 for topic in topics)
        assert all(topic.feasibility_score > 0 for topic in topics)
        assert all(topic.impact_potential > 0 for topic in topics)
        assert all(len(topic.research_gaps) > 0 for topic in topics)
    
    @pytest.mark.asyncio
    async def test_generate_topics_different_domains(self, topic_generator):
        """Test topic generation for different domains"""
        domains = [ResearchDomain.MACHINE_LEARNING, ResearchDomain.QUANTUM_COMPUTING]
        
        for domain in domains:
            topics = await topic_generator.generate_topics(domain, 2)
            assert len(topics) == 2
            assert all(topic.domain == domain for topic in topics)
    
    @pytest.mark.asyncio
    async def test_generate_topics_sorting(self, topic_generator):
        """Test that topics are sorted by combined novelty and impact"""
        domain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
        topics = await topic_generator.generate_topics(domain, 5)
        
        # Check that topics are sorted in descending order of combined score
        for i in range(len(topics) - 1):
            current_score = (topics[i].novelty_score + topics[i].impact_potential) / 2
            next_score = (topics[i + 1].novelty_score + topics[i + 1].impact_potential) / 2
            assert current_score >= next_score
    
    @pytest.mark.asyncio
    async def test_create_research_topic(self, topic_generator):
        """Test individual topic creation"""
        domain = ResearchDomain.ROBOTICS
        keywords = ["autonomous", "navigation", "learning"]
        
        topic = await topic_generator._create_research_topic(domain, keywords, 0)
        
        assert isinstance(topic, ResearchTopic)
        assert topic.domain == domain
        assert len(topic.title) > 0
        assert len(topic.description) > 0
        assert len(topic.keywords) > 0
        assert topic.novelty_score > 0
        assert topic.feasibility_score > 0
        assert topic.impact_potential > 0


class TestLiteratureAnalyzer:
    """Test literature analysis functionality"""
    
    @pytest.fixture
    def literature_analyzer(self):
        return LiteratureAnalyzer()
    
    @pytest.fixture
    def sample_topic(self):
        return ResearchTopic(
            title="Advanced Machine Learning for Autonomous Systems",
            domain=ResearchDomain.MACHINE_LEARNING,
            description="Research on ML applications in autonomous systems",
            keywords=["machine learning", "autonomous systems", "robotics"]
        )
    
    @pytest.mark.asyncio
    async def test_analyze_literature_success(self, literature_analyzer, sample_topic):
        """Test successful literature analysis"""
        analysis = await literature_analyzer.analyze_literature(sample_topic)
        
        assert isinstance(analysis, LiteratureAnalysis)
        assert analysis.topic_id == sample_topic.id
        assert len(analysis.sources) > 0
        assert len(analysis.knowledge_gaps) > 0
        assert len(analysis.research_trends) > 0
        assert len(analysis.key_findings) > 0
        assert len(analysis.methodological_gaps) > 0
        assert len(analysis.theoretical_gaps) > 0
        assert len(analysis.empirical_gaps) > 0
        assert 0 <= analysis.analysis_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_literature_caching(self, literature_analyzer, sample_topic):
        """Test that literature analysis results are cached"""
        # First analysis
        analysis1 = await literature_analyzer.analyze_literature(sample_topic)
        
        # Second analysis should return cached result
        analysis2 = await literature_analyzer.analyze_literature(sample_topic)
        
        assert analysis1 is analysis2  # Same object reference due to caching
    
    @pytest.mark.asyncio
    async def test_search_literature(self, literature_analyzer, sample_topic):
        """Test literature search functionality"""
        sources = await literature_analyzer._search_literature(sample_topic)
        
        assert len(sources) > 0
        assert all(hasattr(source, 'title') for source in sources)
        assert all(hasattr(source, 'authors') for source in sources)
        assert all(hasattr(source, 'relevance_score') for source in sources)
        assert all(0 <= source.relevance_score <= 1 for source in sources)
    
    @pytest.mark.asyncio
    async def test_identify_knowledge_gaps(self, literature_analyzer, sample_topic):
        """Test knowledge gap identification"""
        sources = await literature_analyzer._search_literature(sample_topic)
        gaps = await literature_analyzer._identify_knowledge_gaps(sample_topic, sources)
        
        assert len(gaps) > 0
        assert all(isinstance(gap, str) for gap in gaps)
        assert all(len(gap) > 0 for gap in gaps)


class TestHypothesisFormer:
    """Test hypothesis formation functionality"""
    
    @pytest.fixture
    def hypothesis_former(self):
        return HypothesisFormer()
    
    @pytest.fixture
    def sample_analysis(self):
        return LiteratureAnalysis(
            topic_id="test-topic-id",
            knowledge_gaps=[
                "Limited scalability studies in machine learning",
                "Insufficient real-world validation of algorithms",
                "Lack of standardized evaluation metrics"
            ],
            research_trends=[
                "Increasing focus on autonomous systems",
                "Growing emphasis on explainability"
            ],
            key_findings=[
                "Current methods show promise but lack scalability",
                "Theoretical frameworks need empirical validation"
            ]
        )
    
    @pytest.mark.asyncio
    async def test_form_hypotheses_success(self, hypothesis_former, sample_analysis):
        """Test successful hypothesis formation"""
        hypotheses = await hypothesis_former.form_hypotheses(sample_analysis)
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        assert all(h.topic_id == sample_analysis.topic_id for h in hypotheses)
        assert all(len(h.statement) > 0 for h in hypotheses)
        assert all(len(h.null_hypothesis) > 0 for h in hypotheses)
        assert all(len(h.alternative_hypothesis) > 0 for h in hypotheses)
        assert all(len(h.variables) > 0 for h in hypotheses)
        assert all(0 <= h.testability_score <= 1 for h in hypotheses)
        assert all(0 <= h.novelty_score <= 1 for h in hypotheses)
        assert all(0 <= h.significance_potential <= 1 for h in hypotheses)
    
    @pytest.mark.asyncio
    async def test_hypotheses_sorting(self, hypothesis_former, sample_analysis):
        """Test that hypotheses are sorted by quality metrics"""
        hypotheses = await hypothesis_former.form_hypotheses(sample_analysis)
        
        # Check sorting by combined testability and significance
        for i in range(len(hypotheses) - 1):
            current_score = (hypotheses[i].testability_score + hypotheses[i].significance_potential) / 2
            next_score = (hypotheses[i + 1].testability_score + hypotheses[i + 1].significance_potential) / 2
            assert current_score >= next_score
    
    @pytest.mark.asyncio
    async def test_create_hypothesis_from_gap(self, hypothesis_former, sample_analysis):
        """Test hypothesis creation from knowledge gap"""
        gap = "Limited scalability studies in machine learning"
        hypothesis = await hypothesis_former._create_hypothesis_from_gap(gap, sample_analysis)
        
        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.topic_id == sample_analysis.topic_id
        assert len(hypothesis.statement) > 0
        assert len(hypothesis.variables) > 0
        assert len(hypothesis.required_resources) > 0
    
    @pytest.mark.asyncio
    async def test_create_hypothesis_from_trend(self, hypothesis_former, sample_analysis):
        """Test hypothesis creation from research trend"""
        trend = "Increasing focus on autonomous systems"
        hypothesis = await hypothesis_former._create_hypothesis_from_trend(trend, sample_analysis)
        
        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.topic_id == sample_analysis.topic_id
        assert len(hypothesis.statement) > 0
        assert len(hypothesis.variables) > 0
        assert len(hypothesis.required_resources) > 0


class TestResearchPlanner:
    """Test research planning functionality"""
    
    @pytest.fixture
    def research_planner(self):
        return ResearchPlanner()
    
    @pytest.fixture
    def sample_hypothesis(self):
        return Hypothesis(
            topic_id="test-topic-id",
            statement="Novel machine learning algorithms will improve autonomous system performance",
            null_hypothesis="No improvement in performance",
            alternative_hypothesis="Significant improvement in performance",
            variables={
                "independent": "machine learning algorithm",
                "dependent": "system performance",
                "control": "baseline algorithm"
            }
        )
    
    @pytest.fixture
    def sample_analysis(self):
        return LiteratureAnalysis(topic_id="test-topic-id")
    
    @pytest.mark.asyncio
    async def test_create_research_plan_success(self, research_planner, sample_hypothesis, sample_analysis):
        """Test successful research plan creation"""
        plan = await research_planner.create_research_plan(sample_hypothesis, sample_analysis)
        
        assert isinstance(plan, ResearchPlan)
        assert plan.topic_id == sample_hypothesis.topic_id
        assert plan.hypothesis_id == sample_hypothesis.id
        assert len(plan.title) > 0
        assert len(plan.objectives) > 0
        assert isinstance(plan.methodology, ResearchMethodology)
        assert len(plan.timeline) > 0
        assert len(plan.milestones) > 0
        assert len(plan.resource_requirements) > 0
        assert len(plan.success_criteria) > 0
        assert len(plan.risk_assessment) > 0
    
    @pytest.mark.asyncio
    async def test_select_methodology(self, research_planner, sample_hypothesis, sample_analysis):
        """Test methodology selection"""
        methodology = await research_planner._select_methodology(sample_hypothesis, sample_analysis)
        
        assert isinstance(methodology, ResearchMethodology)
        assert len(methodology.name) > 0
        assert len(methodology.methodology_type) > 0
        assert len(methodology.data_collection_methods) > 0
        assert len(methodology.analysis_methods) > 0
        assert len(methodology.validation_approaches) > 0
    
    @pytest.mark.asyncio
    async def test_create_timeline(self, research_planner, sample_hypothesis):
        """Test timeline creation"""
        timeline = await research_planner._create_timeline(sample_hypothesis)
        
        assert isinstance(timeline, dict)
        assert len(timeline) > 0
        assert all(isinstance(date, datetime) for date in timeline.values())
        
        # Check chronological order
        dates = list(timeline.values())
        for i in range(len(dates) - 1):
            assert dates[i] <= dates[i + 1]
    
    @pytest.mark.asyncio
    async def test_assess_resource_requirements(self, research_planner, sample_hypothesis):
        """Test resource requirement assessment"""
        methodology = ResearchMethodology(methodology_type="computational")
        resources = await research_planner._assess_resource_requirements(sample_hypothesis, methodology)
        
        assert isinstance(resources, dict)
        assert "computational" in resources
        assert "human" in resources
        assert "financial" in resources
        assert "data" in resources
        
        # Check resource structure
        assert "cpu_hours" in resources["computational"]
        assert "researcher_hours" in resources["human"]
        assert "total_budget" in resources["financial"]


class TestAutomatedResearchEngine:
    """Test the main automated research engine"""
    
    @pytest.fixture
    def research_engine(self):
        return AutomatedResearchEngine()
    
    @pytest.mark.asyncio
    async def test_generate_research_topics(self, research_engine):
        """Test topic generation through main engine"""
        domain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
        topics = await research_engine.generate_research_topics(domain, 3)
        
        assert len(topics) == 3
        assert all(isinstance(topic, ResearchTopic) for topic in topics)
        assert all(topic.domain == domain for topic in topics)
    
    @pytest.mark.asyncio
    async def test_analyze_literature(self, research_engine):
        """Test literature analysis through main engine"""
        topic = ResearchTopic(
            title="Test Topic",
            domain=ResearchDomain.MACHINE_LEARNING,
            keywords=["test", "analysis"]
        )
        
        analysis = await research_engine.analyze_literature(topic)
        
        assert isinstance(analysis, LiteratureAnalysis)
        assert analysis.topic_id == topic.id
    
    @pytest.mark.asyncio
    async def test_form_hypotheses(self, research_engine):
        """Test hypothesis formation through main engine"""
        analysis = LiteratureAnalysis(
            topic_id="test-id",
            knowledge_gaps=["gap1", "gap2"],
            research_trends=["trend1", "trend2"]
        )
        
        hypotheses = await research_engine.form_hypotheses(analysis)
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
    
    @pytest.mark.asyncio
    async def test_create_research_plan(self, research_engine):
        """Test research plan creation through main engine"""
        hypothesis = Hypothesis(
            topic_id="test-topic",
            statement="Test hypothesis"
        )
        analysis = LiteratureAnalysis(topic_id="test-topic")
        
        plan = await research_engine.create_research_plan(hypothesis, analysis)
        
        assert isinstance(plan, ResearchPlan)
        assert plan.hypothesis_id == hypothesis.id
    
    @pytest.mark.asyncio
    async def test_conduct_autonomous_research_success(self, research_engine):
        """Test complete autonomous research process"""
        domain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
        topic_count = 2
        
        results = await research_engine.conduct_autonomous_research(domain, topic_count)
        
        assert "domain" in results
        assert "topics" in results
        assert "total_hypotheses" in results
        assert "total_plans" in results
        assert "started_at" in results
        assert "completed_at" in results
        
        assert results["domain"] == domain.value
        assert len(results["topics"]) == topic_count
        assert results["total_hypotheses"] > 0
        assert results["total_plans"] > 0
    
    @pytest.mark.asyncio
    async def test_conduct_autonomous_research_error_handling(self, research_engine):
        """Test error handling in autonomous research"""
        # Mock topic generator to raise exception
        with patch.object(research_engine.topic_generator, 'generate_topics', side_effect=Exception("Test error")):
            results = await research_engine.conduct_autonomous_research(ResearchDomain.ARTIFICIAL_INTELLIGENCE, 1)
            
            assert "error" in results
            assert results["error"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_get_research_status_not_found(self, research_engine):
        """Test getting status of non-existent project"""
        status = await research_engine.get_research_status("non-existent-id")
        
        assert "error" in status
        assert status["error"] == "Project not found"
    
    @pytest.mark.asyncio
    async def test_list_active_projects_empty(self, research_engine):
        """Test listing active projects when none exist"""
        projects = await research_engine.list_active_projects()
        
        assert isinstance(projects, list)
        assert len(projects) == 0


class TestIntegration:
    """Integration tests for the complete research workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_research_workflow(self):
        """Test complete research workflow from topic to plan"""
        engine = AutomatedResearchEngine()
        domain = ResearchDomain.QUANTUM_COMPUTING
        
        # Step 1: Generate topics
        topics = await engine.generate_research_topics(domain, 1)
        assert len(topics) == 1
        topic = topics[0]
        
        # Step 2: Analyze literature
        analysis = await engine.analyze_literature(topic)
        assert analysis.topic_id == topic.id
        
        # Step 3: Form hypotheses
        hypotheses = await engine.form_hypotheses(analysis)
        assert len(hypotheses) > 0
        hypothesis = hypotheses[0]
        
        # Step 4: Create research plan
        plan = await engine.create_research_plan(hypothesis, analysis)
        assert plan.topic_id == topic.id
        assert plan.hypothesis_id == hypothesis.id
        
        # Verify complete workflow
        assert isinstance(topic, ResearchTopic)
        assert isinstance(analysis, LiteratureAnalysis)
        assert isinstance(hypothesis, Hypothesis)
        assert isinstance(plan, ResearchPlan)
    
    @pytest.mark.asyncio
    async def test_autonomous_research_quality_metrics(self):
        """Test that autonomous research produces quality results"""
        engine = AutomatedResearchEngine()
        domain = ResearchDomain.BIOTECHNOLOGY
        
        results = await engine.conduct_autonomous_research(domain, 2)
        
        # Check overall quality
        assert results["total_hypotheses"] >= 2  # At least one hypothesis per topic
        assert results["total_plans"] >= 2      # At least one plan per topic
        
        # Check individual topic quality
        for topic_result in results["topics"]:
            topic = topic_result["topic"]
            analysis = topic_result["literature_analysis"]
            hypotheses = topic_result["hypotheses"]
            plans = topic_result["research_plans"]
            
            # Topic quality checks
            assert topic.novelty_score > 0.5
            assert topic.feasibility_score > 0.5
            assert topic.impact_potential > 0.5
            
            # Analysis quality checks
            assert analysis.analysis_confidence > 0.7
            assert len(analysis.knowledge_gaps) >= 3
            assert len(analysis.research_trends) >= 2
            
            # Hypothesis quality checks
            for hypothesis in hypotheses:
                assert hypothesis.testability_score > 0.6
                assert hypothesis.significance_potential > 0.6
                assert len(hypothesis.required_resources) > 0
            
            # Plan quality checks
            for plan in plans:
                assert len(plan.objectives) >= 3
                assert len(plan.milestones) >= 5
                assert len(plan.success_criteria) >= 3
                assert "computational" in plan.resource_requirements
                assert "human" in plan.resource_requirements


# Performance tests
class TestPerformance:
    """Performance tests for the automated research engine"""
    
    @pytest.mark.asyncio
    async def test_topic_generation_performance(self):
        """Test topic generation performance"""
        engine = AutomatedResearchEngine()
        domain = ResearchDomain.ARTIFICIAL_INTELLIGENCE
        
        start_time = datetime.now()
        topics = await engine.generate_research_topics(domain, 10)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        assert len(topics) == 10
        assert duration < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_literature_analysis_caching_performance(self):
        """Test that literature analysis caching improves performance"""
        analyzer = LiteratureAnalyzer()
        topic = ResearchTopic(
            title="Performance Test Topic",
            domain=ResearchDomain.MACHINE_LEARNING,
            keywords=["performance", "test"]
        )
        
        # First analysis (no cache)
        start_time = datetime.now()
        analysis1 = await analyzer.analyze_literature(topic)
        first_duration = (datetime.now() - start_time).total_seconds()
        
        # Second analysis (cached)
        start_time = datetime.now()
        analysis2 = await analyzer.analyze_literature(topic)
        second_duration = (datetime.now() - start_time).total_seconds()
        
        assert analysis1 is analysis2  # Same object due to caching
        assert second_duration < first_duration  # Cached should be faster


if __name__ == "__main__":
    pytest.main([__file__])