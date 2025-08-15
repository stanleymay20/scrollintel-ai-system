"""
Board Presentation API Routes

This module provides REST API endpoints for board presentation framework functionality.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from ...engines.presentation_design_engine import BoardPresentationFramework
from ...engines.executive_data_visualization_engine import ExecutiveDataVisualizationEngine
from ...engines.qa_preparation_engine import QAPreparationEngine
from ...models.board_presentation_models import (
    BoardPresentation, PresentationFormat, QualityMetrics
)
from ...models.board_dynamics_models import Board


# Initialize router and framework
router = APIRouter(prefix="/api/v1/board-presentation", tags=["board-presentation"])
presentation_framework = BoardPresentationFramework()
visualization_engine = ExecutiveDataVisualizationEngine()
qa_preparation_engine = QAPreparationEngine()
logger = logging.getLogger(__name__)


# Request/Response Models
class PresentationRequest(BaseModel):
    """Request model for creating board presentations"""
    title: str
    content: Dict[str, Any]
    board_id: str
    presenter_id: str
    format_type: str = "strategic_overview"
    presentation_date: Optional[str] = None


class PresentationResponse(BaseModel):
    """Response model for board presentations"""
    id: str
    title: str
    board_id: str
    presenter_id: str
    format_type: str
    slide_count: int
    estimated_duration: int
    quality_score: Optional[float]
    executive_summary: str
    key_messages: List[str]
    success_metrics: List[str]
    created_at: str


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment"""
    clarity_score: float
    relevance_score: float
    engagement_score: float
    professional_score: float
    time_efficiency_score: float
    overall_score: float
    improvement_suggestions: List[str]


class OptimizationRequest(BaseModel):
    """Request model for presentation optimization"""
    presentation_id: str
    board_id: str
    optimization_preferences: Optional[Dict[str, Any]] = None


@router.post("/create", response_model=PresentationResponse)
async def create_board_presentation(request: PresentationRequest):
    """Create a new board presentation"""
    try:
        # Convert format type string to enum
        try:
            format_type = PresentationFormat(request.format_type.lower())
        except ValueError:
            format_type = PresentationFormat.STRATEGIC_OVERVIEW
        
        # Create mock board object (in real implementation, fetch from database)
        board = Board(
            id=request.board_id,
            name=f"Board {request.board_id}",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create presentation
        presentation = presentation_framework.create_optimized_presentation(
            content=request.content,
            board=board,
            format_type=format_type
        )
        
        # Calculate total estimated duration
        total_duration = sum(slide.estimated_duration for slide in presentation.slides)
        
        response = PresentationResponse(
            id=presentation.id,
            title=presentation.title,
            board_id=presentation.board_id,
            presenter_id=presentation.presenter_id,
            format_type=presentation.format_type.value,
            slide_count=len(presentation.slides),
            estimated_duration=total_duration,
            quality_score=presentation.quality_score,
            executive_summary=presentation.executive_summary,
            key_messages=presentation.key_messages,
            success_metrics=presentation.success_metrics,
            created_at=presentation.created_at.isoformat()
        )
        
        logger.info(f"Created board presentation: {presentation.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error creating board presentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create presentation: {str(e)}")


@router.get("/templates", response_model=List[Dict[str, Any]])
async def get_presentation_templates():
    """Get available presentation templates"""
    try:
        templates = presentation_framework.designer.design_templates
        
        template_list = []
        for template in templates:
            template_dict = {
                "id": template.id,
                "name": template.name,
                "format_type": template.format_type.value,
                "target_audience": [audience.value for audience in template.target_audience],
                "slide_structure": template.slide_structure,
                "design_guidelines": template.design_guidelines,
                "timing_recommendations": template.timing_recommendations
            }
            template_list.append(template_dict)
        
        return template_list
        
    except Exception as e:
        logger.error(f"Error retrieving presentation templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve templates: {str(e)}")


@router.post("/assess-quality/{presentation_id}", response_model=QualityAssessmentResponse)
async def assess_presentation_quality(presentation_id: str, board_id: str):
    """Assess the quality of a board presentation"""
    try:
        # In real implementation, fetch presentation from database
        # For now, create a mock assessment
        
        # Create mock board object
        board = Board(
            id=board_id,
            name=f"Board {board_id}",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create mock presentation for assessment
        mock_content = {
            "overview": "Strategic overview content",
            "key_metrics": {"revenue": 1000000, "growth": 15},
            "recommendations": ["Increase investment", "Expand market"]
        }
        
        presentation = presentation_framework.designer.create_board_presentation(
            content=mock_content,
            board=board
        )
        
        # Assess quality
        quality_metrics = presentation_framework.assessor.assess_quality(presentation, board)
        
        response = QualityAssessmentResponse(
            clarity_score=quality_metrics.clarity_score,
            relevance_score=quality_metrics.relevance_score,
            engagement_score=quality_metrics.engagement_score,
            professional_score=quality_metrics.professional_score,
            time_efficiency_score=quality_metrics.time_efficiency_score,
            overall_score=quality_metrics.overall_score,
            improvement_suggestions=quality_metrics.improvement_suggestions
        )
        
        logger.info(f"Assessed presentation quality: {presentation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error assessing presentation quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assess quality: {str(e)}")


@router.post("/optimize", response_model=PresentationResponse)
async def optimize_presentation(request: OptimizationRequest):
    """Optimize presentation for specific board preferences"""
    try:
        # Create mock board object
        board = Board(
            id=request.board_id,
            name=f"Board {request.board_id}",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create mock presentation for optimization
        mock_content = {
            "overview": "Strategic overview content",
            "performance": {"revenue": 1000000, "growth": 15},
            "recommendations": ["Increase investment", "Expand market"],
            "next_steps": "Implementation timeline"
        }
        
        presentation = presentation_framework.designer.create_board_presentation(
            content=mock_content,
            board=board
        )
        
        # Optimize presentation
        optimized_presentation = presentation_framework.optimizer.optimize_for_board(
            presentation, board
        )
        
        # Calculate total estimated duration
        total_duration = sum(slide.estimated_duration for slide in optimized_presentation.slides)
        
        response = PresentationResponse(
            id=optimized_presentation.id,
            title=optimized_presentation.title,
            board_id=optimized_presentation.board_id,
            presenter_id=optimized_presentation.presenter_id,
            format_type=optimized_presentation.format_type.value,
            slide_count=len(optimized_presentation.slides),
            estimated_duration=total_duration,
            quality_score=optimized_presentation.quality_score,
            executive_summary=optimized_presentation.executive_summary,
            key_messages=optimized_presentation.key_messages,
            success_metrics=optimized_presentation.success_metrics,
            created_at=optimized_presentation.created_at.isoformat()
        )
        
        logger.info(f"Optimized presentation: {request.presentation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing presentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize presentation: {str(e)}")


@router.get("/formats", response_model=List[Dict[str, str]])
async def get_presentation_formats():
    """Get available presentation formats"""
    try:
        formats = []
        for format_type in PresentationFormat:
            formats.append({
                "value": format_type.value,
                "name": format_type.value.replace('_', ' ').title(),
                "description": f"Format optimized for {format_type.value.replace('_', ' ')}"
            })
        
        return formats
        
    except Exception as e:
        logger.error(f"Error retrieving presentation formats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve formats: {str(e)}")


@router.post("/enhance/{presentation_id}", response_model=PresentationResponse)
async def enhance_presentation(presentation_id: str, board_id: str):
    """Enhance presentation based on quality assessment"""
    try:
        # Create mock board object
        board = Board(
            id=board_id,
            name=f"Board {board_id}",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create mock presentation
        mock_content = {
            "overview": "Strategic overview content",
            "performance": {"revenue": 1000000, "growth": 15},
            "recommendations": ["Increase investment", "Expand market"]
        }
        
        presentation = presentation_framework.designer.create_board_presentation(
            content=mock_content,
            board=board
        )
        
        # Assess quality
        quality_metrics = presentation_framework.assessor.assess_quality(presentation, board)
        
        # Enhance presentation
        enhanced_presentation = presentation_framework.assessor.enhance_presentation(
            presentation, quality_metrics
        )
        
        # Calculate total estimated duration
        total_duration = sum(slide.estimated_duration for slide in enhanced_presentation.slides)
        
        response = PresentationResponse(
            id=enhanced_presentation.id,
            title=enhanced_presentation.title,
            board_id=enhanced_presentation.board_id,
            presenter_id=enhanced_presentation.presenter_id,
            format_type=enhanced_presentation.format_type.value,
            slide_count=len(enhanced_presentation.slides),
            estimated_duration=total_duration,
            quality_score=enhanced_presentation.quality_score,
            executive_summary=enhanced_presentation.executive_summary,
            key_messages=enhanced_presentation.key_messages,
            success_metrics=enhanced_presentation.success_metrics,
            created_at=enhanced_presentation.created_at.isoformat()
        )
        
        logger.info(f"Enhanced presentation: {presentation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error enhancing presentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enhance presentation: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for board presentation service"""
    return {
        "status": "healthy",
        "service": "board-presentation",
        "version": "1.0.0",
        "timestamp": "2025-01-03T10:00:00Z"
    }


class VisualizationRequest(BaseModel):
    """Request model for creating executive visualizations"""
    data: Dict[str, Any]
    board_id: str
    board_context: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None


class VisualizationResponse(BaseModel):
    """Response model for executive visualizations"""
    id: str
    title: str
    visualization_type: str
    chart_type: str
    insights: List[str]
    executive_summary: str
    impact_score: float
    board_relevance: Dict[str, float]
    created_at: str


class VisualizationImpactResponse(BaseModel):
    """Response model for visualization impact metrics"""
    total_visualizations: int
    average_impact_score: float
    high_impact_count: int
    board_relevance_score: float
    insight_quality_score: float
    recommendations: List[str]


@router.post("/visualizations/create", response_model=List[VisualizationResponse])
async def create_executive_visualizations(request: VisualizationRequest):
    """Create executive-level data visualizations"""
    try:
        # Create optimized visualizations
        visualizations, impact_metrics = visualization_engine.create_optimized_visualizations(
            data=request.data,
            board_context=request.board_context,
            board_preferences=request.preferences
        )
        
        # Convert to response format
        responses = []
        for viz in visualizations:
            response = VisualizationResponse(
                id=viz.id,
                title=viz.title,
                visualization_type=viz.visualization_type.value,
                chart_type=viz.chart_type.value,
                insights=viz.insights,
                executive_summary=viz.executive_summary,
                impact_score=viz.impact_score,
                board_relevance=viz.board_relevance,
                created_at=viz.created_at.isoformat()
            )
            responses.append(response)
        
        logger.info(f"Created {len(responses)} executive visualizations")
        return responses
        
    except Exception as e:
        logger.error(f"Error creating executive visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create visualizations: {str(e)}")


@router.post("/visualizations/impact", response_model=VisualizationImpactResponse)
async def measure_visualization_impact(request: VisualizationRequest):
    """Measure impact of executive visualizations"""
    try:
        # Create visualizations
        visualizations, impact_metrics = visualization_engine.create_optimized_visualizations(
            data=request.data,
            board_context=request.board_context,
            board_preferences=request.preferences
        )
        
        # Return impact metrics
        response = VisualizationImpactResponse(
            total_visualizations=impact_metrics["total_visualizations"],
            average_impact_score=impact_metrics["average_impact_score"],
            high_impact_count=impact_metrics["high_impact_count"],
            board_relevance_score=impact_metrics["board_relevance_score"],
            insight_quality_score=impact_metrics["insight_quality_score"],
            recommendations=impact_metrics["recommendations"]
        )
        
        logger.info(f"Measured impact for {response.total_visualizations} visualizations")
        return response
        
    except Exception as e:
        logger.error(f"Error measuring visualization impact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to measure impact: {str(e)}")


@router.get("/visualizations/types", response_model=List[Dict[str, str]])
async def get_visualization_types():
    """Get available visualization types"""
    try:
        from ...engines.executive_data_visualization_engine import VisualizationType, ChartType
        
        types = []
        for viz_type in VisualizationType:
            types.append({
                "value": viz_type.value,
                "name": viz_type.value.replace('_', ' ').title(),
                "description": f"Visualization optimized for {viz_type.value.replace('_', ' ')}"
            })
        
        return types
        
    except Exception as e:
        logger.error(f"Error retrieving visualization types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve types: {str(e)}")


@router.get("/visualizations/chart-types", response_model=List[Dict[str, str]])
async def get_chart_types():
    """Get available chart types"""
    try:
        from ...engines.executive_data_visualization_engine import ChartType
        
        types = []
        for chart_type in ChartType:
            types.append({
                "value": chart_type.value,
                "name": chart_type.value.replace('_', ' ').title(),
                "description": f"Chart type optimized for {chart_type.value.replace('_', ' ')}"
            })
        
        return types
        
    except Exception as e:
        logger.error(f"Error retrieving chart types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart types: {str(e)}")


@router.post("/visualizations/optimize", response_model=List[VisualizationResponse])
async def optimize_visualizations(request: VisualizationRequest):
    """Optimize visualizations for board consumption"""
    try:
        # Create initial visualizations
        visualizations, _ = visualization_engine.create_optimized_visualizations(
            data=request.data,
            board_context=request.board_context,
            board_preferences=request.preferences
        )
        
        # Apply additional optimization
        optimized_visualizations = visualization_engine.optimizer.optimize_for_board_consumption(
            visualizations, request.preferences
        )
        
        # Convert to response format
        responses = []
        for viz in optimized_visualizations:
            response = VisualizationResponse(
                id=viz.id,
                title=viz.title,
                visualization_type=viz.visualization_type.value,
                chart_type=viz.chart_type.value,
                insights=viz.insights,
                executive_summary=viz.executive_summary,
                impact_score=viz.impact_score,
                board_relevance=viz.board_relevance,
                created_at=viz.created_at.isoformat()
            )
            responses.append(response)
        
        logger.info(f"Optimized {len(responses)} visualizations for board consumption")
        return responses
        
    except Exception as e:
        logger.error(f"Error optimizing visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize visualizations: {str(e)}")
c
lass QAPreparationRequest(BaseModel):
    """Request model for Q&A preparation"""
    presentation_id: str
    board_id: str
    board_context: Optional[Dict[str, Any]] = None


class AnticipatedQuestionResponse(BaseModel):
    """Response model for anticipated questions"""
    id: str
    question_text: str
    category: str
    difficulty: str
    likelihood: float
    board_member_source: Optional[str]
    context: str
    potential_follow_ups: List[str]


class PreparedResponseResponse(BaseModel):
    """Response model for prepared responses"""
    id: str
    question_id: str
    primary_response: str
    response_strategy: str
    key_messages: List[str]
    potential_challenges: List[str]
    backup_responses: List[str]
    confidence_level: float
    preparation_notes: str


class QAPreparationResponse(BaseModel):
    """Response model for complete Q&A preparation"""
    id: str
    presentation_id: str
    board_id: str
    anticipated_questions: List[AnticipatedQuestionResponse]
    prepared_responses: List[PreparedResponseResponse]
    question_categories_covered: List[str]
    overall_preparedness_score: float
    high_risk_questions: List[str]
    key_talking_points: List[str]
    created_at: str


class QAEffectivenessRequest(BaseModel):
    """Request model for Q&A effectiveness tracking"""
    qa_preparation_id: str
    actual_questions: Optional[List[str]] = None
    board_feedback: Optional[Dict[str, Any]] = None


class QAEffectivenessResponse(BaseModel):
    """Response model for Q&A effectiveness metrics"""
    preparation_completeness: float
    question_prediction_accuracy: float
    response_quality_score: float
    board_satisfaction_score: float
    areas_for_improvement: List[str]
    success_indicators: List[str]


@router.post("/qa-preparation/create", response_model=QAPreparationResponse)
async def create_qa_preparation(request: QAPreparationRequest):
    """Create comprehensive Q&A preparation for board presentation"""
    try:
        # Create mock board and presentation objects
        board = Board(
            id=request.board_id,
            name=f"Board {request.board_id}",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        # Create mock presentation
        mock_content = {
            "overview": "Strategic overview content",
            "financial_performance": {"revenue": 15000000, "growth": 18.5},
            "strategic_initiatives": ["Digital transformation", "Market expansion"],
            "risk_assessment": "Comprehensive risk management approach"
        }
        
        presentation = presentation_framework.designer.create_board_presentation(
            content=mock_content,
            board=board
        )
        
        # Create Q&A preparation
        qa_preparation = qa_preparation_engine.create_comprehensive_qa_preparation(
            presentation=presentation,
            board=board,
            board_context=request.board_context
        )
        
        # Convert to response format
        anticipated_questions = []
        for q in qa_preparation.anticipated_questions:
            anticipated_questions.append(AnticipatedQuestionResponse(
                id=q.id,
                question_text=q.question_text,
                category=q.category.value,
                difficulty=q.difficulty.value,
                likelihood=q.likelihood,
                board_member_source=q.board_member_source,
                context=q.context,
                potential_follow_ups=q.potential_follow_ups
            ))
        
        prepared_responses = []
        for r in qa_preparation.prepared_responses:
            prepared_responses.append(PreparedResponseResponse(
                id=r.id,
                question_id=r.question_id,
                primary_response=r.primary_response,
                response_strategy=r.response_strategy.value,
                key_messages=r.key_messages,
                potential_challenges=r.potential_challenges,
                backup_responses=r.backup_responses,
                confidence_level=r.confidence_level,
                preparation_notes=r.preparation_notes
            ))
        
        response = QAPreparationResponse(
            id=qa_preparation.id,
            presentation_id=qa_preparation.presentation_id,
            board_id=qa_preparation.board_id,
            anticipated_questions=anticipated_questions,
            prepared_responses=prepared_responses,
            question_categories_covered=[cat.value for cat in qa_preparation.question_categories_covered],
            overall_preparedness_score=qa_preparation.overall_preparedness_score,
            high_risk_questions=qa_preparation.high_risk_questions,
            key_talking_points=qa_preparation.key_talking_points,
            created_at=qa_preparation.created_at.isoformat()
        )
        
        logger.info(f"Created Q&A preparation: {qa_preparation.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error creating Q&A preparation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create Q&A preparation: {str(e)}")


@router.post("/qa-preparation/effectiveness", response_model=QAEffectivenessResponse)
async def track_qa_effectiveness(request: QAEffectivenessRequest):
    """Track Q&A preparation effectiveness"""
    try:
        # Create mock Q&A preparation for tracking
        board = Board(
            id="test_board",
            name="Test Board",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        mock_content = {
            "overview": "Strategic overview",
            "performance": {"revenue": 10000000}
        }
        
        presentation = presentation_framework.designer.create_board_presentation(
            content=mock_content,
            board=board
        )
        
        qa_preparation = qa_preparation_engine.create_comprehensive_qa_preparation(
            presentation=presentation,
            board=board
        )
        
        # Track effectiveness
        effectiveness_metrics = qa_preparation_engine.effectiveness_tracker.track_qa_effectiveness(
            qa_preparation=qa_preparation,
            actual_questions=request.actual_questions,
            board_feedback=request.board_feedback
        )
        
        response = QAEffectivenessResponse(
            preparation_completeness=effectiveness_metrics.get("preparation_completeness", 0.0),
            question_prediction_accuracy=effectiveness_metrics.get("question_prediction_accuracy", 0.0),
            response_quality_score=effectiveness_metrics.get("response_quality_score", 0.0),
            board_satisfaction_score=effectiveness_metrics.get("board_satisfaction_score", 0.0),
            areas_for_improvement=effectiveness_metrics.get("areas_for_improvement", []),
            success_indicators=effectiveness_metrics.get("success_indicators", [])
        )
        
        logger.info(f"Tracked Q&A effectiveness for preparation: {request.qa_preparation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error tracking Q&A effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track effectiveness: {str(e)}")


@router.get("/qa-preparation/question-categories", response_model=List[Dict[str, str]])
async def get_question_categories():
    """Get available question categories"""
    try:
        from ...engines.qa_preparation_engine import QuestionCategory
        
        categories = []
        for category in QuestionCategory:
            categories.append({
                "value": category.value,
                "name": category.value.replace('_', ' ').title(),
                "description": f"Questions related to {category.value.replace('_', ' ')}"
            })
        
        return categories
        
    except Exception as e:
        logger.error(f"Error retrieving question categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve categories: {str(e)}")


@router.get("/qa-preparation/response-strategies", response_model=List[Dict[str, str]])
async def get_response_strategies():
    """Get available response strategies"""
    try:
        from ...engines.qa_preparation_engine import ResponseStrategy
        
        strategies = []
        for strategy in ResponseStrategy:
            strategies.append({
                "value": strategy.value,
                "name": strategy.value.replace('_', ' ').title(),
                "description": f"Response approach: {strategy.value.replace('_', ' ')}"
            })
        
        return strategies
        
    except Exception as e:
        logger.error(f"Error retrieving response strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategies: {str(e)}")


@router.post("/qa-preparation/anticipate-questions", response_model=List[AnticipatedQuestionResponse])
async def anticipate_questions_only(request: QAPreparationRequest):
    """Anticipate questions without full preparation"""
    try:
        # Create mock board and presentation
        board = Board(
            id=request.board_id,
            name=f"Board {request.board_id}",
            composition={},
            governance_structure={},
            meeting_patterns={},
            decision_processes={}
        )
        
        mock_content = {
            "overview": "Strategic overview",
            "financial_metrics": {"revenue": 12000000, "growth": 15.5},
            "strategic_initiatives": ["Innovation program", "Market expansion"]
        }
        
        presentation = presentation_framework.designer.create_board_presentation(
            content=mock_content,
            board=board
        )
        
        # Anticipate questions only
        anticipated_questions = qa_preparation_engine.question_anticipator.anticipate_board_questions(
            presentation=presentation,
            board=board,
            board_context=request.board_context
        )
        
        # Convert to response format
        responses = []
        for q in anticipated_questions:
            responses.append(AnticipatedQuestionResponse(
                id=q.id,
                question_text=q.question_text,
                category=q.category.value,
                difficulty=q.difficulty.value,
                likelihood=q.likelihood,
                board_member_source=q.board_member_source,
                context=q.context,
                potential_follow_ups=q.potential_follow_ups
            ))
        
        logger.info(f"Anticipated {len(responses)} questions for board: {request.board_id}")
        return responses
        
    except Exception as e:
        logger.error(f"Error anticipating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to anticipate questions: {str(e)}")