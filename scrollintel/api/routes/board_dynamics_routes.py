"""
API routes for Board Dynamics Analysis System

This module provides REST API endpoints for board dynamics analysis,
including board composition analysis, power structure mapping, meeting dynamics assessment,
and governance framework understanding.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

from scrollintel.models.database import get_db
from scrollintel.models.board_dynamics_models import (
    BoardMemberModel, BoardMeetingModel, BoardAnalysisModel,
    GovernanceFrameworkModel, PowerStructureModel, MeetingDynamicsModel,
    BoardMemberSchema, BoardMeetingSchema, AnalysisRequestSchema,
    AnalysisResponseSchema, CompositionAnalysisSchema, PowerStructureMapSchema,
    DynamicsAssessmentSchema, GovernanceFrameworkSchema, BoardAnalysisResultSchema,
    board_member_to_schema, schema_to_board_member, meeting_to_schema
)
from scrollintel.engines.board_dynamics_engine import (
    BoardDynamicsAnalysisEngine, BoardMember, Background, Priority, Relationship,
    InfluenceLevel, CommunicationStyle, DecisionPattern
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/board-dynamics", tags=["Board Dynamics"])

# Initialize the board dynamics engine
board_dynamics_engine = BoardDynamicsAnalysisEngine()


@router.post("/members", response_model=BoardMemberSchema)
async def create_board_member(
    member: BoardMemberSchema,
    db: Session = Depends(get_db)
):
    """Create a new board member"""
    try:
        logger.info(f"Creating board member: {member.name}")
        
        # Check if member already exists
        existing_member = db.query(BoardMemberModel).filter(
            BoardMemberModel.id == member.id
        ).first()
        
        if existing_member:
            raise HTTPException(status_code=400, detail="Board member already exists")
        
        # Convert schema to database model
        member_model = schema_to_board_member(member)
        
        # Add to database
        db.add(member_model)
        db.commit()
        db.refresh(member_model)
        
        # Add relationships and priorities
        for relationship in member.relationships:
            rel_model = BoardRelationshipModel(
                member_id=member.id,
                related_member_id=relationship.member_id,
                relationship_type=relationship.relationship_type,
                strength=relationship.strength,
                influence_direction=relationship.influence_direction
            )
            db.add(rel_model)
        
        for priority in member.priorities:
            priority_model = BoardMemberPriorityModel(
                member_id=member.id,
                area=priority.area,
                importance=priority.importance,
                description=priority.description,
                timeline=priority.timeline
            )
            db.add(priority_model)
        
        db.commit()
        
        logger.info(f"Board member created successfully: {member.id}")
        return member
        
    except Exception as e:
        logger.error(f"Error creating board member: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/members", response_model=List[BoardMemberSchema])
async def get_board_members(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get all board members"""
    try:
        logger.info("Retrieving board members")
        
        members = db.query(BoardMemberModel).offset(offset).limit(limit).all()
        
        # Convert to schemas
        member_schemas = []
        for member in members:
            member_schema = board_member_to_schema(member)
            member_schemas.append(member_schema)
        
        logger.info(f"Retrieved {len(member_schemas)} board members")
        return member_schemas
        
    except Exception as e:
        logger.error(f"Error retrieving board members: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/members/{member_id}", response_model=BoardMemberSchema)
async def get_board_member(
    member_id: str = Path(..., description="Board member ID"),
    db: Session = Depends(get_db)
):
    """Get a specific board member"""
    try:
        logger.info(f"Retrieving board member: {member_id}")
        
        member = db.query(BoardMemberModel).filter(
            BoardMemberModel.id == member_id
        ).first()
        
        if not member:
            raise HTTPException(status_code=404, detail="Board member not found")
        
        member_schema = board_member_to_schema(member)
        
        logger.info(f"Board member retrieved successfully: {member_id}")
        return member_schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving board member: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/members/{member_id}", response_model=BoardMemberSchema)
async def update_board_member(
    member_id: str = Path(..., description="Board member ID"),
    member_update: BoardMemberSchema = None,
    db: Session = Depends(get_db)
):
    """Update a board member"""
    try:
        logger.info(f"Updating board member: {member_id}")
        
        existing_member = db.query(BoardMemberModel).filter(
            BoardMemberModel.id == member_id
        ).first()
        
        if not existing_member:
            raise HTTPException(status_code=404, detail="Board member not found")
        
        # Update member fields
        if member_update.name:
            existing_member.name = member_update.name
        if member_update.background:
            existing_member.background_data = {
                "industry_experience": member_update.background.industry_experience,
                "functional_expertise": member_update.background.functional_expertise,
                "education": member_update.background.education,
                "previous_roles": member_update.background.previous_roles,
                "years_experience": member_update.background.years_experience
            }
        if member_update.expertise_areas:
            existing_member.expertise_areas = member_update.expertise_areas
        if member_update.influence_level:
            existing_member.influence_level = member_update.influence_level
        if member_update.communication_style:
            existing_member.communication_style = member_update.communication_style
        if member_update.decision_making_pattern:
            existing_member.decision_making_pattern = member_update.decision_making_pattern
        if member_update.tenure is not None:
            existing_member.tenure = member_update.tenure
        if member_update.committee_memberships:
            existing_member.committee_memberships = member_update.committee_memberships
        
        existing_member.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(existing_member)
        
        member_schema = board_member_to_schema(existing_member)
        
        logger.info(f"Board member updated successfully: {member_id}")
        return member_schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating board member: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/members/{member_id}")
async def delete_board_member(
    member_id: str = Path(..., description="Board member ID"),
    db: Session = Depends(get_db)
):
    """Delete a board member"""
    try:
        logger.info(f"Deleting board member: {member_id}")
        
        member = db.query(BoardMemberModel).filter(
            BoardMemberModel.id == member_id
        ).first()
        
        if not member:
            raise HTTPException(status_code=404, detail="Board member not found")
        
        # Delete related records
        db.query(BoardRelationshipModel).filter(
            (BoardRelationshipModel.member_id == member_id) |
            (BoardRelationshipModel.related_member_id == member_id)
        ).delete()
        
        db.query(BoardMemberPriorityModel).filter(
            BoardMemberPriorityModel.member_id == member_id
        ).delete()
        
        # Delete member
        db.delete(member)
        db.commit()
        
        logger.info(f"Board member deleted successfully: {member_id}")
        return {"message": "Board member deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting board member: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meetings", response_model=BoardMeetingSchema)
async def create_board_meeting(
    meeting: BoardMeetingSchema,
    db: Session = Depends(get_db)
):
    """Create a new board meeting record"""
    try:
        logger.info(f"Creating board meeting: {meeting.id}")
        
        # Check if meeting already exists
        existing_meeting = db.query(BoardMeetingModel).filter(
            BoardMeetingModel.id == meeting.id
        ).first()
        
        if existing_meeting:
            raise HTTPException(status_code=400, detail="Board meeting already exists")
        
        # Create meeting model
        meeting_model = BoardMeetingModel(
            id=meeting.id,
            meeting_date=meeting.meeting_date,
            meeting_type=meeting.meeting_type,
            agenda_items=meeting.agenda_items,
            attendance=meeting.attendance,
            meeting_duration=meeting.meeting_duration,
            meeting_data=meeting.meeting_data,
            effectiveness_score=meeting.effectiveness_score
        )
        
        db.add(meeting_model)
        db.commit()
        db.refresh(meeting_model)
        
        # Add participation records
        for participation in meeting.participation_records:
            participation_model = MeetingParticipationModel(
                meeting_id=meeting.id,
                member_id=participation.member_id,
                speaking_time=participation.speaking_time,
                questions_asked=participation.questions_asked,
                contributions=participation.contributions,
                engagement_score=participation.engagement_score,
                participation_data=participation.participation_data
            )
            db.add(participation_model)
        
        db.commit()
        
        logger.info(f"Board meeting created successfully: {meeting.id}")
        return meeting
        
    except Exception as e:
        logger.error(f"Error creating board meeting: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meetings", response_model=List[BoardMeetingSchema])
async def get_board_meetings(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Get all board meetings"""
    try:
        logger.info("Retrieving board meetings")
        
        meetings = db.query(BoardMeetingModel).offset(offset).limit(limit).all()
        
        # Convert to schemas
        meeting_schemas = []
        for meeting in meetings:
            meeting_schema = meeting_to_schema(meeting)
            meeting_schemas.append(meeting_schema)
        
        logger.info(f"Retrieved {len(meeting_schemas)} board meetings")
        return meeting_schemas
        
    except Exception as e:
        logger.error(f"Error retrieving board meetings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/composition", response_model=Dict[str, Any])
async def analyze_board_composition(
    member_ids: List[str],
    db: Session = Depends(get_db)
):
    """Analyze board composition"""
    try:
        logger.info(f"Analyzing board composition for {len(member_ids)} members")
        
        # Retrieve board members
        members = db.query(BoardMemberModel).filter(
            BoardMemberModel.id.in_(member_ids)
        ).all()
        
        if not members:
            raise HTTPException(status_code=404, detail="No board members found")
        
        # Convert to engine format
        engine_members = []
        for member in members:
            background = Background(
                industry_experience=member.background_data.get("industry_experience", []),
                functional_expertise=member.background_data.get("functional_expertise", []),
                education=member.background_data.get("education", []),
                previous_roles=member.background_data.get("previous_roles", []),
                years_experience=member.background_data.get("years_experience", 0)
            )
            
            # Get relationships
            relationships = []
            for rel in member.relationships:
                relationships.append(Relationship(
                    member_id=rel.related_member_id,
                    relationship_type=rel.relationship_type,
                    strength=rel.strength,
                    influence_direction=rel.influence_direction
                ))
            
            # Get priorities
            priorities = []
            for priority in member.priorities:
                priorities.append(Priority(
                    area=priority.area,
                    importance=priority.importance,
                    description=priority.description,
                    timeline=priority.timeline
                ))
            
            engine_member = BoardMember(
                id=member.id,
                name=member.name,
                background=background,
                expertise_areas=member.expertise_areas or [],
                influence_level=InfluenceLevel(member.influence_level),
                communication_style=CommunicationStyle(member.communication_style),
                decision_making_pattern=DecisionPattern(member.decision_making_pattern),
                relationships=relationships,
                priorities=priorities,
                tenure=member.tenure,
                committee_memberships=member.committee_memberships or []
            )
            engine_members.append(engine_member)
        
        # Perform analysis
        analysis_result = board_dynamics_engine.analyze_board_composition(engine_members)
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        analysis_model = BoardAnalysisModel(
            id=analysis_id,
            analysis_type="composition",
            board_members_analyzed=member_ids,
            analysis_results=analysis_result.__dict__,
            insights=[],  # Could be generated from analysis
            recommendations=[],  # Could be generated from analysis
            overall_score=0.8  # Could be calculated from analysis
        )
        
        db.add(analysis_model)
        db.commit()
        
        logger.info(f"Board composition analysis completed: {analysis_id}")
        return {
            "analysis_id": analysis_id,
            "analysis_type": "composition",
            "results": analysis_result.__dict__,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing board composition: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/power-structure", response_model=Dict[str, Any])
async def analyze_power_structure(
    member_ids: List[str],
    db: Session = Depends(get_db)
):
    """Analyze board power structure"""
    try:
        logger.info(f"Analyzing power structure for {len(member_ids)} members")
        
        # Retrieve board members (similar to composition analysis)
        members = db.query(BoardMemberModel).filter(
            BoardMemberModel.id.in_(member_ids)
        ).all()
        
        if not members:
            raise HTTPException(status_code=404, detail="No board members found")
        
        # Convert to engine format (reuse logic from composition analysis)
        engine_members = []
        for member in members:
            background = Background(
                industry_experience=member.background_data.get("industry_experience", []),
                functional_expertise=member.background_data.get("functional_expertise", []),
                education=member.background_data.get("education", []),
                previous_roles=member.background_data.get("previous_roles", []),
                years_experience=member.background_data.get("years_experience", 0)
            )
            
            relationships = []
            for rel in member.relationships:
                relationships.append(Relationship(
                    member_id=rel.related_member_id,
                    relationship_type=rel.relationship_type,
                    strength=rel.strength,
                    influence_direction=rel.influence_direction
                ))
            
            priorities = []
            for priority in member.priorities:
                priorities.append(Priority(
                    area=priority.area,
                    importance=priority.importance,
                    description=priority.description,
                    timeline=priority.timeline
                ))
            
            engine_member = BoardMember(
                id=member.id,
                name=member.name,
                background=background,
                expertise_areas=member.expertise_areas or [],
                influence_level=InfluenceLevel(member.influence_level),
                communication_style=CommunicationStyle(member.communication_style),
                decision_making_pattern=DecisionPattern(member.decision_making_pattern),
                relationships=relationships,
                priorities=priorities,
                tenure=member.tenure,
                committee_memberships=member.committee_memberships or []
            )
            engine_members.append(engine_member)
        
        # Perform power structure analysis
        power_structure = board_dynamics_engine.map_power_structures(engine_members)
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        power_structure_model = PowerStructureModel(
            id=analysis_id,
            influence_networks=power_structure.influence_networks,
            decision_makers=power_structure.decision_makers,
            coalition_groups=power_structure.coalition_groups,
            influence_flows=power_structure.influence_flows,
            key_relationships=[rel.__dict__ for rel in power_structure.key_relationships],
            power_distribution_score=0.7  # Could be calculated
        )
        
        db.add(power_structure_model)
        db.commit()
        
        logger.info(f"Power structure analysis completed: {analysis_id}")
        return {
            "analysis_id": analysis_id,
            "analysis_type": "power_structure",
            "results": power_structure.__dict__,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing power structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/meeting-dynamics", response_model=Dict[str, Any])
async def analyze_meeting_dynamics(
    meeting_id: str,
    member_ids: List[str],
    db: Session = Depends(get_db)
):
    """Analyze meeting dynamics"""
    try:
        logger.info(f"Analyzing meeting dynamics for meeting: {meeting_id}")
        
        # Retrieve meeting
        meeting = db.query(BoardMeetingModel).filter(
            BoardMeetingModel.id == meeting_id
        ).first()
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        # Retrieve board members
        members = db.query(BoardMemberModel).filter(
            BoardMemberModel.id.in_(member_ids)
        ).all()
        
        if not members:
            raise HTTPException(status_code=404, detail="No board members found")
        
        # Convert members to engine format
        engine_members = []
        for member in members:
            background = Background(
                industry_experience=member.background_data.get("industry_experience", []),
                functional_expertise=member.background_data.get("functional_expertise", []),
                education=member.background_data.get("education", []),
                previous_roles=member.background_data.get("previous_roles", []),
                years_experience=member.background_data.get("years_experience", 0)
            )
            
            engine_member = BoardMember(
                id=member.id,
                name=member.name,
                background=background,
                expertise_areas=member.expertise_areas or [],
                influence_level=InfluenceLevel(member.influence_level),
                communication_style=CommunicationStyle(member.communication_style),
                decision_making_pattern=DecisionPattern(member.decision_making_pattern),
                tenure=member.tenure,
                committee_memberships=member.committee_memberships or []
            )
            engine_members.append(engine_member)
        
        # Perform meeting dynamics analysis
        dynamics_assessment = board_dynamics_engine.assess_meeting_dynamics(
            meeting.meeting_data, engine_members
        )
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        dynamics_model = MeetingDynamicsModel(
            id=analysis_id,
            meeting_id=meeting_id,
            participation_patterns=dynamics_assessment.engagement_levels,
            communication_patterns=dynamics_assessment.communication_patterns,
            engagement_levels=dynamics_assessment.engagement_levels,
            conflict_indicators=dynamics_assessment.conflict_indicators,
            collaboration_quality=dynamics_assessment.collaboration_quality,
            meeting_effectiveness=dynamics_assessment.meeting_effectiveness,
            decision_efficiency=dynamics_assessment.decision_efficiency
        )
        
        db.add(dynamics_model)
        db.commit()
        
        logger.info(f"Meeting dynamics analysis completed: {analysis_id}")
        return {
            "analysis_id": analysis_id,
            "analysis_type": "meeting_dynamics",
            "results": dynamics_assessment.__dict__,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing meeting dynamics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/governance", response_model=Dict[str, Any])
async def analyze_governance_framework(
    board_info: Dict[str, Any],
    performance_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Analyze governance framework"""
    try:
        logger.info("Analyzing governance framework")
        
        if performance_data is None:
            performance_data = {}
        
        # Perform governance analysis
        governance_analysis = board_dynamics_engine.analyze_governance_framework(
            board_info, performance_data
        )
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        governance_model = GovernanceFrameworkModel(
            id=analysis_id,
            framework_name="current_governance",
            board_structure=governance_analysis["framework"].board_structure,
            committee_structure=governance_analysis["framework"].committee_structure,
            decision_processes=governance_analysis["framework"].decision_processes,
            reporting_requirements=governance_analysis["framework"].reporting_requirements,
            compliance_frameworks=governance_analysis["framework"].compliance_frameworks,
            effectiveness_scores=governance_analysis["effectiveness_scores"],
            governance_gaps=governance_analysis["governance_gaps"],
            last_review_date=datetime.utcnow()
        )
        
        db.add(governance_model)
        db.commit()
        
        logger.info(f"Governance framework analysis completed: {analysis_id}")
        return {
            "analysis_id": analysis_id,
            "analysis_type": "governance",
            "results": governance_analysis,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing governance framework: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/comprehensive", response_model=Dict[str, Any])
async def comprehensive_board_analysis(
    member_ids: List[str],
    meeting_ids: List[str],
    board_info: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Perform comprehensive board dynamics analysis"""
    try:
        logger.info("Performing comprehensive board dynamics analysis")
        
        # Retrieve board members
        members = db.query(BoardMemberModel).filter(
            BoardMemberModel.id.in_(member_ids)
        ).all()
        
        if not members:
            raise HTTPException(status_code=404, detail="No board members found")
        
        # Retrieve meetings
        meetings = db.query(BoardMeetingModel).filter(
            BoardMeetingModel.id.in_(meeting_ids)
        ).all()
        
        # Convert members to engine format
        engine_members = []
        for member in members:
            background = Background(
                industry_experience=member.background_data.get("industry_experience", []),
                functional_expertise=member.background_data.get("functional_expertise", []),
                education=member.background_data.get("education", []),
                previous_roles=member.background_data.get("previous_roles", []),
                years_experience=member.background_data.get("years_experience", 0)
            )
            
            relationships = []
            for rel in member.relationships:
                relationships.append(Relationship(
                    member_id=rel.related_member_id,
                    relationship_type=rel.relationship_type,
                    strength=rel.strength,
                    influence_direction=rel.influence_direction
                ))
            
            priorities = []
            for priority in member.priorities:
                priorities.append(Priority(
                    area=priority.area,
                    importance=priority.importance,
                    description=priority.description,
                    timeline=priority.timeline
                ))
            
            engine_member = BoardMember(
                id=member.id,
                name=member.name,
                background=background,
                expertise_areas=member.expertise_areas or [],
                influence_level=InfluenceLevel(member.influence_level),
                communication_style=CommunicationStyle(member.communication_style),
                decision_making_pattern=DecisionPattern(member.decision_making_pattern),
                relationships=relationships,
                priorities=priorities,
                tenure=member.tenure,
                committee_memberships=member.committee_memberships or []
            )
            engine_members.append(engine_member)
        
        # Aggregate meeting data
        aggregated_meeting_data = {}
        if meetings:
            # Combine meeting data from all meetings
            all_participation = {}
            total_effectiveness = 0
            
            for meeting in meetings:
                meeting_data = meeting.meeting_data or {}
                
                # Aggregate participation data
                participation = meeting_data.get("participation", {})
                for member_id, data in participation.items():
                    if member_id not in all_participation:
                        all_participation[member_id] = {
                            "speaking_time": 0,
                            "questions_asked": 0,
                            "contributions": 0
                        }
                    
                    all_participation[member_id]["speaking_time"] += data.get("speaking_time", 0)
                    all_participation[member_id]["questions_asked"] += data.get("questions_asked", 0)
                    all_participation[member_id]["contributions"] += data.get("contributions", 0)
                
                total_effectiveness += meeting.effectiveness_score or 0.7
            
            aggregated_meeting_data = {
                "participation": all_participation,
                "average_effectiveness": total_effectiveness / len(meetings),
                "meeting_count": len(meetings)
            }
        
        # Perform comprehensive analysis
        comprehensive_analysis = board_dynamics_engine.generate_comprehensive_analysis(
            engine_members, aggregated_meeting_data, board_info
        )
        
        # Store comprehensive analysis result
        analysis_id = str(uuid.uuid4())
        analysis_model = BoardAnalysisModel(
            id=analysis_id,
            analysis_type="comprehensive",
            board_members_analyzed=member_ids,
            analysis_results=comprehensive_analysis,
            insights=comprehensive_analysis.get("insights", []),
            recommendations=comprehensive_analysis.get("recommendations", []),
            overall_score=0.8  # Could be calculated from all analyses
        )
        
        db.add(analysis_model)
        db.commit()
        
        logger.info(f"Comprehensive board analysis completed: {analysis_id}")
        return {
            "analysis_id": analysis_id,
            "analysis_type": "comprehensive",
            "results": comprehensive_analysis,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing comprehensive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses", response_model=List[AnalysisResponseSchema])
async def get_analyses(
    db: Session = Depends(get_db),
    analysis_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Get board analysis results"""
    try:
        logger.info("Retrieving board analyses")
        
        query = db.query(BoardAnalysisModel)
        
        if analysis_type:
            query = query.filter(BoardAnalysisModel.analysis_type == analysis_type)
        
        analyses = query.offset(offset).limit(limit).all()
        
        # Convert to response schemas
        response_schemas = []
        for analysis in analyses:
            response_schema = AnalysisResponseSchema(
                analysis_id=analysis.id,
                analysis_type=analysis.analysis_type,
                status="completed",
                results=analysis.analysis_results,
                insights=analysis.insights,
                recommendations=analysis.recommendations,
                created_at=analysis.created_at
            )
            response_schemas.append(response_schema)
        
        logger.info(f"Retrieved {len(response_schemas)} analyses")
        return response_schemas
        
    except Exception as e:
        logger.error(f"Error retrieving analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponseSchema)
async def get_analysis(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: Session = Depends(get_db)
):
    """Get a specific board analysis result"""
    try:
        logger.info(f"Retrieving analysis: {analysis_id}")
        
        analysis = db.query(BoardAnalysisModel).filter(
            BoardAnalysisModel.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        response_schema = AnalysisResponseSchema(
            analysis_id=analysis.id,
            analysis_type=analysis.analysis_type,
            status="completed",
            results=analysis.analysis_results,
            insights=analysis.insights,
            recommendations=analysis.recommendations,
            created_at=analysis.created_at
        )
        
        logger.info(f"Analysis retrieved successfully: {analysis_id}")
        return response_schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "board_dynamics_analysis",
        "timestamp": datetime.utcnow().isoformat()
    }