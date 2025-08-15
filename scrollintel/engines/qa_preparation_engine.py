"""
Q&A Preparation Engine for Board Presentations

This engine creates board question anticipation and preparation framework
with comprehensive response development and effectiveness tracking.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ..models.board_presentation_models import BoardPresentation
from ..models.board_dynamics_models import Board


class QuestionCategory(Enum):
    """Categories of board questions"""
    FINANCIAL_PERFORMANCE = "financial_performance"
    STRATEGIC_DIRECTION = "strategic_direction"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    RISK_MANAGEMENT = "risk_management"
    COMPETITIVE_POSITION = "competitive_position"
    GOVERNANCE_COMPLIANCE = "governance_compliance"
    MARKET_CONDITIONS = "market_conditions"
    TECHNOLOGY_INNOVATION = "technology_innovation"
    TALENT_MANAGEMENT = "talent_management"
    REGULATORY_ENVIRONMENT = "regulatory_environment"


class QuestionDifficulty(Enum):
    """Difficulty levels of anticipated questions"""
    STRAIGHTFORWARD = "straightforward"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    CRITICAL = "critical"


class ResponseStrategy(Enum):
    """Response strategies for different question types"""
    DIRECT_ANSWER = "direct_answer"
    DATA_DRIVEN = "data_driven"
    STRATEGIC_CONTEXT = "strategic_context"
    ACKNOWLEDGE_AND_PLAN = "acknowledge_and_plan"
    REDIRECT_TO_STRENGTH = "redirect_to_strength"
    COLLABORATIVE_DISCUSSION = "collaborative_discussion"


@dataclass
class AnticipatedQuestion:
    """Anticipated board question with metadata"""
    id: str
    question_text: str
    category: QuestionCategory
    difficulty: QuestionDifficulty
    likelihood: float  # 0.0 to 1.0
    board_member_source: Optional[str]
    context: str
    potential_follow_ups: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PreparedResponse:
    """Prepared response to anticipated question"""
    id: str
    question_id: str
    primary_response: str
    supporting_data: Dict[str, Any]
    response_strategy: ResponseStrategy
    key_messages: List[str]
    potential_challenges: List[str]
    backup_responses: List[str]
    confidence_level: float  # 0.0 to 1.0
    preparation_notes: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QAPreparation:
    """Complete Q&A preparation package"""
    id: str
    presentation_id: str
    board_id: str
    anticipated_questions: List[AnticipatedQuestion]
    prepared_responses: List[PreparedResponse]
    question_categories_covered: List[QuestionCategory]
    overall_preparedness_score: float
    high_risk_questions: List[str]
    key_talking_points: List[str]
    effectiveness_metrics: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class BoardQuestionAnticipator:
    """Anticipates potential board questions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.question_patterns = self._load_question_patterns()
        self.board_member_profiles = self._load_board_member_profiles()
    
    def anticipate_board_questions(
        self, 
        presentation: BoardPresentation, 
        board: Board,
        board_context: Optional[Dict[str, Any]] = None
    ) -> List[AnticipatedQuestion]:
        """Create board question anticipation framework"""
        try:
            anticipated_questions = []
            
            # Analyze presentation content for question triggers
            content_questions = self._analyze_presentation_content(presentation)
            anticipated_questions.extend(content_questions)
            
            # Generate board member specific questions
            member_questions = self._generate_member_specific_questions(board, presentation)
            anticipated_questions.extend(member_questions)
            
            # Add contextual questions based on board context
            if board_context:
                contextual_questions = self._generate_contextual_questions(
                    board_context, presentation
                )
                anticipated_questions.extend(contextual_questions)
            
            # Add standard governance questions
            governance_questions = self._generate_governance_questions(presentation)
            anticipated_questions.extend(governance_questions)
            
            # Prioritize and filter questions
            prioritized_questions = self._prioritize_questions(anticipated_questions, board)
            
            self.logger.info(f"Anticipated {len(prioritized_questions)} board questions")
            return prioritized_questions
            
        except Exception as e:
            self.logger.error(f"Error anticipating board questions: {str(e)}")
            raise
    
    def _analyze_presentation_content(self, presentation: BoardPresentation) -> List[AnticipatedQuestion]:
        """Analyze presentation content to identify potential question areas"""
        questions = []
        
        # Analyze slides for question triggers
        for slide in presentation.slides:
            slide_questions = self._extract_questions_from_slide(slide, presentation)
            questions.extend(slide_questions)
        
        # Analyze key messages for potential challenges
        for i, message in enumerate(presentation.key_messages):
            message_questions = self._generate_questions_from_message(message, i)
            questions.extend(message_questions)
        
        # Analyze executive summary for gaps
        summary_questions = self._identify_summary_gaps(presentation.executive_summary)
        questions.extend(summary_questions)
        
        return questions
    
    def _extract_questions_from_slide(self, slide, presentation: BoardPresentation) -> List[AnticipatedQuestion]:
        """Extract potential questions from slide content"""
        questions = []
        
        # Look for financial data that might trigger questions
        for section in slide.sections:
            if section.content_type in ["metric", "chart", "table"]:
                if section.importance_level == "critical":
                    question = self._create_financial_question(section, slide.title)
                    if question:
                        questions.append(question)
        
        # Look for strategic claims that need justification
        strategic_keywords = ["strategic", "growth", "expansion", "investment"]
        if any(keyword in slide.title.lower() for keyword in strategic_keywords):
            strategic_question = self._create_strategic_question(slide.title)
            if strategic_question:
                questions.append(strategic_question)
        
        return questions
    
    def _create_financial_question(self, section, slide_title: str) -> Optional[AnticipatedQuestion]:
        """Create financial-related question"""
        if "revenue" in str(section.content).lower():
            return AnticipatedQuestion(
                id=f"fin_q_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text=f"What are the key drivers behind the revenue performance shown in {slide_title}?",
                category=QuestionCategory.FINANCIAL_PERFORMANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.8,
                board_member_source="financial_expert",
                context=f"Based on financial data in {slide_title}",
                potential_follow_ups=[
                    "How sustainable are these revenue trends?",
                    "What risks could impact future revenue performance?"
                ]
            )
        return None
    
    def _create_strategic_question(self, slide_title: str) -> Optional[AnticipatedQuestion]:
        """Create strategy-related question"""
        return AnticipatedQuestion(
            id=f"strat_q_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            question_text=f"How does the {slide_title.lower()} align with our long-term strategic objectives?",
            category=QuestionCategory.STRATEGIC_DIRECTION,
            difficulty=QuestionDifficulty.MODERATE,
            likelihood=0.7,
            board_member_source="independent_director",
            context=f"Strategic alignment question from {slide_title}",
            potential_follow_ups=[
                "What are the key success metrics for this strategy?",
                "How do we measure progress against strategic goals?"
            ]
        )
    
    def _generate_questions_from_message(self, message: str, index: int) -> List[AnticipatedQuestion]:
        """Generate questions from key messages"""
        questions = []
        
        # Look for claims that need substantiation
        if "increase" in message.lower() or "improve" in message.lower():
            question = AnticipatedQuestion(
                id=f"msg_q_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text=f"What specific evidence supports the claim that '{message}'?",
                category=QuestionCategory.OPERATIONAL_EFFICIENCY,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.6,
                board_member_source="independent_director",
                context=f"Substantiation needed for key message {index + 1}",
                potential_follow_ups=[
                    "What are the underlying assumptions?",
                    "How do we track progress on this?"
                ]
            )
            questions.append(question)
        
        return questions
    
    def _identify_summary_gaps(self, executive_summary: str) -> List[AnticipatedQuestion]:
        """Identify potential gaps in executive summary"""
        questions = []
        
        # Check for missing risk discussion
        if "risk" not in executive_summary.lower():
            risk_question = AnticipatedQuestion(
                id=f"gap_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text="What are the key risks we should be aware of that aren't covered in this summary?",
                category=QuestionCategory.RISK_MANAGEMENT,
                difficulty=QuestionDifficulty.CHALLENGING,
                likelihood=0.9,
                board_member_source="risk_committee_chair",
                context="Risk discussion gap in executive summary",
                potential_follow_ups=[
                    "How are we mitigating these risks?",
                    "What's our risk appetite for these areas?"
                ]
            )
            questions.append(risk_question)
        
        # Check for missing competitive discussion
        if "competitive" not in executive_summary.lower() and "competition" not in executive_summary.lower():
            competitive_question = AnticipatedQuestion(
                id=f"gap_comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text="How does our performance compare to key competitors?",
                category=QuestionCategory.COMPETITIVE_POSITION,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.7,
                board_member_source="industry_expert",
                context="Competitive analysis gap in summary",
                potential_follow_ups=[
                    "What's our competitive advantage?",
                    "How are competitors responding to our strategy?"
                ]
            )
            questions.append(competitive_question)
        
        return questions
    
    def _generate_member_specific_questions(
        self, 
        board: Board, 
        presentation: BoardPresentation
    ) -> List[AnticipatedQuestion]:
        """Generate questions specific to board member backgrounds"""
        questions = []
        
        # Simulate different board member types and their likely questions
        member_types = [
            ("financial_expert", QuestionCategory.FINANCIAL_PERFORMANCE),
            ("technology_expert", QuestionCategory.TECHNOLOGY_INNOVATION),
            ("industry_veteran", QuestionCategory.COMPETITIVE_POSITION),
            ("independent_director", QuestionCategory.GOVERNANCE_COMPLIANCE),
            ("investor_representative", QuestionCategory.STRATEGIC_DIRECTION)
        ]
        
        for member_type, category in member_types:
            type_questions = self._generate_questions_for_member_type(
                member_type, category, presentation
            )
            questions.extend(type_questions)
        
        return questions
    
    def _generate_questions_for_member_type(
        self, 
        member_type: str, 
        category: QuestionCategory, 
        presentation: BoardPresentation
    ) -> List[AnticipatedQuestion]:
        """Generate questions for specific member type"""
        questions = []
        
        question_templates = {
            "financial_expert": [
                "What are the key financial assumptions underlying these projections?",
                "How do our margins compare to industry benchmarks?",
                "What's driving the variance in our financial performance?"
            ],
            "technology_expert": [
                "How is our technology stack positioned for future scalability?",
                "What's our approach to emerging technology trends?",
                "How do we ensure our technical capabilities remain competitive?"
            ],
            "industry_veteran": [
                "How do these results compare to industry standards?",
                "What industry trends are most likely to impact our business?",
                "How are we positioned relative to industry disruption?"
            ],
            "independent_director": [
                "How does this align with our governance framework?",
                "What oversight mechanisms are in place?",
                "How do we ensure accountability for these initiatives?"
            ],
            "investor_representative": [
                "What's the expected return on this strategic investment?",
                "How does this create shareholder value?",
                "What are the key value creation milestones?"
            ]
        }
        
        templates = question_templates.get(member_type, [])
        
        for i, template in enumerate(templates):
            question = AnticipatedQuestion(
                id=f"{member_type}_q_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text=template,
                category=category,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.6,
                board_member_source=member_type,
                context=f"Typical question from {member_type}",
                potential_follow_ups=self._generate_follow_ups(template)
            )
            questions.append(question)
        
        return questions[:2]  # Limit to 2 questions per member type
    
    def _generate_contextual_questions(
        self, 
        board_context: Dict[str, Any], 
        presentation: BoardPresentation
    ) -> List[AnticipatedQuestion]:
        """Generate questions based on board context"""
        questions = []
        
        # Questions based on meeting type
        meeting_type = board_context.get("meeting_type", "")
        if "quarterly" in meeting_type.lower():
            quarterly_question = AnticipatedQuestion(
                id=f"ctx_quarterly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text="How do these quarterly results position us for the full year?",
                category=QuestionCategory.FINANCIAL_PERFORMANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.8,
                board_member_source="board_chair",
                context="Quarterly meeting context",
                potential_follow_ups=[
                    "What adjustments do we need to make for the remaining quarters?",
                    "Are we on track to meet annual guidance?"
                ]
            )
            questions.append(quarterly_question)
        
        # Questions based on focus areas
        focus_areas = board_context.get("focus_areas", [])
        for area in focus_areas:
            if area == "risk_management":
                risk_question = AnticipatedQuestion(
                    id=f"ctx_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    question_text="What are our top three risk priorities and mitigation strategies?",
                    category=QuestionCategory.RISK_MANAGEMENT,
                    difficulty=QuestionDifficulty.CHALLENGING,
                    likelihood=0.9,
                    board_member_source="risk_committee",
                    context="Risk management focus area",
                    potential_follow_ups=[
                        "How do we monitor these risks?",
                        "What's our risk tolerance for each area?"
                    ]
                )
                questions.append(risk_question)
        
        return questions
    
    def _generate_governance_questions(self, presentation: BoardPresentation) -> List[AnticipatedQuestion]:
        """Generate standard governance questions"""
        questions = []
        
        # Standard governance questions that often arise
        governance_templates = [
            "How do we ensure proper oversight of these initiatives?",
            "What reporting mechanisms are in place for board monitoring?",
            "How do these decisions align with our fiduciary responsibilities?",
            "What stakeholder considerations have been evaluated?"
        ]
        
        for i, template in enumerate(governance_templates):
            question = AnticipatedQuestion(
                id=f"gov_q_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                question_text=template,
                category=QuestionCategory.GOVERNANCE_COMPLIANCE,
                difficulty=QuestionDifficulty.MODERATE,
                likelihood=0.5,
                board_member_source="governance_committee",
                context="Standard governance inquiry",
                potential_follow_ups=self._generate_follow_ups(template)
            )
            questions.append(question)
        
        return questions[:2]  # Limit to most important governance questions
    
    def _prioritize_questions(
        self, 
        questions: List[AnticipatedQuestion], 
        board: Board
    ) -> List[AnticipatedQuestion]:
        """Prioritize questions by likelihood and importance"""
        # Sort by likelihood and difficulty
        prioritized = sorted(
            questions, 
            key=lambda q: (q.likelihood, q.difficulty.value == "critical"), 
            reverse=True
        )
        
        # Limit to manageable number for preparation
        return prioritized[:15]  # Top 15 most likely questions
    
    def _generate_follow_ups(self, question_text: str) -> List[str]:
        """Generate potential follow-up questions"""
        follow_ups = []
        
        if "financial" in question_text.lower() or "revenue" in question_text.lower():
            follow_ups.extend([
                "What are the key assumptions behind these numbers?",
                "How sensitive are these results to market changes?"
            ])
        
        if "strategic" in question_text.lower() or "strategy" in question_text.lower():
            follow_ups.extend([
                "What are the key success metrics?",
                "How do we measure progress?"
            ])
        
        if "risk" in question_text.lower():
            follow_ups.extend([
                "What's our mitigation strategy?",
                "How do we monitor this risk?"
            ])
        
        return follow_ups[:2]  # Limit to 2 follow-ups
    
    def _load_question_patterns(self) -> Dict[str, Any]:
        """Load question patterns and templates"""
        return {
            "financial_patterns": [
                "revenue_drivers", "margin_analysis", "cash_flow", "profitability"
            ],
            "strategic_patterns": [
                "market_position", "competitive_advantage", "growth_strategy", "value_creation"
            ],
            "operational_patterns": [
                "efficiency_metrics", "process_improvement", "scalability", "execution"
            ],
            "governance_patterns": [
                "oversight", "compliance", "accountability", "stakeholder_interests"
            ]
        }
    
    def _load_board_member_profiles(self) -> Dict[str, Any]:
        """Load board member profiles and question tendencies"""
        return {
            "financial_expert": {
                "focus_areas": ["financial_performance", "risk_management"],
                "question_style": "analytical",
                "detail_preference": "high"
            },
            "independent_director": {
                "focus_areas": ["governance_compliance", "strategic_direction"],
                "question_style": "oversight",
                "detail_preference": "medium"
            },
            "industry_expert": {
                "focus_areas": ["competitive_position", "market_conditions"],
                "question_style": "contextual",
                "detail_preference": "high"
            }
        }


class ResponseDeveloper:
    """Develops comprehensive responses to anticipated questions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_frameworks = self._load_response_frameworks()
    
    def develop_comprehensive_responses(
        self, 
        questions: List[AnticipatedQuestion],
        presentation: BoardPresentation,
        board_context: Optional[Dict[str, Any]] = None
    ) -> List[PreparedResponse]:
        """Build comprehensive response development and optimization"""
        try:
            responses = []
            
            for question in questions:
                response = self._develop_individual_response(
                    question, presentation, board_context
                )
                if response:
                    responses.append(response)
            
            # Optimize responses for consistency and effectiveness
            optimized_responses = self._optimize_responses(responses)
            
            self.logger.info(f"Developed {len(optimized_responses)} comprehensive responses")
            return optimized_responses
            
        except Exception as e:
            self.logger.error(f"Error developing responses: {str(e)}")
            raise
    
    def _develop_individual_response(
        self,
        question: AnticipatedQuestion,
        presentation: BoardPresentation,
        board_context: Optional[Dict[str, Any]]
    ) -> Optional[PreparedResponse]:
        """Develop response for individual question"""
        try:
            # Determine response strategy
            strategy = self._determine_response_strategy(question)
            
            # Create primary response
            primary_response = self._create_primary_response(question, presentation, strategy)
            
            # Gather supporting data
            supporting_data = self._gather_supporting_data(question, presentation)
            
            # Extract key messages
            key_messages = self._extract_key_messages(question, primary_response)
            
            # Identify potential challenges
            potential_challenges = self._identify_potential_challenges(question)
            
            # Create backup responses
            backup_responses = self._create_backup_responses(question, primary_response)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(question, supporting_data)
            
            # Generate preparation notes
            preparation_notes = self._generate_preparation_notes(question, strategy)
            
            response = PreparedResponse(
                id=f"resp_{question.id}",
                question_id=question.id,
                primary_response=primary_response,
                supporting_data=supporting_data,
                response_strategy=strategy,
                key_messages=key_messages,
                potential_challenges=potential_challenges,
                backup_responses=backup_responses,
                confidence_level=confidence_level,
                preparation_notes=preparation_notes
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error developing individual response: {str(e)}")
            return None
    
    def _determine_response_strategy(self, question: AnticipatedQuestion) -> ResponseStrategy:
        """Determine optimal response strategy"""
        if question.category == QuestionCategory.FINANCIAL_PERFORMANCE:
            return ResponseStrategy.DATA_DRIVEN
        elif question.category == QuestionCategory.STRATEGIC_DIRECTION:
            return ResponseStrategy.STRATEGIC_CONTEXT
        elif question.category == QuestionCategory.RISK_MANAGEMENT:
            return ResponseStrategy.ACKNOWLEDGE_AND_PLAN
        elif question.difficulty == QuestionDifficulty.CHALLENGING:
            return ResponseStrategy.COLLABORATIVE_DISCUSSION
        else:
            return ResponseStrategy.DIRECT_ANSWER
    
    def _create_primary_response(
        self,
        question: AnticipatedQuestion,
        presentation: BoardPresentation,
        strategy: ResponseStrategy
    ) -> str:
        """Create primary response based on strategy"""
        response_templates = {
            ResponseStrategy.DATA_DRIVEN: self._create_data_driven_response,
            ResponseStrategy.STRATEGIC_CONTEXT: self._create_strategic_response,
            ResponseStrategy.ACKNOWLEDGE_AND_PLAN: self._create_acknowledge_response,
            ResponseStrategy.DIRECT_ANSWER: self._create_direct_response,
            ResponseStrategy.COLLABORATIVE_DISCUSSION: self._create_collaborative_response
        }
        
        response_creator = response_templates.get(strategy, self._create_direct_response)
        return response_creator(question, presentation)
    
    def _create_data_driven_response(
        self, 
        question: AnticipatedQuestion, 
        presentation: BoardPresentation
    ) -> str:
        """Create data-driven response"""
        response_parts = []
        
        # Start with data context
        response_parts.append("Based on our current data and analysis:")
        
        # Add specific metrics if available
        if "revenue" in question.question_text.lower():
            response_parts.append("Our revenue performance shows strong fundamentals with key drivers including market expansion and product innovation.")
        elif "performance" in question.question_text.lower():
            response_parts.append("Performance metrics indicate we're meeting or exceeding our strategic objectives across key areas.")
        
        # Add forward-looking perspective
        response_parts.append("Looking ahead, we expect these trends to continue based on our strategic initiatives and market positioning.")
        
        return " ".join(response_parts)
    
    def _create_strategic_response(
        self, 
        question: AnticipatedQuestion, 
        presentation: BoardPresentation
    ) -> str:
        """Create strategic context response"""
        response_parts = []
        
        # Start with strategic alignment
        response_parts.append("This aligns directly with our strategic framework:")
        
        # Add strategic context
        if "strategic" in question.question_text.lower():
            response_parts.append("Our strategic approach focuses on sustainable growth, market leadership, and stakeholder value creation.")
        
        # Add execution perspective
        response_parts.append("We're executing this through disciplined resource allocation and continuous performance monitoring.")
        
        return " ".join(response_parts)
    
    def _create_acknowledge_response(
        self, 
        question: AnticipatedQuestion, 
        presentation: BoardPresentation
    ) -> str:
        """Create acknowledge and plan response"""
        response_parts = []
        
        # Acknowledge the concern
        response_parts.append("That's an important consideration that we take seriously.")
        
        # Present the plan
        if "risk" in question.question_text.lower():
            response_parts.append("We have comprehensive risk management processes in place, including regular monitoring, mitigation strategies, and contingency planning.")
        
        # Show ongoing commitment
        response_parts.append("We continue to evaluate and enhance our approach based on evolving conditions and best practices.")
        
        return " ".join(response_parts)
    
    def _create_direct_response(
        self, 
        question: AnticipatedQuestion, 
        presentation: BoardPresentation
    ) -> str:
        """Create direct response"""
        # Provide straightforward answer based on question category
        if question.category == QuestionCategory.FINANCIAL_PERFORMANCE:
            return "Our financial performance reflects strong execution of our business strategy with solid fundamentals across key metrics."
        elif question.category == QuestionCategory.OPERATIONAL_EFFICIENCY:
            return "We've implemented systematic improvements that are delivering measurable results in operational efficiency and effectiveness."
        else:
            return "We have a comprehensive approach that addresses the key aspects of this question with clear metrics and accountability."
    
    def _create_collaborative_response(
        self, 
        question: AnticipatedQuestion, 
        presentation: BoardPresentation
    ) -> str:
        """Create collaborative discussion response"""
        response_parts = []
        
        # Invite collaboration
        response_parts.append("This is an excellent question that benefits from board input and perspective.")
        
        # Present current thinking
        response_parts.append("Our current approach considers multiple factors and scenarios.")
        
        # Seek input
        response_parts.append("I'd welcome the board's thoughts on how we can further strengthen our position in this area.")
        
        return " ".join(response_parts)
    
    def _gather_supporting_data(
        self, 
        question: AnticipatedQuestion, 
        presentation: BoardPresentation
    ) -> Dict[str, Any]:
        """Gather supporting data for response"""
        supporting_data = {}
        
        # Extract relevant data from presentation
        for slide in presentation.slides:
            for section in slide.sections:
                if self._is_relevant_to_question(section, question):
                    supporting_data[f"{slide.title}_{section.title}"] = section.content
        
        # Add contextual data based on question category
        if question.category == QuestionCategory.FINANCIAL_PERFORMANCE:
            supporting_data["financial_context"] = "Strong fundamentals with growth trajectory"
        elif question.category == QuestionCategory.STRATEGIC_DIRECTION:
            supporting_data["strategic_context"] = "Aligned with long-term value creation"
        
        return supporting_data
    
    def _is_relevant_to_question(self, section, question: AnticipatedQuestion) -> bool:
        """Check if section is relevant to question"""
        question_keywords = question.question_text.lower().split()
        section_text = f"{section.title} {str(section.content)}".lower()
        
        # Check for keyword overlap
        overlap = sum(1 for keyword in question_keywords if keyword in section_text)
        return overlap >= 2  # At least 2 keyword matches
    
    def _extract_key_messages(self, question: AnticipatedQuestion, response: str) -> List[str]:
        """Extract key messages from response"""
        key_messages = []
        
        # Extract main points from response
        sentences = response.split('.')
        for sentence in sentences[:3]:  # Top 3 sentences
            if len(sentence.strip()) > 20:  # Meaningful length
                key_messages.append(sentence.strip())
        
        return key_messages
    
    def _identify_potential_challenges(self, question: AnticipatedQuestion) -> List[str]:
        """Identify potential challenges to the response"""
        challenges = []
        
        if question.difficulty == QuestionDifficulty.CHALLENGING:
            challenges.append("May require detailed follow-up discussion")
            challenges.append("Could lead to additional board oversight requirements")
        
        if question.category == QuestionCategory.RISK_MANAGEMENT:
            challenges.append("Board may request more detailed risk assessment")
            challenges.append("May need to provide specific mitigation timelines")
        
        if question.category == QuestionCategory.FINANCIAL_PERFORMANCE:
            challenges.append("May need to provide additional financial detail")
            challenges.append("Could trigger questions about assumptions")
        
        return challenges[:3]  # Limit to top 3 challenges
    
    def _create_backup_responses(self, question: AnticipatedQuestion, primary_response: str) -> List[str]:
        """Create backup responses for different scenarios"""
        backup_responses = []
        
        # Shorter version for time constraints
        backup_responses.append("In summary: " + primary_response.split('.')[0] + ".")
        
        # More detailed version if pressed
        backup_responses.append(primary_response + " I can provide additional detail on any specific aspect.")
        
        # Redirect if needed
        backup_responses.append("This relates to our broader strategic framework, which I'm happy to discuss in more detail.")
        
        return backup_responses
    
    def _calculate_confidence_level(
        self, 
        question: AnticipatedQuestion, 
        supporting_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence level for response"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on supporting data availability
        if len(supporting_data) > 3:
            confidence += 0.2
        elif len(supporting_data) < 2:
            confidence -= 0.2
        
        # Adjust based on question difficulty
        if question.difficulty == QuestionDifficulty.STRAIGHTFORWARD:
            confidence += 0.1
        elif question.difficulty == QuestionDifficulty.CRITICAL:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_preparation_notes(
        self, 
        question: AnticipatedQuestion, 
        strategy: ResponseStrategy
    ) -> str:
        """Generate preparation notes for the response"""
        notes = []
        
        notes.append(f"Strategy: {strategy.value}")
        notes.append(f"Question likelihood: {question.likelihood:.1f}")
        notes.append(f"Difficulty: {question.difficulty.value}")
        
        if question.potential_follow_ups:
            notes.append(f"Prepare for follow-ups: {', '.join(question.potential_follow_ups[:2])}")
        
        return " | ".join(notes)
    
    def _optimize_responses(self, responses: List[PreparedResponse]) -> List[PreparedResponse]:
        """Optimize responses for consistency and effectiveness"""
        # Ensure consistent messaging across responses
        for response in responses:
            response = self._ensure_message_consistency(response, responses)
        
        # Enhance low-confidence responses
        for response in responses:
            if response.confidence_level < 0.6:
                response = self._enhance_low_confidence_response(response)
        
        return responses
    
    def _ensure_message_consistency(
        self, 
        response: PreparedResponse, 
        all_responses: List[PreparedResponse]
    ) -> PreparedResponse:
        """Ensure message consistency across responses"""
        # Check for conflicting messages and align them
        # This is a simplified implementation
        return response
    
    def _enhance_low_confidence_response(self, response: PreparedResponse) -> PreparedResponse:
        """Enhance low-confidence response"""
        # Add more supporting data references
        response.primary_response += " I can provide additional supporting analysis if helpful."
        
        # Increase confidence slightly
        response.confidence_level = min(1.0, response.confidence_level + 0.1)
        
        return response
    
    def _load_response_frameworks(self) -> Dict[str, Any]:
        """Load response frameworks and templates"""
        return {
            "data_driven": {
                "structure": ["context", "data", "analysis", "conclusion"],
                "tone": "analytical"
            },
            "strategic": {
                "structure": ["alignment", "rationale", "execution", "outcomes"],
                "tone": "visionary"
            },
            "collaborative": {
                "structure": ["acknowledgment", "perspective", "invitation"],
                "tone": "inclusive"
            }
        }


class QAEffectivenessTracker:
    """Tracks and improves Q&A effectiveness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def track_qa_effectiveness(
        self, 
        qa_preparation: QAPreparation,
        actual_questions: Optional[List[str]] = None,
        board_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Implement Q&A effectiveness tracking and improvement"""
        try:
            effectiveness_metrics = {
                "preparation_completeness": 0.0,
                "question_prediction_accuracy": 0.0,
                "response_quality_score": 0.0,
                "board_satisfaction_score": 0.0,
                "areas_for_improvement": [],
                "success_indicators": []
            }
            
            # Calculate preparation completeness
            effectiveness_metrics["preparation_completeness"] = self._calculate_preparation_completeness(
                qa_preparation
            )
            
            # Calculate prediction accuracy if actual questions available
            if actual_questions:
                effectiveness_metrics["question_prediction_accuracy"] = self._calculate_prediction_accuracy(
                    qa_preparation.anticipated_questions, actual_questions
                )
            
            # Calculate response quality
            effectiveness_metrics["response_quality_score"] = self._calculate_response_quality(
                qa_preparation.prepared_responses
            )
            
            # Incorporate board feedback if available
            if board_feedback:
                effectiveness_metrics["board_satisfaction_score"] = board_feedback.get("satisfaction_score", 0.0)
                effectiveness_metrics = self._incorporate_board_feedback(
                    effectiveness_metrics, board_feedback
                )
            
            # Generate improvement recommendations
            effectiveness_metrics["areas_for_improvement"] = self._identify_improvement_areas(
                effectiveness_metrics
            )
            
            # Identify success indicators
            effectiveness_metrics["success_indicators"] = self._identify_success_indicators(
                effectiveness_metrics
            )
            
            self.logger.info("Tracked Q&A effectiveness metrics")
            return effectiveness_metrics
            
        except Exception as e:
            self.logger.error(f"Error tracking Q&A effectiveness: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_preparation_completeness(self, qa_preparation: QAPreparation) -> float:
        """Calculate how complete the preparation is"""
        completeness_factors = []
        
        # Question coverage
        if len(qa_preparation.anticipated_questions) >= 10:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(len(qa_preparation.anticipated_questions) / 10.0)
        
        # Response coverage
        response_coverage = len(qa_preparation.prepared_responses) / max(1, len(qa_preparation.anticipated_questions))
        completeness_factors.append(min(1.0, response_coverage))
        
        # Category coverage
        total_categories = len(QuestionCategory)
        covered_categories = len(qa_preparation.question_categories_covered)
        category_coverage = covered_categories / total_categories
        completeness_factors.append(category_coverage)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _calculate_prediction_accuracy(
        self, 
        anticipated_questions: List[AnticipatedQuestion], 
        actual_questions: List[str]
    ) -> float:
        """Calculate accuracy of question predictions"""
        if not actual_questions:
            return 0.0
        
        matches = 0
        for actual_q in actual_questions:
            for anticipated_q in anticipated_questions:
                if self._questions_match(actual_q, anticipated_q.question_text):
                    matches += 1
                    break
        
        return matches / len(actual_questions)
    
    def _questions_match(self, actual_question: str, anticipated_question: str) -> bool:
        """Check if actual and anticipated questions match"""
        # Simple keyword-based matching
        actual_keywords = set(actual_question.lower().split())
        anticipated_keywords = set(anticipated_question.lower().split())
        
        # Remove common words
        common_words = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an"}
        actual_keywords -= common_words
        anticipated_keywords -= common_words
        
        # Calculate overlap
        if not actual_keywords or not anticipated_keywords:
            return False
        
        overlap = len(actual_keywords & anticipated_keywords)
        return overlap / len(actual_keywords) >= 0.4  # 40% keyword overlap
    
    def _calculate_response_quality(self, prepared_responses: List[PreparedResponse]) -> float:
        """Calculate overall response quality"""
        if not prepared_responses:
            return 0.0
        
        quality_scores = []
        
        for response in prepared_responses:
            # Base quality from confidence level
            quality = response.confidence_level
            
            # Adjust for response completeness
            if len(response.key_messages) >= 3:
                quality += 0.1
            if len(response.backup_responses) >= 2:
                quality += 0.1
            if response.supporting_data:
                quality += 0.1
            
            quality_scores.append(min(1.0, quality))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _incorporate_board_feedback(
        self, 
        metrics: Dict[str, Any], 
        board_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Incorporate board feedback into metrics"""
        # Add specific feedback areas
        if "response_clarity" in board_feedback:
            metrics["response_clarity_score"] = board_feedback["response_clarity"]
        
        if "preparation_thoroughness" in board_feedback:
            metrics["preparation_thoroughness_score"] = board_feedback["preparation_thoroughness"]
        
        if "improvement_suggestions" in board_feedback:
            metrics["board_improvement_suggestions"] = board_feedback["improvement_suggestions"]
        
        return metrics
    
    def _identify_improvement_areas(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        if metrics["preparation_completeness"] < 0.8:
            improvements.append("Increase question anticipation coverage")
        
        if metrics["question_prediction_accuracy"] < 0.6:
            improvements.append("Improve question prediction accuracy")
        
        if metrics["response_quality_score"] < 0.7:
            improvements.append("Enhance response development and supporting data")
        
        if metrics["board_satisfaction_score"] < 0.8:
            improvements.append("Better align responses with board expectations")
        
        return improvements
    
    def _identify_success_indicators(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify success indicators"""
        successes = []
        
        if metrics["preparation_completeness"] >= 0.8:
            successes.append("Comprehensive preparation achieved")
        
        if metrics["question_prediction_accuracy"] >= 0.7:
            successes.append("Strong question prediction accuracy")
        
        if metrics["response_quality_score"] >= 0.8:
            successes.append("High-quality response development")
        
        if metrics["board_satisfaction_score"] >= 0.8:
            successes.append("Strong board satisfaction with Q&A")
        
        return successes


class QAPreparationEngine:
    """Main engine for Q&A preparation"""
    
    def __init__(self):
        self.question_anticipator = BoardQuestionAnticipator()
        self.response_developer = ResponseDeveloper()
        self.effectiveness_tracker = QAEffectivenessTracker()
        self.logger = logging.getLogger(__name__)
    
    def create_comprehensive_qa_preparation(
        self,
        presentation: BoardPresentation,
        board: Board,
        board_context: Optional[Dict[str, Any]] = None
    ) -> QAPreparation:
        """Create comprehensive Q&A preparation package"""
        try:
            # Anticipate questions
            anticipated_questions = self.question_anticipator.anticipate_board_questions(
                presentation, board, board_context
            )
            
            # Develop responses
            prepared_responses = self.response_developer.develop_comprehensive_responses(
                anticipated_questions, presentation, board_context
            )
            
            # Calculate overall preparedness
            preparedness_score = self._calculate_preparedness_score(
                anticipated_questions, prepared_responses
            )
            
            # Identify high-risk questions
            high_risk_questions = self._identify_high_risk_questions(anticipated_questions)
            
            # Extract key talking points
            key_talking_points = self._extract_key_talking_points(prepared_responses)
            
            # Get covered categories
            categories_covered = list(set(q.category for q in anticipated_questions))
            
            qa_preparation = QAPreparation(
                id=f"qa_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                presentation_id=presentation.id,
                board_id=board.id,
                anticipated_questions=anticipated_questions,
                prepared_responses=prepared_responses,
                question_categories_covered=categories_covered,
                overall_preparedness_score=preparedness_score,
                high_risk_questions=high_risk_questions,
                key_talking_points=key_talking_points,
                effectiveness_metrics={}
            )
            
            self.logger.info(f"Created comprehensive Q&A preparation: {qa_preparation.id}")
            return qa_preparation
            
        except Exception as e:
            self.logger.error(f"Error creating Q&A preparation: {str(e)}")
            raise
    
    def _calculate_preparedness_score(
        self, 
        questions: List[AnticipatedQuestion], 
        responses: List[PreparedResponse]
    ) -> float:
        """Calculate overall preparedness score"""
        if not questions:
            return 0.0
        
        # Base score from response coverage
        response_coverage = len(responses) / len(questions)
        
        # Adjust for question difficulty coverage
        critical_questions = [q for q in questions if q.difficulty == QuestionDifficulty.CRITICAL]
        critical_responses = [r for r in responses 
                            if any(q.id == r.question_id and q.difficulty == QuestionDifficulty.CRITICAL 
                                  for q in questions)]
        
        critical_coverage = len(critical_responses) / max(1, len(critical_questions))
        
        # Adjust for response quality
        avg_confidence = sum(r.confidence_level for r in responses) / max(1, len(responses))
        
        # Calculate overall score
        preparedness = (response_coverage * 0.4 + critical_coverage * 0.4 + avg_confidence * 0.2)
        
        return min(1.0, preparedness)
    
    def _identify_high_risk_questions(self, questions: List[AnticipatedQuestion]) -> List[str]:
        """Identify high-risk questions requiring special attention"""
        high_risk = []
        
        for question in questions:
            if (question.difficulty == QuestionDifficulty.CRITICAL or 
                question.likelihood > 0.8 or
                question.category in [QuestionCategory.RISK_MANAGEMENT, QuestionCategory.GOVERNANCE_COMPLIANCE]):
                high_risk.append(question.question_text)
        
        return high_risk[:5]  # Top 5 high-risk questions
    
    def _extract_key_talking_points(self, responses: List[PreparedResponse]) -> List[str]:
        """Extract key talking points from responses"""
        talking_points = []
        
        for response in responses:
            talking_points.extend(response.key_messages)
        
        # Remove duplicates and limit
        unique_points = list(set(talking_points))
        return unique_points[:10]  # Top 10 talking points