"""
Natural Language Interface for ScrollIntel Core
Handles parsing user queries, intent classification, entity extraction, and response generation
"""
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Intent types for agent routing"""
    CTO = "cto"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    BI = "bi"
    AI_ENGINEER = "ai_engineer"
    QA = "qa"
    FORECAST = "forecast"
    GENERAL = "general"


@dataclass
class Entity:
    """Extracted entity from user query"""
    type: str
    value: str
    confidence: float
    start_pos: int
    end_pos: int


@dataclass
class ParsedQuery:
    """Parsed user query with intent and entities"""
    original_query: str
    intent: Intent
    confidence: float
    entities: List[Entity]
    parameters: Dict[str, Any]
    context_needed: List[str]
    suggested_agent: str


class ConversationMemory:
    """Manages conversation history and context for multi-turn interactions"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.context_cache: Dict[str, Dict[str, Any]] = {}
    
    def add_interaction(self, session_id: str, user_query: str, agent_response: Dict[str, Any], 
                       intent: Intent, entities: List[Entity]):
        """Add interaction to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_query": user_query,
            "intent": intent.value,
            "entities": [{"type": e.type, "value": e.value, "confidence": e.confidence} for e in entities],
            "agent_response": agent_response,
            "agent_name": agent_response.get("agent", "unknown")
        }
        
        self.conversations[session_id].append(interaction)
        
        # Keep only recent interactions
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
        
        # Update context cache
        self._update_context_cache(session_id, interaction)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for session"""
        return self.conversations.get(session_id, [])
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for session"""
        return self.context_cache.get(session_id, {})
    
    def _update_context_cache(self, session_id: str, interaction: Dict[str, Any]):
        """Update context cache with relevant information from interaction"""
        if session_id not in self.context_cache:
            self.context_cache[session_id] = {
                "last_agent": None,
                "current_dataset": None,
                "active_analysis": None,
                "user_preferences": {},
                "mentioned_entities": []
            }
        
        context = self.context_cache[session_id]
        
        # Update last agent
        context["last_agent"] = interaction["agent_name"]
        
        # Extract and store mentioned entities
        for entity in interaction["entities"]:
            if entity not in context["mentioned_entities"]:
                context["mentioned_entities"].append(entity)
        
        # Keep only recent entities
        if len(context["mentioned_entities"]) > 20:
            context["mentioned_entities"] = context["mentioned_entities"][-20:]
        
        # Extract dataset references
        if "dataset" in interaction["user_query"].lower():
            # Simple extraction - could be enhanced with NER
            dataset_match = re.search(r'dataset[:\s]+([^\s,]+)', interaction["user_query"].lower())
            if dataset_match:
                context["current_dataset"] = dataset_match.group(1)
    
    def clear_session(self, session_id: str):
        """Clear conversation history for session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.context_cache:
            del self.context_cache[session_id]


class EntityExtractor:
    """Extracts entities from user queries"""
    
    def __init__(self):
        # Define entity patterns
        self.patterns = {
            "dataset": [
                r'\b(?:dataset|data|file|csv|excel)\s+(?:named\s+)?["\']?([a-zA-Z0-9_\-\.]+)["\']?',
                r'\b(?:from|using|with)\s+(?:the\s+)?(?:dataset\s+)?["\']?([a-zA-Z0-9_\-\.]+)["\']?'
            ],
            "column": [
                r'\b(?:column|field|variable)\s+["\']?([a-zA-Z0-9_\-]+)["\']?',
                r'\b(?:by|group by|sort by)\s+["\']?([a-zA-Z0-9_\-]+)["\']?'
            ],
            "metric": [
                r'\b(accuracy|precision|recall|f1|rmse|mae|r2|auc)\b',
                r'\b(mean|average|sum|count|max|min|median)\b'
            ],
            "time_period": [
                r'\b(?:last|past|next)\s+(\d+)\s+(days?|weeks?|months?|years?)\b',
                r'\b(daily|weekly|monthly|yearly|quarterly)\b'
            ],
            "number": [
                r'\b(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)?\b'
            ],
            "model_type": [
                r'\b(classification|regression|clustering|forecasting)\s+model\b',
                r'\b(random\s+forest|linear\s+regression|svm|neural\s+network|xgboost)\b'
            ]
        }
    
    def extract_entities(self, query: str) -> List[Entity]:
        """Extract entities from query"""
        entities = []
        query_lower = query.lower()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    entity_value = match.group(1) if match.groups() else match.group(0)
                    
                    # Calculate confidence based on pattern specificity
                    confidence = 0.8 if len(match.groups()) > 0 else 0.6
                    
                    entities.append(Entity(
                        type=entity_type,
                        value=entity_value.strip(),
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return entities


class IntentClassifier:
    """Classifies user intent to route to appropriate agent"""
    
    def __init__(self):
        # Define intent keywords with weights
        self.intent_keywords = {
            Intent.CTO: {
                "architecture": 1.0, "technology": 0.9, "stack": 0.9, "scaling": 1.0,
                "infrastructure": 1.0, "system": 0.7, "design": 0.8, "cto": 1.0,
                "technical": 0.8, "platform": 0.7, "deployment": 0.8, "performance": 0.7
            },
            Intent.DATA_SCIENTIST: {
                "analyze": 0.9, "analysis": 0.9, "statistics": 1.0, "correlation": 1.0,
                "insights": 0.9, "explore": 0.8, "data": 0.6, "scientist": 1.0,
                "distribution": 0.9, "outliers": 0.9, "patterns": 0.8, "trends": 0.7
            },
            Intent.ML_ENGINEER: {
                "machine": 0.9, "learning": 0.9, "model": 1.0, "predict": 0.9,
                "train": 0.9, "algorithm": 1.0, "ml": 1.0, "engineer": 0.8,
                "classification": 1.0, "regression": 1.0, "clustering": 1.0,
                "accuracy": 0.8, "precision": 0.8, "recall": 0.8
            },
            Intent.BI: {
                "dashboard": 1.0, "report": 0.9, "kpi": 1.0, "business": 0.7,
                "intelligence": 0.9, "bi": 1.0, "visualization": 0.9, "metrics": 0.9,
                "chart": 0.8, "graph": 0.8, "summary": 0.7, "overview": 0.7
            },
            Intent.AI_ENGINEER: {
                "ai": 1.0, "artificial": 0.9, "intelligence": 0.9, "strategy": 0.9,
                "implementation": 0.8, "roadmap": 0.9, "engineer": 0.7,
                "neural": 0.9, "deep": 0.8, "automation": 0.7
            },
            Intent.QA: {
                "what": 0.8, "how": 0.8, "when": 0.8, "where": 0.8, "why": 0.8,
                "show": 0.9, "find": 0.9, "search": 0.9, "query": 0.9,
                "question": 1.0, "tell": 0.8, "explain": 0.8
            },
            Intent.FORECAST: {
                "forecast": 1.0, "predict": 0.9, "future": 0.9, "trend": 0.9,
                "projection": 1.0, "time": 0.7, "series": 0.8, "seasonal": 0.9,
                "growth": 0.8, "decline": 0.8, "next": 0.7
            }
        }
    
    def classify_intent(self, query: str, context: Dict[str, Any] = None) -> Tuple[Intent, float]:
        """Classify user intent with confidence score"""
        query_lower = query.lower()
        context = context or {}
        
        # Calculate scores for each intent
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0.0
            word_count = 0
            
            for keyword, weight in keywords.items():
                if keyword in query_lower:
                    score += weight
                    word_count += 1
            
            # Normalize by query length and keyword density
            query_words = len(query_lower.split())
            if query_words > 0 and word_count > 0:
                intent_scores[intent] = (score * word_count) / query_words
            else:
                intent_scores[intent] = 0.0
        
        # Apply context boost
        if context.get("last_agent"):
            last_agent_intent = Intent(context["last_agent"])
            if last_agent_intent in intent_scores:
                intent_scores[last_agent_intent] *= 1.2  # 20% boost for context continuity
        
        # Find best intent
        if not intent_scores or max(intent_scores.values()) == 0:
            return Intent.QA, 0.1  # Default to QA with low confidence
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1], 1.0)  # Cap at 1.0
        
        return best_intent[0], confidence


class ResponseGenerator:
    """Generates natural language responses from agent results"""
    
    def __init__(self):
        self.response_templates = {
            Intent.CTO: {
                "success": [
                    "Based on my technical analysis, here's what I recommend:",
                    "From a CTO perspective, here are my suggestions:",
                    "After evaluating your technical requirements:"
                ],
                "error": [
                    "I encountered an issue while analyzing your technical requirements.",
                    "There was a problem processing your architecture request."
                ]
            },
            Intent.DATA_SCIENTIST: {
                "success": [
                    "I've analyzed your data and found some interesting insights:",
                    "Here's what the data analysis reveals:",
                    "Based on my statistical analysis:"
                ],
                "error": [
                    "I couldn't complete the data analysis due to an issue.",
                    "There was a problem analyzing your dataset."
                ]
            },
            Intent.ML_ENGINEER: {
                "success": [
                    "I've built and evaluated the machine learning model:",
                    "Here are the results from the ML model training:",
                    "The machine learning analysis shows:"
                ],
                "error": [
                    "I encountered an issue while building the ML model.",
                    "There was a problem with the machine learning process."
                ]
            },
            Intent.BI: {
                "success": [
                    "I've created a business intelligence summary:",
                    "Here's your business dashboard overview:",
                    "The BI analysis shows:"
                ],
                "error": [
                    "I couldn't generate the business intelligence report.",
                    "There was an issue creating your dashboard."
                ]
            },
            Intent.AI_ENGINEER: {
                "success": [
                    "Here's my AI strategy recommendation:",
                    "Based on AI best practices, I suggest:",
                    "From an AI implementation perspective:"
                ],
                "error": [
                    "I couldn't complete the AI strategy analysis.",
                    "There was an issue with the AI recommendation."
                ]
            },
            Intent.QA: {
                "success": [
                    "Here's what I found:",
                    "Based on your query:",
                    "The answer to your question is:"
                ],
                "error": [
                    "I couldn't find an answer to your question.",
                    "There was an issue processing your query."
                ]
            },
            Intent.FORECAST: {
                "success": [
                    "Here's the forecast analysis:",
                    "Based on the time series data:",
                    "The prediction shows:"
                ],
                "error": [
                    "I couldn't generate the forecast.",
                    "There was an issue with the prediction analysis."
                ]
            }
        }
    
    def generate_response(self, agent_response: Dict[str, Any], intent: Intent, 
                         original_query: str) -> str:
        """Generate natural language response from agent result"""
        try:
            success = agent_response.get("success", False)
            result = agent_response.get("result")
            error = agent_response.get("error")
            
            # Get appropriate template
            templates = self.response_templates.get(intent, self.response_templates[Intent.QA])
            template_key = "success" if success else "error"
            intro = templates[template_key][0]  # Use first template for now
            
            if success and result:
                # Format successful response
                if isinstance(result, dict):
                    response_parts = [intro]
                    
                    # Add key findings
                    if "summary" in result:
                        response_parts.append(f"\n{result['summary']}")
                    
                    if "insights" in result and isinstance(result["insights"], list):
                        response_parts.append("\nKey insights:")
                        for insight in result["insights"][:3]:  # Limit to top 3
                            response_parts.append(f"• {insight}")
                    
                    if "recommendations" in result and isinstance(result["recommendations"], list):
                        response_parts.append("\nRecommendations:")
                        for rec in result["recommendations"][:3]:  # Limit to top 3
                            response_parts.append(f"• {rec}")
                    
                    # Add metrics if available
                    if "metrics" in result:
                        response_parts.append(f"\nMetrics: {self._format_metrics(result['metrics'])}")
                    
                    return "\n".join(response_parts)
                
                elif isinstance(result, str):
                    return f"{intro}\n{result}"
                
                else:
                    return f"{intro}\n{str(result)}"
            
            else:
                # Format error response
                error_msg = error or "Unknown error occurred"
                return f"{intro}\nError: {error_msg}"
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an issue generating a response to your query: {original_query}"
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display"""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.3f}")
            else:
                formatted.append(f"{key}: {value}")
        return ", ".join(formatted)


class NLProcessor:
    """Main Natural Language Processor that coordinates all NL components"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self.conversation_memory = ConversationMemory()
        
        logger.info("NLProcessor initialized")
    
    def parse_query(self, query: str, session_id: str = None, 
                   user_context: Dict[str, Any] = None) -> ParsedQuery:
        """Parse user query and extract intent, entities, and parameters"""
        try:
            # Get conversation context
            context = {}
            if session_id:
                context = self.conversation_memory.get_context(session_id)
                context.update(user_context or {})
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(query)
            
            # Classify intent
            intent, confidence = self.intent_classifier.classify_intent(query, context)
            
            # Extract parameters from entities
            parameters = self._entities_to_parameters(entities)
            
            # Determine context needed
            context_needed = self._determine_context_needed(query, entities, intent)
            
            # Map intent to agent name
            agent_mapping = {
                Intent.CTO: "cto",
                Intent.DATA_SCIENTIST: "data_scientist",
                Intent.ML_ENGINEER: "ml_engineer",
                Intent.BI: "bi",
                Intent.AI_ENGINEER: "ai_engineer",
                Intent.QA: "qa",
                Intent.FORECAST: "forecast",
                Intent.GENERAL: "qa"
            }
            
            return ParsedQuery(
                original_query=query,
                intent=intent,
                confidence=confidence,
                entities=entities,
                parameters=parameters,
                context_needed=context_needed,
                suggested_agent=agent_mapping[intent]
            )
        
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Return default parsing result
            return ParsedQuery(
                original_query=query,
                intent=Intent.QA,
                confidence=0.1,
                entities=[],
                parameters={},
                context_needed=[],
                suggested_agent="qa"
            )
    
    def process_conversation_turn(self, query: str, agent_response: Dict[str, Any],
                                session_id: str, user_context: Dict[str, Any] = None) -> str:
        """Process a complete conversation turn and generate response"""
        try:
            # Parse the query
            parsed = self.parse_query(query, session_id, user_context)
            
            # Store interaction in memory
            self.conversation_memory.add_interaction(
                session_id, query, agent_response, parsed.intent, parsed.entities
            )
            
            # Generate natural language response
            nl_response = self.response_generator.generate_response(
                agent_response, parsed.intent, query
            )
            
            return nl_response
        
        except Exception as e:
            logger.error(f"Error processing conversation turn: {e}")
            return f"I apologize, but I encountered an issue processing your request: {query}"
    
    def _entities_to_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """Convert extracted entities to parameters"""
        parameters = {}
        
        for entity in entities:
            if entity.type == "dataset":
                parameters["dataset_name"] = entity.value
            elif entity.type == "column":
                if "columns" not in parameters:
                    parameters["columns"] = []
                parameters["columns"].append(entity.value)
            elif entity.type == "metric":
                if "metrics" not in parameters:
                    parameters["metrics"] = []
                parameters["metrics"].append(entity.value)
            elif entity.type == "time_period":
                parameters["time_period"] = entity.value
            elif entity.type == "number":
                try:
                    parameters["number"] = float(entity.value)
                except ValueError:
                    parameters["number_text"] = entity.value
            elif entity.type == "model_type":
                parameters["model_type"] = entity.value
        
        return parameters
    
    def _determine_context_needed(self, query: str, entities: List[Entity], 
                                intent: Intent) -> List[str]:
        """Determine what context information is needed"""
        context_needed = []
        
        # Check if dataset is mentioned but not specified
        if any(word in query.lower() for word in ["data", "dataset", "file"]):
            if not any(e.type == "dataset" for e in entities):
                context_needed.append("dataset_selection")
        
        # Check for ML-specific context needs
        if intent == Intent.ML_ENGINEER:
            if not any(e.type == "model_type" for e in entities):
                context_needed.append("model_type")
            if "target" not in query.lower() and "predict" in query.lower():
                context_needed.append("target_variable")
        
        # Check for BI-specific context needs
        if intent == Intent.BI:
            if not any(e.type == "metric" for e in entities):
                context_needed.append("metrics_selection")
        
        # Check for forecast-specific context needs
        if intent == Intent.FORECAST:
            if not any(e.type == "time_period" for e in entities):
                context_needed.append("forecast_horizon")
        
        return context_needed
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for session"""
        return self.conversation_memory.get_conversation_history(session_id)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for session"""
        self.conversation_memory.clear_session(session_id)
    
    def get_context_suggestions(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Get suggestions for improving query based on context"""
        parsed = self.parse_query(query, session_id)
        
        suggestions = {
            "confidence": parsed.confidence,
            "suggested_agent": parsed.suggested_agent,
            "missing_context": parsed.context_needed,
            "extracted_entities": [
                {"type": e.type, "value": e.value, "confidence": e.confidence} 
                for e in parsed.entities
            ],
            "suggestions": []
        }
        
        # Add specific suggestions based on missing context
        if "dataset_selection" in parsed.context_needed:
            suggestions["suggestions"].append("Please specify which dataset you'd like to analyze")
        
        if "model_type" in parsed.context_needed:
            suggestions["suggestions"].append("What type of model would you like to build? (classification, regression, etc.)")
        
        if "target_variable" in parsed.context_needed:
            suggestions["suggestions"].append("What variable would you like to predict?")
        
        if "metrics_selection" in parsed.context_needed:
            suggestions["suggestions"].append("What metrics or KPIs would you like to track?")
        
        if "forecast_horizon" in parsed.context_needed:
            suggestions["suggestions"].append("How far into the future would you like to forecast?")
        
        # Add general suggestions for low confidence
        if parsed.confidence < 0.5:
            suggestions["suggestions"].append("Try being more specific about what you'd like to accomplish")
            suggestions["suggestions"].append("Consider mentioning the type of analysis or task you need")
        
        return suggestions