"""
Intent classification system for automated code generation.
Classifies user intents and requirements for appropriate code generation strategies.
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

from scrollintel.models.code_generation_models import Intent, RequirementType
from scrollintel.core.config import get_settings


class IntentClassifier:
    """Classifies user intents for code generation requirements."""
    
    def __init__(self):
        """Initialize the intent classifier."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = "gpt-4-turbo-preview"
        
        # Intent patterns for fallback classification
        self.intent_patterns = {
            Intent.CREATE_APPLICATION: [
                r'\b(?:create|build|develop|make)\s+(?:an?\s+)?(?:application|app|system|platform)\b',
                r'\bneed\s+(?:an?\s+)?(?:new|complete)\s+(?:application|system)\b',
                r'\bfrom\s+scratch\b'
            ],
            Intent.MODIFY_FEATURE: [
                r'\b(?:modify|change|update|alter|enhance)\s+(?:the\s+)?(?:feature|functionality)\b',
                r'\b(?:improve|upgrade)\s+(?:existing|current)\b',
                r'\badd\s+(?:new\s+)?(?:feature|functionality)\b'
            ],
            Intent.ADD_INTEGRATION: [
                r'\b(?:integrate|connect)\s+(?:with|to)\b',
                r'\b(?:api|service|system)\s+integration\b',
                r'\bthird[- ]party\s+(?:service|api)\b'
            ],
            Intent.IMPROVE_PERFORMANCE: [
                r'\b(?:optimize|improve|enhance)\s+(?:performance|speed|efficiency)\b',
                r'\b(?:faster|quicker|more\s+efficient)\b',
                r'\b(?:reduce|minimize)\s+(?:latency|response\s+time)\b'
            ],
            Intent.ADD_SECURITY: [
                r'\b(?:secure|protect|authenticate|authorize)\b',
                r'\bsecurity\s+(?:feature|measure|requirement)\b',
                r'\b(?:encryption|authentication|authorization)\b'
            ],
            Intent.CREATE_API: [
                r'\b(?:create|build|develop)\s+(?:an?\s+)?(?:api|rest\s+api|graphql)\b',
                r'\b(?:endpoint|route)\b',
                r'\bapi\s+(?:design|specification)\b'
            ],
            Intent.DESIGN_DATABASE: [
                r'\b(?:database|db)\s+(?:design|schema|structure)\b',
                r'\b(?:create|design)\s+(?:tables|entities|models)\b',
                r'\bdata\s+(?:model|structure|schema)\b'
            ],
            Intent.BUILD_UI: [
                r'\b(?:user\s+interface|ui|frontend|web\s+interface)\b',
                r'\b(?:create|build|design)\s+(?:forms|pages|components)\b',
                r'\b(?:responsive|mobile[- ]friendly)\s+(?:design|interface)\b'
            ],
            Intent.SETUP_DEPLOYMENT: [
                r'\b(?:deploy|deployment|hosting)\b',
                r'\b(?:docker|container|kubernetes)\b',
                r'\b(?:cloud|aws|azure|gcp)\s+(?:deployment|hosting)\b'
            ],
            Intent.GENERATE_TESTS: [
                r'\b(?:test|testing|unit\s+test|integration\s+test)\b',
                r'\b(?:automated|automatic)\s+testing\b',
                r'\btest\s+(?:coverage|suite|cases)\b'
            ]
        }
    
    def classify_intent(self, requirement: str) -> Tuple[Intent, float]:
        """
        Classify the intent of a requirement.
        
        Args:
            requirement: The requirement text to classify
            
        Returns:
            Tuple of (Intent, confidence_score)
        """
        try:
            # Try GPT-4 classification first
            return self._classify_with_gpt4(requirement)
        except Exception:
            # Fallback to pattern-based classification
            return self._classify_with_patterns(requirement)
    
    def classify_multiple_intents(self, requirements: List[str]) -> List[Tuple[Intent, float]]:
        """
        Classify intents for multiple requirements.
        
        Args:
            requirements: List of requirement texts
            
        Returns:
            List of (Intent, confidence_score) tuples
        """
        results = []
        for requirement in requirements:
            intent, confidence = self.classify_intent(requirement)
            results.append((intent, confidence))
        return results
    
    def get_intent_distribution(self, requirements: List[str]) -> Dict[Intent, int]:
        """
        Get distribution of intents across requirements.
        
        Args:
            requirements: List of requirement texts
            
        Returns:
            Dictionary mapping intents to their counts
        """
        distribution = {intent: 0 for intent in Intent}
        
        for requirement in requirements:
            intent, _ = self.classify_intent(requirement)
            distribution[intent] += 1
        
        return distribution
    
    def suggest_architecture_patterns(self, intents: List[Intent]) -> List[str]:
        """
        Suggest architecture patterns based on identified intents.
        
        Args:
            intents: List of identified intents
            
        Returns:
            List of suggested architecture patterns
        """
        patterns = []
        intent_set = set(intents)
        
        # Microservices for complex integrations
        if Intent.ADD_INTEGRATION in intent_set and len(intent_set) > 3:
            patterns.append("Microservices Architecture")
        
        # MVC for web applications
        if Intent.BUILD_UI in intent_set and Intent.CREATE_API in intent_set:
            patterns.append("Model-View-Controller (MVC)")
        
        # Layered architecture for data-heavy applications
        if Intent.DESIGN_DATABASE in intent_set and Intent.CREATE_API in intent_set:
            patterns.append("Layered Architecture")
        
        # Event-driven for real-time features
        if Intent.IMPROVE_PERFORMANCE in intent_set:
            patterns.append("Event-Driven Architecture")
        
        # Serverless for simple applications
        if len(intent_set) <= 2 and Intent.CREATE_APPLICATION in intent_set:
            patterns.append("Serverless Architecture")
        
        return patterns if patterns else ["Monolithic Architecture"]
    
    def _classify_with_gpt4(self, requirement: str) -> Tuple[Intent, float]:
        """Classify intent using GPT-4."""
        intent_descriptions = {
            Intent.CREATE_APPLICATION: "Building a complete new application or system from scratch",
            Intent.MODIFY_FEATURE: "Modifying, updating, or enhancing existing features",
            Intent.ADD_INTEGRATION: "Integrating with external services, APIs, or systems",
            Intent.IMPROVE_PERFORMANCE: "Optimizing performance, speed, or efficiency",
            Intent.ADD_SECURITY: "Adding security features, authentication, or authorization",
            Intent.CREATE_API: "Creating REST APIs, GraphQL endpoints, or web services",
            Intent.DESIGN_DATABASE: "Designing database schemas, data models, or data structures",
            Intent.BUILD_UI: "Creating user interfaces, web pages, or frontend components",
            Intent.SETUP_DEPLOYMENT: "Setting up deployment, hosting, or infrastructure",
            Intent.GENERATE_TESTS: "Creating automated tests, test suites, or testing frameworks"
        }
        
        prompt = f"""
        Classify the following requirement into one of these intents:
        
        {json.dumps(intent_descriptions, indent=2)}
        
        Requirement: "{requirement}"
        
        Return a JSON object with:
        - "intent": the most appropriate intent key
        - "confidence": confidence score from 0.0 to 1.0
        - "reasoning": brief explanation of the classification
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at classifying software requirements by intent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        intent = Intent(result["intent"])
        confidence = float(result["confidence"])
        
        return intent, confidence
    
    def _classify_with_patterns(self, requirement: str) -> Tuple[Intent, float]:
        """Classify intent using pattern matching."""
        requirement_lower = requirement.lower()
        best_intent = Intent.CREATE_APPLICATION
        best_score = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, requirement_lower):
                    matches += 1
                    score += 1.0
            
            # Normalize score by number of patterns
            if matches > 0:
                score = score / len(patterns)
                if score > best_score:
                    best_score = score
                    best_intent = intent
        
        # Default confidence based on pattern matching
        confidence = min(0.8, best_score + 0.3) if best_score > 0 else 0.4
        
        return best_intent, confidence
    
    def get_requirement_complexity(self, requirement: str, intent: Intent) -> int:
        """
        Estimate requirement complexity based on intent and content.
        
        Args:
            requirement: The requirement text
            intent: The classified intent
            
        Returns:
            Complexity score from 1 (simple) to 5 (very complex)
        """
        base_complexity = {
            Intent.CREATE_APPLICATION: 4,
            Intent.MODIFY_FEATURE: 2,
            Intent.ADD_INTEGRATION: 3,
            Intent.IMPROVE_PERFORMANCE: 4,
            Intent.ADD_SECURITY: 3,
            Intent.CREATE_API: 2,
            Intent.DESIGN_DATABASE: 3,
            Intent.BUILD_UI: 2,
            Intent.SETUP_DEPLOYMENT: 3,
            Intent.GENERATE_TESTS: 2
        }
        
        complexity = base_complexity.get(intent, 3)
        
        # Adjust based on requirement content
        requirement_lower = requirement.lower()
        
        # Increase complexity for multiple systems/technologies
        if len(re.findall(r'\b(?:system|service|api|database|ui|frontend|backend)\b', requirement_lower)) > 2:
            complexity += 1
        
        # Increase complexity for real-time requirements
        if re.search(r'\b(?:real[- ]time|live|instant|immediate)\b', requirement_lower):
            complexity += 1
        
        # Increase complexity for scalability requirements
        if re.search(r'\b(?:scalable|scale|high[- ]volume|concurrent)\b', requirement_lower):
            complexity += 1
        
        return min(5, max(1, complexity))