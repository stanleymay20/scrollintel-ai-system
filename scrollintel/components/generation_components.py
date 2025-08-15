"""
Generation Components - Focused content generation modules

This module contains specialized generation components that follow the
modular architecture pattern for creating different types of content.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re
from datetime import datetime
import random

from ..core.modular_components import (
    BaseComponent, ComponentType, ComponentMetadata, component_registry
)

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Standardized generation request format"""
    request_id: str
    generation_type: str
    prompt: str
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class GenerationResult:
    """Standardized generation result format"""
    request_id: str
    generation_type: str
    content: str
    quality_metrics: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any]

class ContentTemplateComponent(BaseComponent):
    """Component for managing content templates and patterns"""
    
    def __init__(self):
        super().__init__(
            component_id="content_template",
            component_type=ComponentType.SERVICE,
            name="Content Template Manager",
            version="1.0.0"
        )
        self._templates = {}
        self._patterns = {}
    
    async def _initialize_impl(self) -> bool:
        """Initialize content templates and patterns"""
        self._templates = {
            "summary": {
                "structure": ["introduction", "key_points", "conclusion"],
                "intro_phrases": [
                    "This document summarizes",
                    "The following provides an overview of",
                    "Key findings include"
                ],
                "transition_phrases": [
                    "Furthermore", "Additionally", "Moreover", "In addition"
                ],
                "conclusion_phrases": [
                    "In conclusion", "To summarize", "Overall"
                ]
            },
            "report": {
                "structure": ["executive_summary", "methodology", "findings", "recommendations"],
                "section_headers": {
                    "executive_summary": "Executive Summary",
                    "methodology": "Methodology",
                    "findings": "Key Findings",
                    "recommendations": "Recommendations"
                }
            },
            "explanation": {
                "structure": ["definition", "examples", "applications"],
                "explanation_starters": [
                    "To understand this concept",
                    "This can be explained as",
                    "Simply put"
                ]
            }
        }
        
        self._patterns = {
            "bullet_points": r"^[\s]*[-*•]\s+(.+)$",
            "numbered_list": r"^[\s]*\d+\.\s+(.+)$",
            "headers": r"^#+\s+(.+)$",
            "emphasis": r"\*\*(.+?)\*\*|__(.+?)__"
        }
        
        logger.info("Content template component initialized")
        return True
    
    async def _shutdown_impl(self) -> bool:
        """Cleanup template resources"""
        self._templates.clear()
        self._patterns.clear()
        logger.info("Content template component shutdown")
        return True
    
    def get_template(self, template_type: str) -> Optional[Dict[str, Any]]:
        """Get template by type"""
        return self._templates.get(template_type)
    
    def get_pattern(self, pattern_name: str) -> Optional[str]:
        """Get regex pattern by name"""
        return self._patterns.get(pattern_name)
    
    def format_content(self, content: str, format_type: str) -> str:
        """Format content according to specified type"""
        if format_type == "markdown":
            return self._format_as_markdown(content)
        elif format_type == "html":
            return self._format_as_html(content)
        elif format_type == "plain":
            return self._format_as_plain_text(content)
        else:
            return content
    
    def _format_as_markdown(self, content: str) -> str:
        """Format content as markdown"""
        # Add basic markdown formatting
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Check if it's a header (starts with capital and ends with colon)
            if line.endswith(':') and line[0].isupper():
                formatted_lines.append(f"## {line[:-1]}")
            # Check if it's a list item
            elif line.startswith('-') or line.startswith('*'):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_as_html(self, content: str) -> str:
        """Format content as HTML"""
        # Basic HTML formatting
        content = content.replace('\n\n', '</p><p>')
        content = f"<p>{content}</p>"
        
        # Format headers
        content = re.sub(r'<p>([^<]+):</p>', r'<h2>\1</h2>', content)
        
        return content
    
    def _format_as_plain_text(self, content: str) -> str:
        """Format content as plain text"""
        # Remove any markdown or HTML formatting
        content = re.sub(r'[#*_`]', '', content)
        content = re.sub(r'<[^>]+>', '', content)
        return content.strip()

class TextGeneratorComponent(BaseComponent):
    """Component for generating text content"""
    
    def __init__(self):
        super().__init__(
            component_id="text_generator",
            component_type=ComponentType.GENERATOR,
            name="Text Generator",
            version="1.0.0"
        )
        self.add_dependency("content_template")
        self._generation_strategies = {}
    
    async def _initialize_impl(self) -> bool:
        """Initialize text generation strategies"""
        self._generation_strategies = {
            "summary": self._generate_summary,
            "explanation": self._generate_explanation,
            "report": self._generate_report,
            "creative": self._generate_creative_content,
            "technical": self._generate_technical_content
        }
        logger.info("Text generator component initialized")
        return True
    
    async def _shutdown_impl(self) -> bool:
        """Cleanup generation resources"""
        self._generation_strategies.clear()
        logger.info("Text generator component shutdown")
        return True
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text content based on request"""
        start_time = datetime.now()
        
        try:
            if request.generation_type not in self._generation_strategies:
                available_types = list(self._generation_strategies.keys())
                return GenerationResult(
                    request_id=request.request_id,
                    generation_type=request.generation_type,
                    content=f"Error: Unknown generation type '{request.generation_type}'. Available types: {available_types}",
                    quality_metrics={"error": 1.0},
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"error": "unknown_generation_type"}
                )
            
            strategy = self._generation_strategies[request.generation_type]
            content = await strategy(request.prompt, request.parameters, request.constraints)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(content, request)
            
            # Format content if requested
            format_type = request.parameters.get("format", "plain")
            template_component = component_registry.get_component("content_template")
            if template_component and template_component.status.value == "ready":
                content = template_component.component.format_content(content, format_type)
            
            return GenerationResult(
                request_id=request.request_id,
                generation_type=request.generation_type,
                content=content,
                quality_metrics=quality_metrics,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"strategy": request.generation_type, "format": format_type}
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return GenerationResult(
                request_id=request.request_id,
                generation_type=request.generation_type,
                content=f"Error generating content: {str(e)}",
                quality_metrics={"error": 1.0},
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"exception": str(e)}
            )
    
    async def _generate_summary(self, prompt: str, parameters: Dict[str, Any], 
                              constraints: Dict[str, Any]) -> str:
        """Generate summary content"""
        max_length = constraints.get("max_length", 500)
        key_points = parameters.get("key_points", 3)
        
        # Get template
        template_component = component_registry.get_component("content_template")
        template = None
        if template_component and template_component.status.value == "ready":
            template = template_component.component.get_template("summary")
        
        # Generate summary structure
        intro_phrases = template["intro_phrases"] if template else ["This summarizes"]
        intro = f"{random.choice(intro_phrases)} the following: {prompt[:100]}..."
        
        # Generate key points (simulated)
        points = []
        for i in range(key_points):
            points.append(f"Key point {i+1}: Important aspect related to {prompt.split()[0] if prompt.split() else 'topic'}")
        
        conclusion = "In summary, these points highlight the main aspects of the topic."
        
        content = f"{intro}\n\n"
        for point in points:
            content += f"• {point}\n"
        content += f"\n{conclusion}"
        
        # Trim to max length if needed
        if len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        return content
    
    async def _generate_explanation(self, prompt: str, parameters: Dict[str, Any], 
                                  constraints: Dict[str, Any]) -> str:
        """Generate explanation content"""
        complexity_level = parameters.get("complexity", "medium")
        include_examples = parameters.get("include_examples", True)
        
        # Adjust explanation based on complexity
        if complexity_level == "simple":
            intro = f"Simply put, {prompt.lower()} is"
            detail_level = "basic"
        elif complexity_level == "advanced":
            intro = f"From a technical perspective, {prompt}"
            detail_level = "detailed"
        else:
            intro = f"To understand {prompt}, we need to consider"
            detail_level = "moderate"
        
        explanation = f"{intro} a concept that involves multiple aspects.\n\n"
        explanation += f"The main idea is that {prompt.lower()} represents an important principle in its domain. "
        explanation += f"This concept has several key characteristics that make it significant.\n\n"
        
        if include_examples:
            explanation += "For example:\n"
            explanation += f"• Example 1: A practical application of {prompt.lower()}\n"
            explanation += f"• Example 2: Another way {prompt.lower()} is used\n\n"
        
        explanation += f"Understanding {prompt.lower()} is important because it helps us grasp fundamental principles in the field."
        
        return explanation
    
    async def _generate_report(self, prompt: str, parameters: Dict[str, Any], 
                             constraints: Dict[str, Any]) -> str:
        """Generate report content"""
        sections = parameters.get("sections", ["summary", "analysis", "recommendations"])
        formal_tone = parameters.get("formal_tone", True)
        
        report = f"# Report: {prompt}\n\n"
        
        for section in sections:
            if section == "summary":
                report += "## Executive Summary\n\n"
                report += f"This report examines {prompt.lower()} and provides key insights. "
                report += "The analysis reveals important findings that inform our understanding.\n\n"
            
            elif section == "analysis":
                report += "## Analysis\n\n"
                report += f"Our analysis of {prompt.lower()} indicates several important factors:\n\n"
                report += "• Factor 1: Primary consideration in the analysis\n"
                report += "• Factor 2: Secondary but significant element\n"
                report += "• Factor 3: Supporting evidence and data\n\n"
            
            elif section == "recommendations":
                report += "## Recommendations\n\n"
                report += "Based on our analysis, we recommend the following actions:\n\n"
                report += "1. Immediate action: Address primary concerns\n"
                report += "2. Short-term strategy: Implement supporting measures\n"
                report += "3. Long-term planning: Develop comprehensive approach\n\n"
        
        report += "## Conclusion\n\n"
        report += f"This report provides a comprehensive overview of {prompt.lower()} and offers actionable insights for decision-making."
        
        return report
    
    async def _generate_creative_content(self, prompt: str, parameters: Dict[str, Any], 
                                       constraints: Dict[str, Any]) -> str:
        """Generate creative content"""
        style = parameters.get("style", "narrative")
        tone = parameters.get("tone", "engaging")
        
        if style == "narrative":
            content = f"Once upon a time, there was a fascinating concept called {prompt}. "
            content += "This idea captured the imagination of many, leading to discoveries and innovations. "
            content += f"The story of {prompt} is one of curiosity, exploration, and breakthrough moments.\n\n"
            content += "As we delve deeper into this topic, we uncover layers of meaning and significance. "
            content += "Each aspect reveals new possibilities and connections to other ideas.\n\n"
            content += f"The journey of understanding {prompt} continues to inspire and challenge us, "
            content += "opening doors to new perspectives and opportunities."
        
        elif style == "poetic":
            content = f"In the realm of knowledge vast and wide,\n"
            content += f"Lives a concept we call {prompt}.\n"
            content += f"With wisdom deep and insight bright,\n"
            content += f"It guides us through both day and night.\n\n"
            content += f"Oh {prompt}, mysterious and true,\n"
            content += f"What secrets do you hold for me and you?\n"
            content += f"In every question, every thought,\n"
            content += f"New understanding can be sought."
        
        else:
            content = f"Imagine a world where {prompt} plays a central role. "
            content += "This concept shapes how we think, act, and interact with our environment. "
            content += f"The beauty of {prompt} lies in its ability to connect different ideas and create new possibilities."
        
        return content
    
    async def _generate_technical_content(self, prompt: str, parameters: Dict[str, Any], 
                                        constraints: Dict[str, Any]) -> str:
        """Generate technical content"""
        include_code = parameters.get("include_code", False)
        technical_level = parameters.get("technical_level", "intermediate")
        
        content = f"# Technical Overview: {prompt}\n\n"
        content += f"## Introduction\n\n"
        content += f"{prompt} is a technical concept that requires careful consideration of implementation details. "
        content += "This document provides a comprehensive technical analysis.\n\n"
        
        content += f"## Technical Specifications\n\n"
        content += f"The implementation of {prompt} involves several key components:\n\n"
        content += "• **Architecture**: Modular design with clear interfaces\n"
        content += "• **Performance**: Optimized for efficiency and scalability\n"
        content += "• **Security**: Built-in security measures and best practices\n"
        content += "• **Maintainability**: Clean code structure and documentation\n\n"
        
        if include_code:
            content += f"## Code Example\n\n"
            content += "```python\n"
            content += f"class {prompt.replace(' ', '')}:\n"
            content += f"    def __init__(self):\n"
            content += f"        self.initialized = True\n"
            content += f"    \n"
            content += f"    def process(self, data):\n"
            content += f"        # Implementation logic here\n"
            content += f"        return processed_data\n"
            content += "```\n\n"
        
        content += f"## Best Practices\n\n"
        content += f"When working with {prompt}, consider the following best practices:\n\n"
        content += "1. **Error Handling**: Implement comprehensive error handling\n"
        content += "2. **Testing**: Write thorough unit and integration tests\n"
        content += "3. **Documentation**: Maintain clear and up-to-date documentation\n"
        content += "4. **Performance**: Monitor and optimize performance regularly\n"
        
        return content
    
    def _calculate_quality_metrics(self, content: str, request: GenerationRequest) -> Dict[str, float]:
        """Calculate quality metrics for generated content"""
        metrics = {}
        
        # Length metrics
        word_count = len(content.split())
        char_count = len(content)
        
        metrics["word_count"] = word_count
        metrics["character_count"] = char_count
        
        # Readability (simple approximation)
        sentences = content.count('.') + content.count('!') + content.count('?')
        if sentences > 0:
            avg_words_per_sentence = word_count / sentences
            metrics["avg_words_per_sentence"] = avg_words_per_sentence
            # Simple readability score (lower is better)
            metrics["readability_score"] = min(1.0, max(0.0, 1.0 - (avg_words_per_sentence - 15) / 20))
        else:
            metrics["readability_score"] = 0.5
        
        # Structure quality
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        metrics["paragraph_count"] = paragraphs
        metrics["structure_score"] = min(1.0, paragraphs / 3)  # Prefer 3+ paragraphs
        
        # Content relevance (simple keyword matching)
        prompt_words = set(request.prompt.lower().split())
        content_words = set(content.lower().split())
        relevance = len(prompt_words.intersection(content_words)) / len(prompt_words) if prompt_words else 0
        metrics["relevance_score"] = relevance
        
        # Overall quality score
        metrics["overall_quality"] = (
            metrics["readability_score"] * 0.3 +
            metrics["structure_score"] * 0.3 +
            metrics["relevance_score"] * 0.4
        )
        
        return metrics

# Register components
async def register_generation_components():
    """Register all generation components"""
    
    # Content Template Manager
    template_manager = ContentTemplateComponent()
    template_metadata = ComponentMetadata(
        component_id="content_template",
        component_type=ComponentType.SERVICE,
        name="Content Template Manager",
        description="Manages content templates and formatting patterns",
        version="1.0.0",
        interface_version="1.0.0",
        provides=["content_templates", "content_formatting"],
        tags=["templates", "formatting", "content"]
    )
    
    # Text Generator
    text_generator = TextGeneratorComponent()
    generator_metadata = ComponentMetadata(
        component_id="text_generator",
        component_type=ComponentType.GENERATOR,
        name="Text Generator",
        description="Generates various types of text content",
        version="1.0.0",
        interface_version="1.0.0",
        dependencies=["content_template"],
        provides=["text_generation"],
        requires=["content_templates"],
        tags=["generation", "text", "content", "nlp"]
    )
    
    # Register components
    component_registry.register_component(template_manager, template_metadata)
    component_registry.register_component(text_generator, generator_metadata)
    
    logger.info("Generation components registered successfully")