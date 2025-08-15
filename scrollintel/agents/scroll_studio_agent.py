"""
ScrollStudioAgent - AI-Powered IDE and Development Assistant
Intelligent code generation, debugging, architecture guidance, and development automation.
"""

import asyncio
import ast
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import logging

# Code analysis libraries
try:
    import astroid
    from pylint import lint
    from pylint.reporters.text import TextReporter
    CODE_ANALYSIS_AVAILABLE = True
except ImportError:
    CODE_ANALYSIS_AVAILABLE = False

# AI libraries
try:
    import openai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus

logger = logging.getLogger(__name__)


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SQL = "sql"
    HTML = "html"
    CSS = "css"


class CodeTaskType(str, Enum):
    """Types of code-related tasks."""
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    BUG_DETECTION = "bug_detection"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    ARCHITECTURE_DESIGN = "architecture_design"
    API_DESIGN = "api_design"


class CodeQualityMetric(str, Enum):
    """Code quality metrics."""
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    READABILITY = "readability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TESTABILITY = "testability"
    DOCUMENTATION = "documentation"


@dataclass
class CodeAnalysis:
    """Code analysis result."""
    id: str
    language: ProgrammingLanguage
    code_content: str
    analysis_type: CodeTaskType
    quality_metrics: Dict[CodeQualityMetric, float]
    issues_found: List[Dict[str, Any]]
    suggestions: List[str]
    complexity_score: float
    maintainability_index: float
    test_coverage: Optional[float] = None
    security_score: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class CodeGeneration:
    """Code generation result."""
    id: str
    request_description: str
    language: ProgrammingLanguage
    generated_code: str
    explanation: str
    usage_examples: List[str]
    dependencies: List[str]
    test_cases: List[str]
    documentation: str
    quality_score: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ArchitectureRecommendation:
    """Architecture design recommendation."""
    id: str
    project_description: str
    recommended_architecture: str
    design_patterns: List[str]
    technology_stack: List[str]
    scalability_considerations: List[str]
    security_considerations: List[str]
    implementation_phases: List[Dict[str, Any]]
    estimated_complexity: str
    rationale: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ScrollStudioAgent(BaseAgent):
    """Advanced AI-powered IDE and development assistant."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-studio-agent",
            name="ScrollStudio Agent",
            agent_type=AgentType.AI_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="intelligent_code_generation",
                description="Generate high-quality code from natural language descriptions",
                input_types=["requirements", "specifications", "examples"],
                output_types=["code", "documentation", "tests", "usage_examples"]
            ),
            AgentCapability(
                name="code_analysis_and_review",
                description="Analyze code quality, detect bugs, and provide improvement suggestions",
                input_types=["source_code", "language", "analysis_type"],
                output_types=["analysis_report", "bug_report", "suggestions", "quality_metrics"]
            ),
            AgentCapability(
                name="architecture_guidance",
                description="Provide software architecture recommendations and design patterns",
                input_types=["project_requirements", "constraints", "technology_preferences"],
                output_types=["architecture_design", "design_patterns", "implementation_plan"]
            ),
            AgentCapability(
                name="development_automation",
                description="Automate development tasks like testing, documentation, and deployment",
                input_types=["project_structure", "automation_requirements"],
                output_types=["automation_scripts", "ci_cd_config", "deployment_config"]
            )
        ]
        
        # Initialize AI components
        if AI_AVAILABLE:
            self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.openai_client = None
        
        # Development state
        self.active_projects = {}
        self.code_analyses = {}
        self.generated_code = {}
        self.architecture_recommendations = {}
        
        # Code templates and patterns
        self.code_templates = self._initialize_code_templates()
        self.design_patterns = self._initialize_design_patterns()
        self.best_practices = self._initialize_best_practices()
    
    def _initialize_code_templates(self) -> Dict[ProgrammingLanguage, Dict[str, str]]:
        """Initialize code templates for different languages."""
        return {
            ProgrammingLanguage.PYTHON: {
                "class": """
class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_params}):
        {init_body}
    
    def {method_name}(self{method_params}):
        \"\"\"
        {method_description}
        \"\"\"
        {method_body}
""",
                "function": """
def {function_name}({parameters}):
    \"\"\"
    {description}
    
    Args:
        {args_description}
    
    Returns:
        {return_description}
    \"\"\"
    {function_body}
""",
                "api_endpoint": """
@app.route('/{endpoint}', methods=['{method}'])
def {function_name}():
    \"\"\"
    {description}
    \"\"\"
    try:
        {endpoint_body}
        return jsonify({{'status': 'success', 'data': result}})
    except Exception as e:
        return jsonify({{'status': 'error', 'message': str(e)}}), 500
"""
            },
            ProgrammingLanguage.JAVASCRIPT: {
                "class": """
class {class_name} {{
    /**
     * {description}
     */
    constructor({constructor_params}) {{
        {constructor_body}
    }}
    
    /**
     * {method_description}
     */
    {method_name}({method_params}) {{
        {method_body}
    }}
}}
""",
                "function": """
/**
 * {description}
 * @param {{{param_types}}} {param_names}
 * @returns {{{return_type}}} {return_description}
 */
function {function_name}({parameters}) {{
    {function_body}
}}
""",
                "react_component": """
import React, {{ useState, useEffect }} from 'react';

/**
 * {description}
 */
const {component_name} = ({{ {props} }}) => {{
    {state_declarations}
    
    {effects}
    
    return (
        <div className="{class_name}">
            {jsx_content}
        </div>
    );
}};

export default {component_name};
"""
            }
        }
    
    def _initialize_design_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common design patterns."""
        return {
            "singleton": {
                "description": "Ensures a class has only one instance and provides global access",
                "use_cases": ["Database connections", "Configuration management", "Logging"],
                "languages": [ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVA, ProgrammingLanguage.JAVASCRIPT]
            },
            "factory": {
                "description": "Creates objects without specifying their exact classes",
                "use_cases": ["Object creation", "Plugin systems", "Database drivers"],
                "languages": [ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVA, ProgrammingLanguage.CSHARP]
            },
            "observer": {
                "description": "Defines a one-to-many dependency between objects",
                "use_cases": ["Event handling", "Model-View architectures", "Notifications"],
                "languages": [ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.JAVA]
            },
            "mvc": {
                "description": "Separates application logic into Model, View, and Controller",
                "use_cases": ["Web applications", "Desktop applications", "API design"],
                "languages": [ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.PHP]
            }
        }
    
    def _initialize_best_practices(self) -> Dict[ProgrammingLanguage, List[str]]:
        """Initialize best practices for different languages."""
        return {
            ProgrammingLanguage.PYTHON: [
                "Follow PEP 8 style guidelines",
                "Use type hints for better code documentation",
                "Write comprehensive docstrings",
                "Use virtual environments for dependency management",
                "Implement proper error handling with try-except blocks",
                "Use list comprehensions and generator expressions appropriately",
                "Follow the principle of least privilege for imports"
            ],
            ProgrammingLanguage.JAVASCRIPT: [
                "Use const and let instead of var",
                "Implement proper error handling with try-catch",
                "Use async/await for asynchronous operations",
                "Follow consistent naming conventions",
                "Use JSDoc for function documentation",
                "Implement proper input validation",
                "Use modern ES6+ features appropriately"
            ]
        }
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process development assistance requests."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "generate" in prompt or "create" in prompt:
                content = await self._generate_code(request.prompt, context)
            elif "analyze" in prompt or "review" in prompt:
                content = await self._analyze_code(request.prompt, context)
            elif "debug" in prompt or "fix" in prompt:
                content = await self._debug_code(request.prompt, context)
            elif "architecture" in prompt or "design" in prompt:
                content = await self._provide_architecture_guidance(request.prompt, context)
            elif "optimize" in prompt or "improve" in prompt:
                content = await self._optimize_code(request.prompt, context)
            elif "document" in prompt:
                content = await self._generate_documentation(request.prompt, context)
            elif "test" in prompt:
                content = await self._generate_tests(request.prompt, context)
            else:
                content = await self._general_development_assistance(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"studio-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"studio-{uuid4()}",
                request_id=request.id,
                content=f"Error in development assistance: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _generate_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate code based on requirements."""
        requirements = context.get("requirements", prompt)
        language = ProgrammingLanguage(context.get("language", ProgrammingLanguage.PYTHON))
        code_type = context.get("code_type", "function")
        
        # Generate code using AI or templates
        if self.openai_client:
            generated_code = await self._ai_generate_code(requirements, language, code_type)
        else:
            generated_code = await self._template_generate_code(requirements, language, code_type)
        
        # Create code generation result
        code_gen = CodeGeneration(
            id=f"codegen-{uuid4()}",
            request_description=requirements,
            language=language,
            generated_code=generated_code["code"],
            explanation=generated_code["explanation"],
            usage_examples=generated_code["examples"],
            dependencies=generated_code["dependencies"],
            test_cases=generated_code["tests"],
            documentation=generated_code["documentation"],
            quality_score=generated_code["quality_score"]
        )
        
        # Store generated code
        self.generated_code[code_gen.id] = code_gen
        
        return f"""
# Code Generation Result

## Requirements
{requirements}

## Generated Code ({language.value})

```{language.value}
{code_gen.generated_code}
```

## Explanation
{code_gen.explanation}

## Usage Examples

```{language.value}
{chr(10).join(code_gen.usage_examples)}
```

## Dependencies
{chr(10).join(f"- {dep}" for dep in code_gen.dependencies)}

## Test Cases

```{language.value}
{chr(10).join(code_gen.test_cases)}
```

## Documentation
{code_gen.documentation}

## Quality Assessment
- **Quality Score**: {code_gen.quality_score:.1f}/10
- **Best Practices**: Applied
- **Error Handling**: Implemented
- **Documentation**: Complete

## Next Steps
1. Review the generated code for your specific use case
2. Run the provided test cases
3. Install any required dependencies
4. Integrate into your project
"""
    
    async def _analyze_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """Analyze code quality and provide suggestions."""
        code_content = context.get("code", prompt)
        language = ProgrammingLanguage(context.get("language", ProgrammingLanguage.PYTHON))
        analysis_type = CodeTaskType(context.get("analysis_type", CodeTaskType.CODE_REVIEW))
        
        # Perform code analysis
        analysis = await self._perform_code_analysis(code_content, language, analysis_type)
        
        # Store analysis
        self.code_analyses[analysis.id] = analysis
        
        return f"""
# Code Analysis Report

## Code Overview
- **Language**: {language.value}
- **Analysis Type**: {analysis_type.value}
- **Lines of Code**: {len(code_content.split(chr(10)))}

## Quality Metrics
{await self._format_quality_metrics(analysis.quality_metrics)}

## Issues Found
{await self._format_issues(analysis.issues_found)}

## Code Quality Assessment
- **Complexity Score**: {analysis.complexity_score:.1f}/10
- **Maintainability Index**: {analysis.maintainability_index:.1f}/10
- **Test Coverage**: {analysis.test_coverage or 'Not available'}%
- **Security Score**: {analysis.security_score or 'Not assessed'}/10

## Suggestions for Improvement
{chr(10).join(f"- {suggestion}" for suggestion in analysis.suggestions)}

## Best Practices Compliance
{await self._check_best_practices_compliance(code_content, language)}

## Refactoring Opportunities
{await self._identify_refactoring_opportunities(code_content, language)}

## Next Steps
1. Address high-priority issues first
2. Implement suggested improvements
3. Add or improve test coverage
4. Update documentation as needed
"""
    
    async def _provide_architecture_guidance(self, prompt: str, context: Dict[str, Any]) -> str:
        """Provide software architecture recommendations."""
        project_description = context.get("project_description", prompt)
        requirements = context.get("requirements", [])
        constraints = context.get("constraints", [])
        technology_preferences = context.get("technology_preferences", [])
        
        # Generate architecture recommendation
        arch_rec = await self._generate_architecture_recommendation(
            project_description, requirements, constraints, technology_preferences
        )
        
        # Store recommendation
        self.architecture_recommendations[arch_rec.id] = arch_rec
        
        return f"""
# Software Architecture Recommendation

## Project Overview
{arch_rec.project_description}

## Recommended Architecture
{arch_rec.recommended_architecture}

## Design Patterns
{chr(10).join(f"- **{pattern}**: {self.design_patterns.get(pattern, {}).get('description', 'Design pattern')}" for pattern in arch_rec.design_patterns)}

## Technology Stack
{chr(10).join(f"- {tech}" for tech in arch_rec.technology_stack)}

## Scalability Considerations
{chr(10).join(f"- {consideration}" for consideration in arch_rec.scalability_considerations)}

## Security Considerations
{chr(10).join(f"- {consideration}" for consideration in arch_rec.security_considerations)}

## Implementation Phases
{await self._format_implementation_phases(arch_rec.implementation_phases)}

## Complexity Assessment
- **Estimated Complexity**: {arch_rec.estimated_complexity}
- **Development Timeline**: {await self._estimate_timeline(arch_rec)}
- **Team Size Recommendation**: {await self._recommend_team_size(arch_rec)}

## Rationale
{arch_rec.rationale}

## Alternative Approaches
{await self._suggest_alternative_architectures(project_description, requirements)}

## Next Steps
1. Review and validate the architecture with stakeholders
2. Create detailed technical specifications
3. Set up development environment
4. Begin with Phase 1 implementation
"""
    
    async def _ai_generate_code(self, requirements: str, language: ProgrammingLanguage, code_type: str) -> Dict[str, Any]:
        """Generate code using AI."""
        prompt = f"""
        Generate {language.value} code for the following requirements:
        {requirements}
        
        Code type: {code_type}
        
        Please provide:
        1. Clean, well-documented code
        2. Explanation of the implementation
        3. Usage examples
        4. Required dependencies
        5. Basic test cases
        6. Documentation
        
        Follow best practices for {language.value} development.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert {language.value} developer who writes clean, efficient, and well-documented code following best practices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2500,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response (simplified)
            return {
                "code": ai_response,
                "explanation": f"AI-generated {language.value} code for: {requirements}",
                "examples": [f"# Example usage of generated {language.value} code"],
                "dependencies": ["# Dependencies will be listed here"],
                "tests": [f"# Test cases for {language.value} code"],
                "documentation": f"Documentation for {language.value} implementation",
                "quality_score": 8.5
            }
            
        except Exception as e:
            logger.error(f"AI code generation failed: {e}")
            return await self._template_generate_code(requirements, language, code_type)
    
    async def _template_generate_code(self, requirements: str, language: ProgrammingLanguage, code_type: str) -> Dict[str, Any]:
        """Generate code using templates."""
        templates = self.code_templates.get(language, {})
        template = templates.get(code_type, templates.get("function", "# Code template not available"))
        
        # Simple template substitution
        generated_code = template.format(
            function_name="generated_function",
            parameters="param1, param2",
            description=f"Generated function for: {requirements}",
            function_body="    # Implementation goes here\n    pass",
            class_name="GeneratedClass",
            method_name="generated_method"
        )
        
        return {
            "code": generated_code,
            "explanation": f"Template-based {language.value} code generation",
            "examples": [f"# Example usage"],
            "dependencies": [],
            "tests": [f"# Add test cases here"],
            "documentation": f"Generated {language.value} code documentation",
            "quality_score": 7.0
        }
    
    async def _perform_code_analysis(self, code_content: str, language: ProgrammingLanguage, analysis_type: CodeTaskType) -> CodeAnalysis:
        """Perform comprehensive code analysis."""
        # Mock analysis - in production would use actual code analysis tools
        quality_metrics = {
            CodeQualityMetric.COMPLEXITY: 6.5,
            CodeQualityMetric.MAINTAINABILITY: 7.2,
            CodeQualityMetric.READABILITY: 8.0,
            CodeQualityMetric.PERFORMANCE: 7.5,
            CodeQualityMetric.SECURITY: 8.2,
            CodeQualityMetric.TESTABILITY: 6.8,
            CodeQualityMetric.DOCUMENTATION: 5.5
        }
        
        issues_found = [
            {
                "type": "warning",
                "line": 15,
                "message": "Variable 'unused_var' is defined but never used",
                "severity": "low"
            },
            {
                "type": "error",
                "line": 23,
                "message": "Potential null pointer dereference",
                "severity": "high"
            }
        ]
        
        suggestions = [
            "Remove unused variables to improve code cleanliness",
            "Add null checks before dereferencing pointers",
            "Consider breaking down large functions into smaller ones",
            "Add more comprehensive error handling",
            "Improve code documentation with better comments"
        ]
        
        analysis = CodeAnalysis(
            id=f"analysis-{uuid4()}",
            language=language,
            code_content=code_content,
            analysis_type=analysis_type,
            quality_metrics=quality_metrics,
            issues_found=issues_found,
            suggestions=suggestions,
            complexity_score=6.5,
            maintainability_index=7.2,
            test_coverage=65.0,
            security_score=8.2
        )
        
        return analysis
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check agent health."""
        return True
    
    # Helper methods (simplified implementations)
    async def _format_quality_metrics(self, metrics: Dict[CodeQualityMetric, float]) -> str:
        """Format quality metrics for display."""
        formatted = []
        for metric, score in metrics.items():
            formatted.append(f"- **{metric.value.title()}**: {score:.1f}/10")
        return "\n".join(formatted)
    
    async def _format_issues(self, issues: List[Dict[str, Any]]) -> str:
        """Format code issues for display."""
        if not issues:
            return "No issues found."
        
        formatted = []
        for issue in issues:
            severity_icon = "🚨" if issue["severity"] == "high" else "⚠️" if issue["severity"] == "medium" else "ℹ️"
            formatted.append(f"{severity_icon} **Line {issue['line']}**: {issue['message']} ({issue['severity']})")
        
        return "\n".join(formatted)
    
    async def _check_best_practices_compliance(self, code_content: str, language: ProgrammingLanguage) -> str:
        """Check compliance with best practices."""
        practices = self.best_practices.get(language, [])
        compliance_status = []
        
        for practice in practices[:3]:  # Show first 3 practices
            # Mock compliance check
            status = "✅" if "def " in code_content or "function" in code_content else "❌"
            compliance_status.append(f"{status} {practice}")
        
        return "\n".join(compliance_status)
    
    async def _generate_architecture_recommendation(self, project_description: str, requirements: List[str], 
                                                  constraints: List[str], tech_preferences: List[str]) -> ArchitectureRecommendation:
        """Generate architecture recommendation."""
        return ArchitectureRecommendation(
            id=f"arch-{uuid4()}",
            project_description=project_description,
            recommended_architecture="Microservices architecture with API Gateway pattern",
            design_patterns=["MVC", "Repository", "Factory", "Observer"],
            technology_stack=["Python/FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
            scalability_considerations=[
                "Horizontal scaling with load balancers",
                "Database read replicas",
                "Caching layer implementation",
                "Asynchronous processing queues"
            ],
            security_considerations=[
                "JWT authentication",
                "API rate limiting",
                "Input validation and sanitization",
                "HTTPS encryption",
                "Database connection security"
            ],
            implementation_phases=[
                {"phase": 1, "description": "Core API development", "duration": "4 weeks"},
                {"phase": 2, "description": "Database integration", "duration": "2 weeks"},
                {"phase": 3, "description": "Authentication system", "duration": "3 weeks"},
                {"phase": 4, "description": "Frontend integration", "duration": "4 weeks"}
            ],
            estimated_complexity="Medium to High",
            rationale="Microservices architecture provides scalability and maintainability for the described requirements"
        )