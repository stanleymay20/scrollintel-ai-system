"""
Security Training System
Provides comprehensive security training modules, progress tracking, and certification
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TrainingType(Enum):
    DEVELOPER = "developer"
    OPERATIONS = "operations"
    SECURITY = "security"
    EXECUTIVE = "executive"
    GENERAL = "general"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class TrainingModule:
    """Security training module definition"""
    id: str
    title: str
    description: str
    type: TrainingType
    difficulty: DifficultyLevel
    duration_minutes: int
    prerequisites: List[str]
    learning_objectives: List[str]
    content_sections: List[Dict[str, Any]]
    assessment_questions: List[Dict[str, Any]]
    passing_score: int
    certification_points: int
    mandatory: bool
    frequency_days: Optional[int]  # For recurring training
    created_date: datetime
    last_updated: datetime

@dataclass
class TrainingProgress:
    """User training progress tracking"""
    user_id: str
    module_id: str
    status: str  # not_started, in_progress, completed, failed
    start_date: Optional[datetime]
    completion_date: Optional[datetime]
    score: Optional[int]
    attempts: int
    time_spent_minutes: int
    last_accessed: datetime

@dataclass
class SecurityCertification:
    """Security certification tracking"""
    id: str
    name: str
    description: str
    required_modules: List[str]
    total_points_required: int
    validity_days: int
    renewal_required: bool

class SecurityTrainingSystem:
    """Comprehensive security training management system"""
    
    def __init__(self, training_path: str = "security/training"):
        self.training_path = Path(training_path)
        self.training_path.mkdir(parents=True, exist_ok=True)
        
        # Training data storage
        self.modules: Dict[str, TrainingModule] = {}
        self.user_progress: Dict[str, List[TrainingProgress]] = {}
        self.certifications: Dict[str, SecurityCertification] = {}
        
        self._load_training_data()
        self._initialize_default_modules()
    
    def _load_training_data(self):
        """Load training data from storage"""
        # Load modules
        modules_file = self.training_path / "modules.json"
        if modules_file.exists():
            with open(modules_file, 'r') as f:
                data = json.load(f)
                for module_id, module_data in data.items():
                    # Convert datetime strings
                    for date_field in ['created_date', 'last_updated']:
                        if module_data.get(date_field):
                            module_data[date_field] = datetime.fromisoformat(module_data[date_field])
                    
                    module_data['type'] = TrainingType(module_data['type'])
                    module_data['difficulty'] = DifficultyLevel(module_data['difficulty'])
                    self.modules[module_id] = TrainingModule(**module_data)
        
        # Load user progress
        progress_file = self.training_path / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                data = json.load(f)
                for user_id, progress_list in data.items():
                    user_progress = []
                    for progress_data in progress_list:
                        # Convert datetime strings
                        for date_field in ['start_date', 'completion_date', 'last_accessed']:
                            if progress_data.get(date_field):
                                progress_data[date_field] = datetime.fromisoformat(progress_data[date_field])
                        user_progress.append(TrainingProgress(**progress_data))
                    self.user_progress[user_id] = user_progress
        
        # Load certifications
        certs_file = self.training_path / "certifications.json"
        if certs_file.exists():
            with open(certs_file, 'r') as f:
                data = json.load(f)
                for cert_id, cert_data in data.items():
                    self.certifications[cert_id] = SecurityCertification(**cert_data)
    
    def _save_training_data(self):
        """Save training data to storage"""
        # Save modules
        modules_data = {}
        for module_id, module in self.modules.items():
            module_data = asdict(module)
            # Convert datetime objects to strings
            for date_field in ['created_date', 'last_updated']:
                if module_data.get(date_field):
                    module_data[date_field] = module_data[date_field].isoformat()
            module_data['type'] = module_data['type'].value
            module_data['difficulty'] = module_data['difficulty'].value
            modules_data[module_id] = module_data
        
        with open(self.training_path / "modules.json", 'w') as f:
            json.dump(modules_data, f, indent=2)
        
        # Save user progress
        progress_data = {}
        for user_id, progress_list in self.user_progress.items():
            user_progress_data = []
            for progress in progress_list:
                progress_dict = asdict(progress)
                # Convert datetime objects to strings
                for date_field in ['start_date', 'completion_date', 'last_accessed']:
                    if progress_dict.get(date_field):
                        progress_dict[date_field] = progress_dict[date_field].isoformat()
                user_progress_data.append(progress_dict)
            progress_data[user_id] = user_progress_data
        
        with open(self.training_path / "progress.json", 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Save certifications
        with open(self.training_path / "certifications.json", 'w') as f:
            json.dump({k: asdict(v) for k, v in self.certifications.items()}, f, indent=2)
    
    def _initialize_default_modules(self):
        """Initialize default security training modules"""
        if not self.modules:
            default_modules = [
                {
                    "id": "sec-basics-001",
                    "title": "Security Fundamentals",
                    "description": "Basic security principles and best practices",
                    "type": TrainingType.GENERAL,
                    "difficulty": DifficultyLevel.BEGINNER,
                    "duration_minutes": 45,
                    "prerequisites": [],
                    "learning_objectives": [
                        "Understand basic security principles",
                        "Identify common security threats",
                        "Apply security best practices"
                    ],
                    "content_sections": [
                        {
                            "title": "Introduction to Security",
                            "type": "video",
                            "content": "security_intro.mp4",
                            "duration": 15
                        },
                        {
                            "title": "Common Threats",
                            "type": "interactive",
                            "content": "threat_scenarios.html",
                            "duration": 20
                        },
                        {
                            "title": "Best Practices",
                            "type": "document",
                            "content": "security_best_practices.pdf",
                            "duration": 10
                        }
                    ],
                    "assessment_questions": [
                        {
                            "question": "What is the principle of least privilege?",
                            "type": "multiple_choice",
                            "options": [
                                "Give users maximum access",
                                "Give users minimum required access",
                                "Give users no access",
                                "Give users temporary access"
                            ],
                            "correct_answer": 1,
                            "points": 10
                        }
                    ],
                    "passing_score": 80,
                    "certification_points": 10,
                    "mandatory": True,
                    "frequency_days": 365
                },
                {
                    "id": "dev-secure-coding-001",
                    "title": "Secure Coding Practices",
                    "description": "Security best practices for developers",
                    "type": TrainingType.DEVELOPER,
                    "difficulty": DifficultyLevel.INTERMEDIATE,
                    "duration_minutes": 90,
                    "prerequisites": ["sec-basics-001"],
                    "learning_objectives": [
                        "Implement secure coding practices",
                        "Identify and prevent common vulnerabilities",
                        "Use security testing tools"
                    ],
                    "content_sections": [
                        {
                            "title": "OWASP Top 10",
                            "type": "interactive",
                            "content": "owasp_top10.html",
                            "duration": 30
                        },
                        {
                            "title": "Input Validation",
                            "type": "hands_on",
                            "content": "input_validation_lab.py",
                            "duration": 25
                        },
                        {
                            "title": "Authentication & Authorization",
                            "type": "video",
                            "content": "auth_security.mp4",
                            "duration": 20
                        },
                        {
                            "title": "Security Testing",
                            "type": "hands_on",
                            "content": "security_testing_lab.py",
                            "duration": 15
                        }
                    ],
                    "assessment_questions": [
                        {
                            "question": "Which of the following is NOT in the OWASP Top 10?",
                            "type": "multiple_choice",
                            "options": [
                                "SQL Injection",
                                "Cross-Site Scripting",
                                "Buffer Overflow",
                                "Broken Authentication"
                            ],
                            "correct_answer": 2,
                            "points": 15
                        }
                    ],
                    "passing_score": 85,
                    "certification_points": 25,
                    "mandatory": True,
                    "frequency_days": 180
                },
                {
                    "id": "ops-security-001",
                    "title": "Operations Security",
                    "description": "Security practices for operations teams",
                    "type": TrainingType.OPERATIONS,
                    "difficulty": DifficultyLevel.INTERMEDIATE,
                    "duration_minutes": 75,
                    "prerequisites": ["sec-basics-001"],
                    "learning_objectives": [
                        "Implement infrastructure security",
                        "Monitor security events",
                        "Respond to security incidents"
                    ],
                    "content_sections": [
                        {
                            "title": "Infrastructure Hardening",
                            "type": "hands_on",
                            "content": "infra_hardening_lab.yml",
                            "duration": 25
                        },
                        {
                            "title": "Security Monitoring",
                            "type": "interactive",
                            "content": "monitoring_dashboard.html",
                            "duration": 25
                        },
                        {
                            "title": "Incident Response",
                            "type": "simulation",
                            "content": "incident_response_sim.py",
                            "duration": 25
                        }
                    ],
                    "assessment_questions": [
                        {
                            "question": "What is the first step in incident response?",
                            "type": "multiple_choice",
                            "options": [
                                "Containment",
                                "Identification",
                                "Recovery",
                                "Lessons Learned"
                            ],
                            "correct_answer": 1,
                            "points": 15
                        }
                    ],
                    "passing_score": 85,
                    "certification_points": 20,
                    "mandatory": True,
                    "frequency_days": 180
                }
            ]
            
            for module_data in default_modules:
                module_data['created_date'] = datetime.now()
                module_data['last_updated'] = datetime.now()
                module = TrainingModule(**module_data)
                self.modules[module.id] = module
            
            self._save_training_data()
    
    def create_training_module(self, module_data: Dict[str, Any]) -> str:
        """Create a new training module"""
        try:
            module_id = module_data.get('id', str(uuid.uuid4()))
            module_data['id'] = module_id
            module_data['created_date'] = datetime.now()
            module_data['last_updated'] = datetime.now()
            
            # Convert string enums to enum objects
            module_data['type'] = TrainingType(module_data['type'])
            module_data['difficulty'] = DifficultyLevel(module_data['difficulty'])
            
            module = TrainingModule(**module_data)
            self.modules[module_id] = module
            self._save_training_data()
            
            logger.info(f"Created training module: {module_id}")
            return module_id
            
        except Exception as e:
            logger.error(f"Failed to create training module: {str(e)}")
            raise
    
    def start_training(self, user_id: str, module_id: str) -> bool:
        """Start training for a user"""
        try:
            if module_id not in self.modules:
                raise ValueError(f"Training module {module_id} not found")
            
            # Check if user already has progress for this module
            if user_id not in self.user_progress:
                self.user_progress[user_id] = []
            
            # Check for existing progress
            existing_progress = None
            for progress in self.user_progress[user_id]:
                if progress.module_id == module_id:
                    existing_progress = progress
                    break
            
            if existing_progress and existing_progress.status == "completed":
                # Check if renewal is needed
                module = self.modules[module_id]
                if module.frequency_days:
                    days_since_completion = (datetime.now() - existing_progress.completion_date).days
                    if days_since_completion < module.frequency_days:
                        logger.info(f"User {user_id} already completed module {module_id} recently")
                        return False
            
            # Create or update progress
            if existing_progress:
                existing_progress.status = "in_progress"
                existing_progress.start_date = datetime.now()
                existing_progress.last_accessed = datetime.now()
                existing_progress.attempts += 1
            else:
                progress = TrainingProgress(
                    user_id=user_id,
                    module_id=module_id,
                    status="in_progress",
                    start_date=datetime.now(),
                    completion_date=None,
                    score=None,
                    attempts=1,
                    time_spent_minutes=0,
                    last_accessed=datetime.now()
                )
                self.user_progress[user_id].append(progress)
            
            self._save_training_data()
            logger.info(f"Started training for user {user_id}, module {module_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def complete_assessment(self, user_id: str, module_id: str, answers: List[int]) -> Dict[str, Any]:
        """Complete training assessment and calculate score"""
        try:
            if module_id not in self.modules:
                raise ValueError(f"Training module {module_id} not found")
            
            module = self.modules[module_id]
            
            # Find user progress
            user_progress_list = self.user_progress.get(user_id, [])
            progress = None
            for p in user_progress_list:
                if p.module_id == module_id:
                    progress = p
                    break
            
            if not progress:
                raise ValueError(f"No training progress found for user {user_id}, module {module_id}")
            
            # Calculate score
            total_points = 0
            earned_points = 0
            
            for i, question in enumerate(module.assessment_questions):
                total_points += question.get('points', 10)
                if i < len(answers) and answers[i] == question.get('correct_answer'):
                    earned_points += question.get('points', 10)
            
            score = int((earned_points / total_points) * 100) if total_points > 0 else 0
            passed = score >= module.passing_score
            
            # Update progress
            progress.score = score
            progress.completion_date = datetime.now()
            progress.status = "completed" if passed else "failed"
            progress.last_accessed = datetime.now()
            
            self._save_training_data()
            
            result = {
                "passed": passed,
                "score": score,
                "passing_score": module.passing_score,
                "earned_points": earned_points,
                "total_points": total_points,
                "certification_points": module.certification_points if passed else 0
            }
            
            logger.info(f"Assessment completed for user {user_id}, module {module_id}: {score}%")
            return result
            
        except Exception as e:
            logger.error(f"Failed to complete assessment: {str(e)}")
            raise
    
    def get_user_training_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive training status for a user"""
        user_progress_list = self.user_progress.get(user_id, [])
        
        completed_modules = []
        in_progress_modules = []
        failed_modules = []
        total_certification_points = 0
        
        for progress in user_progress_list:
            module = self.modules.get(progress.module_id)
            if not module:
                continue
            
            progress_info = {
                "module_id": progress.module_id,
                "module_title": module.title,
                "status": progress.status,
                "score": progress.score,
                "completion_date": progress.completion_date.isoformat() if progress.completion_date else None,
                "attempts": progress.attempts
            }
            
            if progress.status == "completed":
                completed_modules.append(progress_info)
                total_certification_points += module.certification_points
            elif progress.status == "in_progress":
                in_progress_modules.append(progress_info)
            elif progress.status == "failed":
                failed_modules.append(progress_info)
        
        # Check mandatory training compliance
        mandatory_modules = [m for m in self.modules.values() if m.mandatory]
        completed_mandatory = [p.module_id for p in user_progress_list if p.status == "completed"]
        missing_mandatory = [m.id for m in mandatory_modules if m.id not in completed_mandatory]
        
        return {
            "user_id": user_id,
            "completed_modules": completed_modules,
            "in_progress_modules": in_progress_modules,
            "failed_modules": failed_modules,
            "total_certification_points": total_certification_points,
            "mandatory_compliance": len(missing_mandatory) == 0,
            "missing_mandatory_training": missing_mandatory,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_training_analytics(self) -> Dict[str, Any]:
        """Generate training analytics and reporting"""
        total_users = len(self.user_progress)
        total_modules = len(self.modules)
        
        # Calculate completion rates
        module_completion_rates = {}
        for module_id, module in self.modules.items():
            completed_count = 0
            total_attempts = 0
            
            for user_progress_list in self.user_progress.values():
                for progress in user_progress_list:
                    if progress.module_id == module_id:
                        total_attempts += 1
                        if progress.status == "completed":
                            completed_count += 1
            
            completion_rate = (completed_count / total_attempts * 100) if total_attempts > 0 else 0
            module_completion_rates[module_id] = {
                "module_title": module.title,
                "completion_rate": completion_rate,
                "total_attempts": total_attempts,
                "completed": completed_count
            }
        
        # Calculate compliance rates
        mandatory_modules = [m.id for m in self.modules.values() if m.mandatory]
        compliant_users = 0
        
        for user_id, user_progress_list in self.user_progress.items():
            completed_mandatory = [p.module_id for p in user_progress_list if p.status == "completed"]
            if all(m_id in completed_mandatory for m_id in mandatory_modules):
                compliant_users += 1
        
        compliance_rate = (compliant_users / total_users * 100) if total_users > 0 else 0
        
        return {
            "total_users": total_users,
            "total_modules": total_modules,
            "overall_compliance_rate": compliance_rate,
            "compliant_users": compliant_users,
            "module_completion_rates": module_completion_rates,
            "mandatory_modules_count": len(mandatory_modules),
            "report_generated": datetime.now().isoformat()
        }