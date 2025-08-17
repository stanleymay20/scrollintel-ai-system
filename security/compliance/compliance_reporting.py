"""
Automated Compliance Reporting for SOC 2 Type II, GDPR, HIPAA, and ISO 27001
Generates comprehensive compliance reports and evidence packages
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import hashlib
import uuid


class ComplianceFramework(Enum):
    SOC2_TYPE_II = "soc2_type_ii"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO_27001 = "iso_27001"
    PCI_DSS = "pci_dss"
    NIST = "nist"


class ControlStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    status: ControlStatus
    evidence: List[str]
    last_assessment: datetime
    next_assessment: datetime
    responsible_party: str
    implementation_notes: str
    risk_level: str
    automated_check: bool
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['framework'] = self.framework.value
        data['status'] = self.status.value
        data['last_assessment'] = self.last_assessment.isoformat()
        data['next_assessment'] = self.next_assessment.isoformat()
        return data


@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    framework: ComplianceFramework
    assessment_date: datetime
    reporting_period_start: datetime
    reporting_period_end: datetime
    overall_status: str
    compliance_score: float
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    controls: List[ComplianceControl]
    recommendations: List[str]
    evidence_package: Dict[str, Any]
    assessor: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['framework'] = self.framework.value
        data['assessment_date'] = self.assessment_date.isoformat()
        data['reporting_period_start'] = self.reporting_period_start.isoformat()
        data['reporting_period_end'] = self.reporting_period_end.isoformat()
        data['controls'] = [control.to_dict() for control in self.controls]
        return data


class ComplianceReportingEngine:
    """
    Automated compliance reporting engine for multiple frameworks
    """
    
    def __init__(self, db_path: str = "security/compliance.db"):
        self.db_path = db_path
        self._init_database()
        self._load_control_frameworks()
    
    def _init_database(self):
        """Initialize compliance database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_controls (
                    control_id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    requirement TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    last_assessment DATETIME NOT NULL,
                    next_assessment DATETIME NOT NULL,
                    responsible_party TEXT NOT NULL,
                    implementation_notes TEXT,
                    risk_level TEXT NOT NULL,
                    automated_check BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    assessment_date DATETIME NOT NULL,
                    reporting_period_start DATETIME NOT NULL,
                    reporting_period_end DATETIME NOT NULL,
                    overall_status TEXT NOT NULL,
                    compliance_score REAL NOT NULL,
                    total_controls INTEGER NOT NULL,
                    compliant_controls INTEGER NOT NULL,
                    non_compliant_controls INTEGER NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    evidence_package_json TEXT NOT NULL,
                    assessor TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    control_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    file_path TEXT,
                    hash_value TEXT,
                    collected_date DATETIME NOT NULL,
                    collector TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (control_id) REFERENCES compliance_controls (control_id)
                )
            """)
    
    def _load_control_frameworks(self):
        """Load predefined control frameworks"""
        frameworks = {
            ComplianceFramework.SOC2_TYPE_II: self._get_soc2_controls(),
            ComplianceFramework.GDPR: self._get_gdpr_controls(),
            ComplianceFramework.HIPAA: self._get_hipaa_controls(),
            ComplianceFramework.ISO_27001: self._get_iso27001_controls()
        }
        
        for framework, controls in frameworks.items():
            for control in controls:
                self._upsert_control(control)
    
    def _get_soc2_controls(self) -> List[ComplianceControl]:
        """Get SOC 2 Type II controls"""
        return [
            ComplianceControl(
                control_id="SOC2-CC1.1",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Control Environment - Integrity and Ethical Values",
                description="The entity demonstrates a commitment to integrity and ethical values",
                requirement="Establish and maintain policies and procedures for integrity and ethical values",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="CISO",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=False
            ),
            ComplianceControl(
                control_id="SOC2-CC2.1",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Communication and Information - Internal Communication",
                description="The entity internally communicates information necessary to support the functioning of internal control",
                requirement="Establish communication channels and procedures for internal control information",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="IT Manager",
                implementation_notes="",
                risk_level="MEDIUM",
                automated_check=True
            ),
            ComplianceControl(
                control_id="SOC2-CC3.1",
                framework=ComplianceFramework.SOC2_TYPE_II,
                title="Risk Assessment - Objectives",
                description="The entity specifies objectives with sufficient clarity to enable identification and assessment of risks",
                requirement="Define clear security objectives and risk assessment procedures",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="Risk Manager",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=False
            )
        ]
    
    def _get_gdpr_controls(self) -> List[ComplianceControl]:
        """Get GDPR controls"""
        return [
            ComplianceControl(
                control_id="GDPR-ART5",
                framework=ComplianceFramework.GDPR,
                title="Principles of Processing Personal Data",
                description="Personal data shall be processed lawfully, fairly and transparently",
                requirement="Implement lawful basis for processing and transparency measures",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="DPO",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=True
            ),
            ComplianceControl(
                control_id="GDPR-ART25",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design and by Default",
                description="Implement appropriate technical and organizational measures for data protection",
                requirement="Implement privacy by design principles in all systems",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="Engineering Lead",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=True
            ),
            ComplianceControl(
                control_id="GDPR-ART32",
                framework=ComplianceFramework.GDPR,
                title="Security of Processing",
                description="Implement appropriate technical and organizational measures to ensure security",
                requirement="Implement encryption, access controls, and security monitoring",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="CISO",
                implementation_notes="",
                risk_level="CRITICAL",
                automated_check=True
            )
        ]
    
    def _get_hipaa_controls(self) -> List[ComplianceControl]:
        """Get HIPAA controls"""
        return [
            ComplianceControl(
                control_id="HIPAA-164.308",
                framework=ComplianceFramework.HIPAA,
                title="Administrative Safeguards",
                description="Implement administrative safeguards for PHI protection",
                requirement="Establish security officer, workforce training, and access management",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="Privacy Officer",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=False
            ),
            ComplianceControl(
                control_id="HIPAA-164.310",
                framework=ComplianceFramework.HIPAA,
                title="Physical Safeguards",
                description="Implement physical safeguards for PHI protection",
                requirement="Control physical access to systems containing PHI",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="Facilities Manager",
                implementation_notes="",
                risk_level="MEDIUM",
                automated_check=True
            ),
            ComplianceControl(
                control_id="HIPAA-164.312",
                framework=ComplianceFramework.HIPAA,
                title="Technical Safeguards",
                description="Implement technical safeguards for PHI protection",
                requirement="Implement access controls, audit controls, integrity, and transmission security",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="CISO",
                implementation_notes="",
                risk_level="CRITICAL",
                automated_check=True
            )
        ]
    
    def _get_iso27001_controls(self) -> List[ComplianceControl]:
        """Get ISO 27001 controls"""
        return [
            ComplianceControl(
                control_id="ISO27001-A.5.1.1",
                framework=ComplianceFramework.ISO_27001,
                title="Information Security Policies",
                description="A set of policies for information security shall be defined",
                requirement="Establish and maintain information security policies",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="CISO",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=False
            ),
            ComplianceControl(
                control_id="ISO27001-A.9.1.1",
                framework=ComplianceFramework.ISO_27001,
                title="Access Control Policy",
                description="An access control policy shall be established and reviewed",
                requirement="Implement comprehensive access control policies and procedures",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="IAM Team",
                implementation_notes="",
                risk_level="HIGH",
                automated_check=True
            ),
            ComplianceControl(
                control_id="ISO27001-A.10.1.1",
                framework=ComplianceFramework.ISO_27001,
                title="Cryptographic Controls",
                description="A policy on the use of cryptographic controls shall be developed",
                requirement="Implement cryptographic controls for data protection",
                status=ControlStatus.UNDER_REVIEW,
                evidence=[],
                last_assessment=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=90),
                responsible_party="Crypto Team",
                implementation_notes="",
                risk_level="CRITICAL",
                automated_check=True
            )
        ]
    
    def _upsert_control(self, control: ComplianceControl):
        """Insert or update a compliance control"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO compliance_controls 
                (control_id, framework, title, description, requirement, status, 
                 evidence_json, last_assessment, next_assessment, responsible_party,
                 implementation_notes, risk_level, automated_check, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                control.control_id,
                control.framework.value,
                control.title,
                control.description,
                control.requirement,
                control.status.value,
                json.dumps(control.evidence),
                control.last_assessment,
                control.next_assessment,
                control.responsible_party,
                control.implementation_notes,
                control.risk_level,
                control.automated_check,
                datetime.utcnow()
            ))
    
    def generate_compliance_report(self, 
                                 framework: ComplianceFramework,
                                 assessor: str = "System",
                                 period_days: int = 90) -> ComplianceReport:
        """
        Generate comprehensive compliance report for specified framework
        """
        report_id = str(uuid.uuid4())
        assessment_date = datetime.utcnow()
        period_start = assessment_date - timedelta(days=period_days)
        
        # Get all controls for framework
        controls = self._get_controls_by_framework(framework)
        
        # Calculate compliance metrics
        total_controls = len(controls)
        compliant_controls = len([c for c in controls if c.status == ControlStatus.COMPLIANT])
        non_compliant_controls = len([c for c in controls if c.status == ControlStatus.NON_COMPLIANT])
        
        compliance_score = (compliant_controls / total_controls * 100) if total_controls > 0 else 0
        
        # Determine overall status
        if compliance_score >= 95:
            overall_status = "FULLY_COMPLIANT"
        elif compliance_score >= 80:
            overall_status = "SUBSTANTIALLY_COMPLIANT"
        elif compliance_score >= 60:
            overall_status = "PARTIALLY_COMPLIANT"
        else:
            overall_status = "NON_COMPLIANT"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(controls)
        
        # Generate evidence package
        evidence_package = self._generate_evidence_package(controls, period_start, assessment_date)
        
        report = ComplianceReport(
            report_id=report_id,
            framework=framework,
            assessment_date=assessment_date,
            reporting_period_start=period_start,
            reporting_period_end=assessment_date,
            overall_status=overall_status,
            compliance_score=compliance_score,
            total_controls=total_controls,
            compliant_controls=compliant_controls,
            non_compliant_controls=non_compliant_controls,
            controls=controls,
            recommendations=recommendations,
            evidence_package=evidence_package,
            assessor=assessor
        )
        
        # Store report
        self._store_report(report)
        
        return report
    
    def _get_controls_by_framework(self, framework: ComplianceFramework) -> List[ComplianceControl]:
        """Get all controls for a specific framework"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT control_id, framework, title, description, requirement, status,
                       evidence_json, last_assessment, next_assessment, responsible_party,
                       implementation_notes, risk_level, automated_check
                FROM compliance_controls 
                WHERE framework = ?
                ORDER BY control_id
            """, (framework.value,))
            
            controls = []
            for row in cursor.fetchall():
                control = ComplianceControl(
                    control_id=row[0],
                    framework=ComplianceFramework(row[1]),
                    title=row[2],
                    description=row[3],
                    requirement=row[4],
                    status=ControlStatus(row[5]),
                    evidence=json.loads(row[6]),
                    last_assessment=datetime.fromisoformat(row[7]),
                    next_assessment=datetime.fromisoformat(row[8]),
                    responsible_party=row[9],
                    implementation_notes=row[10] or "",
                    risk_level=row[11],
                    automated_check=bool(row[12])
                )
                controls.append(control)
            
            return controls
    
    def _generate_recommendations(self, controls: List[ComplianceControl]) -> List[str]:
        """Generate compliance recommendations based on control status"""
        recommendations = []
        
        non_compliant = [c for c in controls if c.status == ControlStatus.NON_COMPLIANT]
        partially_compliant = [c for c in controls if c.status == ControlStatus.PARTIALLY_COMPLIANT]
        
        if non_compliant:
            recommendations.append(
                f"Address {len(non_compliant)} non-compliant controls immediately: " +
                ", ".join([c.control_id for c in non_compliant[:5]])
            )
        
        if partially_compliant:
            recommendations.append(
                f"Complete implementation of {len(partially_compliant)} partially compliant controls: " +
                ", ".join([c.control_id for c in partially_compliant[:5]])
            )
        
        # Risk-based recommendations
        critical_controls = [c for c in controls if c.risk_level == "CRITICAL" and c.status != ControlStatus.COMPLIANT]
        if critical_controls:
            recommendations.append(
                f"Prioritize {len(critical_controls)} critical risk controls for immediate remediation"
            )
        
        # Automation recommendations
        manual_controls = [c for c in controls if not c.automated_check and c.status == ControlStatus.COMPLIANT]
        if len(manual_controls) > 5:
            recommendations.append(
                f"Consider automating {len(manual_controls)} manual compliance checks to reduce audit burden"
            )
        
        return recommendations
    
    def _generate_evidence_package(self, 
                                 controls: List[ComplianceControl],
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate evidence package for compliance report"""
        evidence_package = {
            'generation_date': datetime.utcnow().isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'evidence_summary': {},
            'automated_evidence': {},
            'manual_evidence': {},
            'evidence_gaps': []
        }
        
        for control in controls:
            control_evidence = {
                'control_id': control.control_id,
                'evidence_count': len(control.evidence),
                'evidence_items': control.evidence,
                'automated_checks': control.automated_check,
                'last_assessment': control.last_assessment.isoformat()
            }
            
            if control.automated_check:
                evidence_package['automated_evidence'][control.control_id] = control_evidence
            else:
                evidence_package['manual_evidence'][control.control_id] = control_evidence
            
            # Identify evidence gaps
            if len(control.evidence) == 0:
                evidence_package['evidence_gaps'].append({
                    'control_id': control.control_id,
                    'title': control.title,
                    'risk_level': control.risk_level
                })
        
        evidence_package['evidence_summary'] = {
            'total_controls': len(controls),
            'automated_controls': len([c for c in controls if c.automated_check]),
            'manual_controls': len([c for c in controls if not c.automated_check]),
            'controls_with_evidence': len([c for c in controls if c.evidence]),
            'evidence_gaps': len(evidence_package['evidence_gaps'])
        }
        
        return evidence_package
    
    def _store_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO compliance_reports 
                (report_id, framework, assessment_date, reporting_period_start, 
                 reporting_period_end, overall_status, compliance_score, total_controls,
                 compliant_controls, non_compliant_controls, recommendations_json,
                 evidence_package_json, assessor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.framework.value,
                report.assessment_date,
                report.reporting_period_start,
                report.reporting_period_end,
                report.overall_status,
                report.compliance_score,
                report.total_controls,
                report.compliant_controls,
                report.non_compliant_controls,
                json.dumps(report.recommendations),
                json.dumps(report.evidence_package),
                report.assessor
            ))
    
    def update_control_status(self, 
                            control_id: str, 
                            status: ControlStatus,
                            evidence: Optional[List[str]] = None,
                            notes: Optional[str] = None):
        """Update compliance control status and evidence"""
        with sqlite3.connect(self.db_path) as conn:
            if evidence is not None:
                conn.execute("""
                    UPDATE compliance_controls 
                    SET status = ?, evidence_json = ?, implementation_notes = ?,
                        last_assessment = ?, updated_at = ?
                    WHERE control_id = ?
                """, (
                    status.value,
                    json.dumps(evidence),
                    notes or "",
                    datetime.utcnow(),
                    datetime.utcnow(),
                    control_id
                ))
            else:
                conn.execute("""
                    UPDATE compliance_controls 
                    SET status = ?, implementation_notes = ?,
                        last_assessment = ?, updated_at = ?
                    WHERE control_id = ?
                """, (
                    status.value,
                    notes or "",
                    datetime.utcnow(),
                    datetime.utcnow(),
                    control_id
                ))
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        dashboard = {
            'frameworks': {},
            'overall_metrics': {
                'total_controls': 0,
                'compliant_controls': 0,
                'non_compliant_controls': 0,
                'average_compliance_score': 0
            },
            'recent_assessments': [],
            'upcoming_assessments': []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Framework-specific metrics
            for framework in ComplianceFramework:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'compliant' THEN 1 ELSE 0 END) as compliant,
                        SUM(CASE WHEN status = 'non_compliant' THEN 1 ELSE 0 END) as non_compliant
                    FROM compliance_controls 
                    WHERE framework = ?
                """, (framework.value,))
                
                result = cursor.fetchone()
                total, compliant, non_compliant = result
                
                # Handle None values
                total = total or 0
                compliant = compliant or 0
                non_compliant = non_compliant or 0
                
                compliance_score = (compliant / total * 100) if total > 0 else 0
                
                dashboard['frameworks'][framework.value] = {
                    'total_controls': total,
                    'compliant_controls': compliant,
                    'non_compliant_controls': non_compliant,
                    'compliance_score': compliance_score
                }
                
                dashboard['overall_metrics']['total_controls'] += total
                dashboard['overall_metrics']['compliant_controls'] += compliant
                dashboard['overall_metrics']['non_compliant_controls'] += non_compliant
            
            # Calculate average compliance score
            if dashboard['overall_metrics']['total_controls'] > 0:
                dashboard['overall_metrics']['average_compliance_score'] = (
                    dashboard['overall_metrics']['compliant_controls'] / 
                    dashboard['overall_metrics']['total_controls'] * 100
                )
            
            # Recent reports
            cursor = conn.execute("""
                SELECT report_id, framework, assessment_date, overall_status, compliance_score
                FROM compliance_reports 
                ORDER BY assessment_date DESC 
                LIMIT 10
            """)
            
            dashboard['recent_assessments'] = [
                {
                    'report_id': row[0],
                    'framework': row[1],
                    'assessment_date': row[2],
                    'overall_status': row[3],
                    'compliance_score': row[4]
                }
                for row in cursor.fetchall()
            ]
            
            # Upcoming assessments
            cursor = conn.execute("""
                SELECT control_id, framework, title, next_assessment, responsible_party
                FROM compliance_controls 
                WHERE next_assessment <= ?
                ORDER BY next_assessment ASC 
                LIMIT 10
            """, (datetime.utcnow() + timedelta(days=30),))
            
            dashboard['upcoming_assessments'] = [
                {
                    'control_id': row[0],
                    'framework': row[1],
                    'title': row[2],
                    'next_assessment': row[3],
                    'responsible_party': row[4]
                }
                for row in cursor.fetchall()
            ]
        
        return dashboard