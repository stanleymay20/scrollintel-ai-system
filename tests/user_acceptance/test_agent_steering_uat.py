"""
User Acceptance Tests for Agent Steering System
Tests real business scenarios with actual stakeholder workflows.
"""

import pytest
import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class UserAcceptanceTestSuite:
    """Comprehensive user acceptance test suite for Agent Steering System"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}
        self.business_scenarios = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all user acceptance tests"""
        logger.info("🧪 Starting User Acceptance Test Suite for Agent Steering System")
        
        test_methods = [
            self.test_executive_dashboard_scenario,
            self.test_market_analysis_workflow,
            self.test_real_time_decision_making,
            self.test_multi_agent_collaboration,
            self.test_business_intelligence_queries,
            self.test_compliance_and_audit_trail,
            self.test_performance_under_load,
            self.test_security_and_access_control,
            self.test_mobile_and_responsive_interface,
            self.test_integration_with_existing_systems
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                self.test_results[test_method.__name__] = result
                logger.info(f"✅ {test_method.__name__}: {'PASSED' if result['success'] else 'FAILED'}")
            except Exception as e:
                self.test_results[test_method.__name__] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"❌ {test_method.__name__}: FAILED - {str(e)}")
        
        return self._generate_test_report()
    
    def test_executive_dashboard_scenario(self) -> Dict[str, Any]:
        """Test: Executive Dashboard Real-Time Business Metrics"""
        scenario = {
            "name": "Executive Dashboard Scenario",
            "description": "CEO needs real-time business metrics for board meeting",
            "user_persona": "Chief Executive Officer",
            "business_value": "Strategic decision making with real-time data"
        }
        
        try:
            # Step 1: Access executive dashboard
            dashboard_response = requests.get(f"{self.base_url}/api/dashboard/executive", timeout=10)
            assert dashboard_response.status_code == 200, "Executive dashboard not accessible"
            
            dashboard_data = dashboard_response.json()
            
            # Step 2: Verify key metrics are present
            required_metrics = [
                "revenue_trend", "customer_acquisition", "operational_efficiency",
                "market_share", "risk_indicators", "growth_projections"
            ]
            
            for metric in required_metrics:
                assert metric in dashboard_data, f"Missing critical metric: {metric}"
            
            # Step 3: Test real-time data updates
            initial_timestamp = dashboard_data.get("last_updated")
            time.sleep(5)  # Wait for data refresh
            
            updated_response = requests.get(f"{self.base_url}/api/dashboard/executive", timeout=10)
            updated_data = updated_response.json()
            updated_timestamp = updated_data.get("last_updated")
            
            assert updated_timestamp != initial_timestamp, "Dashboard data not updating in real-time"
            
            # Step 4: Test drill-down capabilities
            drill_down_response = requests.get(
                f"{self.base_url}/api/dashboard/executive/revenue_trend/details",
                timeout=10
            )
            assert drill_down_response.status_code == 200, "Drill-down functionality not working"
            
            return {
                "success": True,
                "scenario": scenario,
                "metrics_count": len(dashboard_data),
                "response_time": dashboard_response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_market_analysis_workflow(self) -> Dict[str, Any]:
        """Test: Market Analysis Workflow with Multiple Agents"""
        scenario = {
            "name": "Market Analysis Workflow",
            "description": "Marketing director requests comprehensive market analysis",
            "user_persona": "Marketing Director",
            "business_value": "Data-driven market strategy development"
        }
        
        try:
            # Step 1: Submit market analysis request
            analysis_request = {
                "title": "Q4 Market Analysis for Product Launch",
                "description": "Comprehensive analysis for new product launch strategy",
                "priority": "high",
                "requirements": {
                    "market_segments": ["enterprise", "mid-market", "smb"],
                    "competitors": ["competitor_a", "competitor_b", "competitor_c"],
                    "timeframe": "Q4 2024",
                    "deliverables": ["market_size", "competitive_landscape", "pricing_strategy", "go_to_market_plan"]
                }
            }
            
            request_response = requests.post(
                f"{self.base_url}/api/orchestration/tasks",
                json=analysis_request,
                timeout=30
            )
            assert request_response.status_code == 201, "Failed to submit analysis request"
            
            task_id = request_response.json()["task_id"]
            
            # Step 2: Monitor task progress
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status_response = requests.get(
                    f"{self.base_url}/api/orchestration/tasks/{task_id}/status",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data["status"]
                    
                    if status == "completed":
                        break
                    elif status == "failed":
                        raise Exception(f"Task failed: {status_data.get('error', 'Unknown error')}")
                
                time.sleep(10)
            else:
                raise Exception("Task did not complete within expected time")
            
            # Step 3: Retrieve and validate results
            results_response = requests.get(
                f"{self.base_url}/api/orchestration/tasks/{task_id}/results",
                timeout=10
            )
            assert results_response.status_code == 200, "Failed to retrieve task results"
            
            results = results_response.json()
            
            # Validate deliverables
            for deliverable in analysis_request["requirements"]["deliverables"]:
                assert deliverable in results, f"Missing deliverable: {deliverable}"
            
            # Step 4: Test collaboration features
            collaboration_response = requests.get(
                f"{self.base_url}/api/orchestration/tasks/{task_id}/collaboration",
                timeout=10
            )
            assert collaboration_response.status_code == 200, "Collaboration features not accessible"
            
            return {
                "success": True,
                "scenario": scenario,
                "task_id": task_id,
                "completion_time": time.time() - start_time,
                "deliverables_count": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_real_time_decision_making(self) -> Dict[str, Any]:
        """Test: Real-Time Decision Making with Intelligence Engine"""
        scenario = {
            "name": "Real-Time Decision Making",
            "description": "Operations manager needs immediate decision support for supply chain issue",
            "user_persona": "Operations Manager",
            "business_value": "Rapid response to operational challenges"
        }
        
        try:
            # Step 1: Submit urgent decision request
            decision_request = {
                "context": "supply_chain_disruption",
                "urgency": "critical",
                "description": "Major supplier experiencing delays, need alternative sourcing strategy",
                "constraints": {
                    "budget_limit": 500000,
                    "timeline": "48_hours",
                    "quality_requirements": "iso_certified"
                },
                "options": [
                    {"supplier": "alternative_a", "cost": 450000, "timeline": "36_hours"},
                    {"supplier": "alternative_b", "cost": 380000, "timeline": "72_hours"},
                    {"supplier": "alternative_c", "cost": 520000, "timeline": "24_hours"}
                ]
            }
            
            decision_response = requests.post(
                f"{self.base_url}/api/intelligence/decision",
                json=decision_request,
                timeout=30
            )
            assert decision_response.status_code == 200, "Failed to get decision recommendation"
            
            decision_data = decision_response.json()
            
            # Step 2: Validate decision quality
            required_fields = ["recommended_option", "confidence_score", "risk_assessment", "reasoning"]
            for field in required_fields:
                assert field in decision_data, f"Missing decision field: {field}"
            
            assert decision_data["confidence_score"] >= 0.7, "Decision confidence too low"
            
            # Step 3: Test real-time updates
            update_request = {
                "new_information": "Alternative supplier A just confirmed 24-hour delivery capability"
            }
            
            update_response = requests.put(
                f"{self.base_url}/api/intelligence/decision/{decision_data['decision_id']}/update",
                json=update_request,
                timeout=10
            )
            assert update_response.status_code == 200, "Failed to update decision with new information"
            
            # Step 4: Validate decision tracking
            tracking_response = requests.get(
                f"{self.base_url}/api/intelligence/decision/{decision_data['decision_id']}/tracking",
                timeout=10
            )
            assert tracking_response.status_code == 200, "Decision tracking not available"
            
            return {
                "success": True,
                "scenario": scenario,
                "decision_id": decision_data["decision_id"],
                "confidence_score": decision_data["confidence_score"],
                "response_time": decision_response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_multi_agent_collaboration(self) -> Dict[str, Any]:
        """Test: Multi-Agent Collaboration on Complex Business Task"""
        scenario = {
            "name": "Multi-Agent Collaboration",
            "description": "Complex project requiring coordination between multiple specialized agents",
            "user_persona": "Project Manager",
            "business_value": "Efficient coordination of specialized expertise"
        }
        
        try:
            # Step 1: Create collaboration session
            collaboration_request = {
                "project_name": "Digital Transformation Initiative",
                "objective": "Develop comprehensive digital transformation roadmap",
                "required_agents": ["cto_agent", "data_scientist", "business_analyst", "security_expert"],
                "timeline": "2_weeks",
                "deliverables": ["technical_assessment", "data_strategy", "business_case", "security_framework"]
            }
            
            session_response = requests.post(
                f"{self.base_url}/api/collaboration/sessions",
                json=collaboration_request,
                timeout=30
            )
            assert session_response.status_code == 201, "Failed to create collaboration session"
            
            session_id = session_response.json()["session_id"]
            
            # Step 2: Verify agent participation
            participants_response = requests.get(
                f"{self.base_url}/api/collaboration/sessions/{session_id}/participants",
                timeout=10
            )
            assert participants_response.status_code == 200, "Failed to get session participants"
            
            participants = participants_response.json()["participants"]
            assert len(participants) >= 4, "Not all required agents joined the session"
            
            # Step 3: Test real-time communication
            message_data = {
                "sender": "project_manager",
                "message": "Please provide initial assessment of current state",
                "priority": "normal"
            }
            
            message_response = requests.post(
                f"{self.base_url}/api/collaboration/sessions/{session_id}/messages",
                json=message_data,
                timeout=10
            )
            assert message_response.status_code == 201, "Failed to send collaboration message"
            
            # Step 4: Monitor progress
            time.sleep(30)  # Allow time for agents to respond
            
            progress_response = requests.get(
                f"{self.base_url}/api/collaboration/sessions/{session_id}/progress",
                timeout=10
            )
            assert progress_response.status_code == 200, "Failed to get collaboration progress"
            
            progress_data = progress_response.json()
            assert progress_data["progress_percentage"] > 0, "No progress made in collaboration"
            
            return {
                "success": True,
                "scenario": scenario,
                "session_id": session_id,
                "participants_count": len(participants),
                "progress_percentage": progress_data["progress_percentage"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_business_intelligence_queries(self) -> Dict[str, Any]:
        """Test: Natural Language Business Intelligence Queries"""
        scenario = {
            "name": "Business Intelligence Queries",
            "description": "Business analyst performs complex data analysis using natural language",
            "user_persona": "Business Analyst",
            "business_value": "Accessible data insights without technical expertise"
        }
        
        try:
            # Test various business intelligence queries
            queries = [
                "What are our top 5 performing products by revenue this quarter?",
                "Show me customer churn rate trends over the last 6 months",
                "Compare our market share with competitors in the enterprise segment",
                "What factors are driving customer satisfaction scores?",
                "Predict revenue for next quarter based on current pipeline"
            ]
            
            query_results = []
            
            for query in queries:
                query_request = {
                    "query": query,
                    "context": "business_analysis",
                    "format": "executive_summary"
                }
                
                query_response = requests.post(
                    f"{self.base_url}/api/intelligence/query",
                    json=query_request,
                    timeout=30
                )
                
                assert query_response.status_code == 200, f"Query failed: {query}"
                
                result = query_response.json()
                assert "insights" in result, f"No insights returned for query: {query}"
                assert "confidence" in result, f"No confidence score for query: {query}"
                
                query_results.append({
                    "query": query,
                    "confidence": result["confidence"],
                    "response_time": query_response.elapsed.total_seconds()
                })
            
            # Test query refinement
            refinement_request = {
                "original_query": queries[0],
                "refinement": "Focus on products launched in the last 12 months"
            }
            
            refinement_response = requests.post(
                f"{self.base_url}/api/intelligence/query/refine",
                json=refinement_request,
                timeout=20
            )
            assert refinement_response.status_code == 200, "Query refinement failed"
            
            return {
                "success": True,
                "scenario": scenario,
                "queries_tested": len(queries),
                "average_confidence": sum(r["confidence"] for r in query_results) / len(query_results),
                "average_response_time": sum(r["response_time"] for r in query_results) / len(query_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_compliance_and_audit_trail(self) -> Dict[str, Any]:
        """Test: Compliance and Audit Trail Functionality"""
        scenario = {
            "name": "Compliance and Audit Trail",
            "description": "Compliance officer reviews system audit trails and compliance reports",
            "user_persona": "Compliance Officer",
            "business_value": "Regulatory compliance and risk management"
        }
        
        try:
            # Step 1: Generate compliance report
            report_request = {
                "report_type": "comprehensive_audit",
                "time_period": "last_30_days",
                "compliance_frameworks": ["SOX", "GDPR", "ISO27001"]
            }
            
            report_response = requests.post(
                f"{self.base_url}/api/compliance/reports",
                json=report_request,
                timeout=60
            )
            assert report_response.status_code == 201, "Failed to generate compliance report"
            
            report_id = report_response.json()["report_id"]
            
            # Step 2: Verify audit trail completeness
            audit_response = requests.get(
                f"{self.base_url}/api/audit/trail",
                params={"days": 7},
                timeout=30
            )
            assert audit_response.status_code == 200, "Failed to retrieve audit trail"
            
            audit_data = audit_response.json()
            assert len(audit_data["events"]) > 0, "No audit events found"
            
            # Step 3: Test data privacy compliance
            privacy_response = requests.get(
                f"{self.base_url}/api/compliance/privacy/assessment",
                timeout=20
            )
            assert privacy_response.status_code == 200, "Privacy assessment failed"
            
            privacy_data = privacy_response.json()
            assert privacy_data["compliance_score"] >= 0.9, "Privacy compliance score too low"
            
            # Step 4: Verify access controls
            access_response = requests.get(
                f"{self.base_url}/api/security/access-controls/review",
                timeout=15
            )
            assert access_response.status_code == 200, "Access control review failed"
            
            return {
                "success": True,
                "scenario": scenario,
                "report_id": report_id,
                "audit_events_count": len(audit_data["events"]),
                "compliance_score": privacy_data["compliance_score"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_performance_under_load(self) -> Dict[str, Any]:
        """Test: System Performance Under Realistic Load"""
        scenario = {
            "name": "Performance Under Load",
            "description": "System handles multiple concurrent users and complex operations",
            "user_persona": "System Administrator",
            "business_value": "Reliable performance during peak usage"
        }
        
        try:
            import concurrent.futures
            import threading
            
            # Simulate concurrent users
            def simulate_user_session():
                session_start = time.time()
                
                # Simulate typical user workflow
                requests.get(f"{self.base_url}/health", timeout=5)
                requests.get(f"{self.base_url}/api/dashboard/overview", timeout=10)
                requests.post(f"{self.base_url}/api/intelligence/query", 
                            json={"query": "Show system status"}, timeout=15)
                
                return time.time() - session_start
            
            # Run concurrent sessions
            num_concurrent_users = 20
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_users) as executor:
                futures = [executor.submit(simulate_user_session) for _ in range(num_concurrent_users)]
                session_times = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Analyze performance
            avg_session_time = sum(session_times) / len(session_times)
            max_session_time = max(session_times)
            
            # Performance thresholds
            assert avg_session_time < 10.0, f"Average session time too high: {avg_session_time}s"
            assert max_session_time < 20.0, f"Maximum session time too high: {max_session_time}s"
            
            # Test system health after load
            health_response = requests.get(f"{self.base_url}/health/detailed", timeout=10)
            assert health_response.status_code == 200, "System health check failed after load test"
            
            health_data = health_response.json()
            assert health_data["status"] == "healthy", "System not healthy after load test"
            
            return {
                "success": True,
                "scenario": scenario,
                "concurrent_users": num_concurrent_users,
                "average_session_time": avg_session_time,
                "max_session_time": max_session_time,
                "system_health": health_data["status"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_security_and_access_control(self) -> Dict[str, Any]:
        """Test: Security and Access Control Features"""
        scenario = {
            "name": "Security and Access Control",
            "description": "Security administrator validates access controls and security features",
            "user_persona": "Security Administrator",
            "business_value": "Data protection and secure access management"
        }
        
        try:
            # Test authentication
            auth_response = requests.post(
                f"{self.base_url}/api/auth/validate",
                headers={"Authorization": "Bearer test_token"},
                timeout=10
            )
            # Should return 401 for invalid token
            assert auth_response.status_code == 401, "Authentication not properly enforced"
            
            # Test rate limiting
            rate_limit_responses = []
            for _ in range(10):
                response = requests.get(f"{self.base_url}/api/test/rate-limit", timeout=5)
                rate_limit_responses.append(response.status_code)
            
            # Should see rate limiting after several requests
            assert 429 in rate_limit_responses, "Rate limiting not working"
            
            # Test security headers
            security_response = requests.get(f"{self.base_url}/health", timeout=5)
            security_headers = security_response.headers
            
            required_headers = ["X-Content-Type-Options", "X-Frame-Options", "X-XSS-Protection"]
            for header in required_headers:
                assert header in security_headers, f"Missing security header: {header}"
            
            # Test data encryption
            encryption_response = requests.get(
                f"{self.base_url}/api/security/encryption/status",
                timeout=10
            )
            assert encryption_response.status_code == 200, "Encryption status check failed"
            
            encryption_data = encryption_response.json()
            assert encryption_data["data_at_rest_encrypted"], "Data at rest not encrypted"
            assert encryption_data["data_in_transit_encrypted"], "Data in transit not encrypted"
            
            return {
                "success": True,
                "scenario": scenario,
                "security_headers_count": len([h for h in required_headers if h in security_headers]),
                "encryption_status": "enabled",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_mobile_and_responsive_interface(self) -> Dict[str, Any]:
        """Test: Mobile and Responsive Interface"""
        scenario = {
            "name": "Mobile and Responsive Interface",
            "description": "Mobile user accesses system features on smartphone",
            "user_persona": "Mobile Executive",
            "business_value": "Access to business intelligence on the go"
        }
        
        try:
            # Test mobile-optimized endpoints
            mobile_headers = {
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
            }
            
            # Test mobile dashboard
            mobile_response = requests.get(
                f"{self.base_url}/api/dashboard/mobile",
                headers=mobile_headers,
                timeout=10
            )
            assert mobile_response.status_code == 200, "Mobile dashboard not accessible"
            
            mobile_data = mobile_response.json()
            assert "mobile_optimized" in mobile_data, "Dashboard not mobile optimized"
            
            # Test responsive API responses
            responsive_response = requests.get(
                f"{self.base_url}/api/intelligence/query/mobile",
                headers=mobile_headers,
                timeout=15
            )
            assert responsive_response.status_code == 200, "Mobile intelligence API not working"
            
            # Test offline capabilities
            offline_response = requests.get(
                f"{self.base_url}/api/offline/sync-status",
                headers=mobile_headers,
                timeout=10
            )
            assert offline_response.status_code == 200, "Offline sync not available"
            
            return {
                "success": True,
                "scenario": scenario,
                "mobile_optimized": True,
                "offline_capable": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_integration_with_existing_systems(self) -> Dict[str, Any]:
        """Test: Integration with Existing Business Systems"""
        scenario = {
            "name": "Integration with Existing Systems",
            "description": "IT administrator validates integration with enterprise systems",
            "user_persona": "IT Administrator",
            "business_value": "Seamless integration with existing business infrastructure"
        }
        
        try:
            # Test database connectivity
            db_response = requests.get(
                f"{self.base_url}/api/integrations/database/status",
                timeout=10
            )
            assert db_response.status_code == 200, "Database integration not working"
            
            # Test API integrations
            api_integrations = ["crm", "erp", "data_warehouse", "monitoring"]
            integration_results = {}
            
            for integration in api_integrations:
                integration_response = requests.get(
                    f"{self.base_url}/api/integrations/{integration}/health",
                    timeout=15
                )
                integration_results[integration] = integration_response.status_code == 200
            
            # At least 75% of integrations should be working
            working_integrations = sum(integration_results.values())
            integration_success_rate = working_integrations / len(api_integrations)
            assert integration_success_rate >= 0.75, f"Too many integration failures: {integration_success_rate}"
            
            # Test data synchronization
            sync_response = requests.post(
                f"{self.base_url}/api/integrations/sync/test",
                json={"systems": ["crm", "erp"]},
                timeout=30
            )
            assert sync_response.status_code == 200, "Data synchronization test failed"
            
            return {
                "success": True,
                "scenario": scenario,
                "integration_success_rate": integration_success_rate,
                "working_integrations": working_integrations,
                "total_integrations": len(api_integrations),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report = {
            "test_suite": "Agent Steering System User Acceptance Tests",
            "execution_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED"
            },
            "test_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [name for name, result in self.test_results.items() 
                       if not result.get("success", False)]
        
        if failed_tests:
            recommendations.append(f"Address failures in: {', '.join(failed_tests)}")
        
        # Performance recommendations
        performance_results = self.test_results.get("test_performance_under_load", {})
        if performance_results.get("success") and performance_results.get("average_session_time", 0) > 5:
            recommendations.append("Consider performance optimization for better response times")
        
        # Security recommendations
        security_results = self.test_results.get("test_security_and_access_control", {})
        if not security_results.get("success"):
            recommendations.append("Review and strengthen security controls")
        
        # Integration recommendations
        integration_results = self.test_results.get("test_integration_with_existing_systems", {})
        if integration_results.get("integration_success_rate", 0) < 1.0:
            recommendations.append("Improve integration reliability with external systems")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system ready for production")
        
        return recommendations

def run_user_acceptance_tests(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run user acceptance tests and return results"""
    test_suite = UserAcceptanceTestSuite(base_url)
    return test_suite.run_all_tests()

if __name__ == "__main__":
    # Run tests if executed directly
    results = run_user_acceptance_tests()
    print(json.dumps(results, indent=2))