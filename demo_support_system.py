"""
Demo script for ScrollIntel support system functionality.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from scrollintel.core.support_system import (
    SupportTicketManager, KnowledgeBaseManager, 
    FeedbackManager, ContactFormManager
)
from scrollintel.models.support_models import (
    SupportTicketCreate, SupportMessageCreate, FeedbackCreate,
    ContactFormSubmission, TicketStatus, TicketPriority
)
from scrollintel.core.knowledge_base_seeder import seed_knowledge_base

class SupportSystemDemo:
    """Demonstration of ScrollIntel support system capabilities."""
    
    def __init__(self):
        """Initialize demo with mock database session."""
        self.db_session = self._create_mock_db_session()
        self.ticket_manager = SupportTicketManager(self.db_session)
        self.kb_manager = KnowledgeBaseManager(self.db_session)
        self.feedback_manager = FeedbackManager(self.db_session)
        self.contact_manager = ContactFormManager(self.db_session)
    
    def _create_mock_db_session(self):
        """Create mock database session for demo purposes."""
        class MockDBSession:
            def __init__(self):
                self.tickets = []
                self.messages = []
                self.articles = []
                self.faqs = []
                self.feedback = []
                self._id_counter = 1
            
            def add(self, obj):
                obj.id = self._id_counter
                self._id_counter += 1
                
                if hasattr(obj, 'ticket_number'):
                    self.tickets.append(obj)
                elif hasattr(obj, 'message'):
                    self.messages.append(obj)
                elif hasattr(obj, 'feedback_type'):
                    self.feedback.append(obj)
            
            def commit(self):
                pass
            
            def refresh(self, obj):
                pass
            
            def rollback(self):
                pass
            
            def query(self, model):
                return MockQuery(self, model)
        
        class MockQuery:
            def __init__(self, session, model):
                self.session = session
                self.model = model
            
            def filter(self, *args):
                return self
            
            def first(self):
                if hasattr(self.model, 'ticket_number'):
                    return self.session.tickets[0] if self.session.tickets else None
                return None
            
            def all(self):
                return []
            
            def count(self):
                return len(self.session.tickets)
            
            def order_by(self, *args):
                return self
            
            def limit(self, limit):
                return self
            
            def group_by(self, *args):
                return self
            
            def scalar(self):
                return 4.5
        
        return MockDBSession()
    
    async def demo_ticket_management(self):
        """Demonstrate ticket management functionality."""
        print("\n" + "="*60)
        print("SUPPORT TICKET MANAGEMENT DEMO")
        print("="*60)
        
        # Create sample tickets
        tickets_data = [
            {
                "email": "user1@example.com",
                "subject": "Unable to upload CSV file",
                "description": "I'm getting an error when trying to upload my customer data CSV file. The error message says 'Invalid file format' but I'm sure it's a valid CSV.",
                "category": "technical_issue",
                "priority": TicketPriority.HIGH,
                "user_id": "user_001"
            },
            {
                "email": "user2@example.com",
                "subject": "API integration question",
                "description": "I'm trying to integrate the ScrollIntel API with our existing system. Can you provide examples of how to authenticate and make requests?",
                "category": "api_integration",
                "priority": TicketPriority.MEDIUM,
                "user_id": "user_002"
            },
            {
                "email": "user3@example.com",
                "subject": "Billing inquiry about usage limits",
                "description": "I've reached my monthly usage limit but I need to process more data. How can I upgrade my plan or purchase additional credits?",
                "category": "billing_question",
                "priority": TicketPriority.MEDIUM,
                "user_id": "user_003"
            }
        ]
        
        created_tickets = []
        for ticket_data in tickets_data:
            try:
                ticket_create = SupportTicketCreate(**ticket_data)
                ticket = self.ticket_manager.create_ticket(ticket_create)
                created_tickets.append(ticket)
                
                print(f"✅ Created ticket: {ticket.ticket_number}")
                print(f"   Subject: {ticket.subject}")
                print(f"   Priority: {ticket.priority}")
                print(f"   Status: {ticket.status}")
                print()
                
            except Exception as e:
                print(f"❌ Failed to create ticket: {str(e)}")
        
        # Add messages to tickets
        print("\n📝 Adding messages to tickets...")
        for i, ticket in enumerate(created_tickets[:2]):  # Add messages to first 2 tickets
            try:
                message_data = SupportMessageCreate(
                    message=f"Thank you for contacting ScrollIntel support. We've received your request and are looking into it. Ticket #{ticket.ticket_number}",
                    sender_name="Sarah Chen",
                    sender_email="support@scrollintel.com"
                )
                
                message = self.ticket_manager.add_message(ticket.id, message_data, "agent")
                print(f"✅ Added agent response to ticket {ticket.ticket_number}")
                
            except Exception as e:
                print(f"❌ Failed to add message: {str(e)}")
        
        # Update ticket statuses
        print("\n🔄 Updating ticket statuses...")
        try:
            # Resolve first ticket
            resolved_ticket = self.ticket_manager.update_ticket_status(
                created_tickets[0].id, 
                TicketStatus.RESOLVED, 
                "sarah.chen@scrollintel.com"
            )
            print(f"✅ Resolved ticket {resolved_ticket.ticket_number}")
            
            # Mark second ticket as in progress
            in_progress_ticket = self.ticket_manager.update_ticket_status(
                created_tickets[1].id, 
                TicketStatus.IN_PROGRESS, 
                "mike.rodriguez@scrollintel.com"
            )
            print(f"✅ Updated ticket {in_progress_ticket.ticket_number} to in progress")
            
        except Exception as e:
            print(f"❌ Failed to update ticket status: {str(e)}")
        
        # Get ticket metrics
        print("\n📊 Support Metrics:")
        try:
            metrics = self.ticket_manager.get_ticket_metrics()
            print(f"   Total Tickets: {metrics.get('total_tickets', 0)}")
            print(f"   Open Tickets: {metrics.get('open_tickets', 0)}")
            print(f"   Resolved Tickets: {metrics.get('resolved_tickets', 0)}")
            print(f"   Resolution Rate: {metrics.get('resolution_rate', 0):.1%}")
            print(f"   Avg Resolution Time: {metrics.get('avg_resolution_hours', 0):.1f} hours")
            
        except Exception as e:
            print(f"❌ Failed to get metrics: {str(e)}")
    
    async def demo_knowledge_base(self):
        """Demonstrate knowledge base functionality."""
        print("\n" + "="*60)
        print("KNOWLEDGE BASE DEMO")
        print("="*60)
        
        # Seed knowledge base (in real implementation, this would be done once)
        print("📚 Seeding knowledge base with sample content...")
        try:
            seed_knowledge_base(self.db_session)
            print("✅ Knowledge base seeded successfully")
        except Exception as e:
            print(f"❌ Failed to seed knowledge base: {str(e)}")
        
        # Search help content
        print("\n🔍 Searching help content...")
        search_queries = [
            "upload data",
            "API integration",
            "billing plans",
            "AI agents"
        ]
        
        for query in search_queries:
            try:
                results = self.kb_manager.search_help_content(query, limit=3)
                print(f"\n   Query: '{query}'")
                print(f"   Found {len(results.get('articles', []))} articles, {len(results.get('faqs', []))} FAQs")
                
                # Show first result
                if results.get('articles'):
                    article = results['articles'][0]
                    print(f"   📄 Top Article: {article.title}")
                
                if results.get('faqs'):
                    faq = results['faqs'][0]
                    print(f"   ❓ Top FAQ: {faq.question}")
                    
            except Exception as e:
                print(f"❌ Search failed for '{query}': {str(e)}")
        
        # Get featured FAQs
        print("\n⭐ Featured FAQs:")
        try:
            featured_faqs = self.kb_manager.get_faqs(featured_only=True)
            for faq in featured_faqs[:5]:  # Show first 5
                print(f"   ❓ {faq.question}")
                print(f"      👍 {faq.helpful_votes} helpful, 👎 {faq.unhelpful_votes} unhelpful")
                
        except Exception as e:
            print(f"❌ Failed to get featured FAQs: {str(e)}")
        
        # Simulate voting on content
        print("\n👍 Simulating user votes on content...")
        try:
            self.kb_manager.vote_helpful("article", 1, True)
            self.kb_manager.vote_helpful("faq", 1, True)
            self.kb_manager.vote_helpful("faq", 2, False)
            print("✅ Recorded user votes on help content")
            
        except Exception as e:
            print(f"❌ Failed to record votes: {str(e)}")
    
    async def demo_feedback_system(self):
        """Demonstrate feedback collection functionality."""
        print("\n" + "="*60)
        print("FEEDBACK SYSTEM DEMO")
        print("="*60)
        
        # Sample feedback submissions
        feedback_data = [
            {
                "feedback_type": "feature_request",
                "title": "Add support for PostgreSQL data sources",
                "description": "It would be great if ScrollIntel could directly connect to PostgreSQL databases instead of requiring CSV exports.",
                "rating": 5,
                "email": "developer@company.com",
                "user_id": "user_004"
            },
            {
                "feedback_type": "bug_report",
                "title": "Chart export to PDF is blurry",
                "description": "When I export charts to PDF, the resolution is too low and the text is blurry. This makes it hard to use in presentations.",
                "rating": 3,
                "email": "analyst@company.com",
                "user_id": "user_005"
            },
            {
                "feedback_type": "general_feedback",
                "title": "Love the new AI agents!",
                "description": "The CTO agent has been incredibly helpful for our technology planning. It's like having a senior technical advisor on our team.",
                "rating": 5,
                "email": "ceo@startup.com",
                "user_id": "user_006"
            },
            {
                "feedback_type": "user_experience",
                "title": "Dashboard could be more intuitive",
                "description": "The main dashboard has a lot of information but it's not immediately clear where to start. Maybe add a guided tour for new users?",
                "rating": 4,
                "email": "newuser@company.com",
                "user_id": "user_007"
            }
        ]
        
        print("📝 Collecting user feedback...")
        for feedback_item in feedback_data:
            try:
                feedback_create = FeedbackCreate(**feedback_item)
                feedback = self.feedback_manager.submit_feedback(feedback_create)
                
                print(f"✅ Feedback submitted: {feedback.title}")
                print(f"   Type: {feedback.feedback_type}")
                print(f"   Rating: {feedback.rating}/5 stars")
                print()
                
            except Exception as e:
                print(f"❌ Failed to submit feedback: {str(e)}")
        
        # Get feedback summary
        print("📊 Feedback Summary:")
        try:
            summary = self.feedback_manager.get_feedback_summary()
            print(f"   Total Feedback: {summary.get('total_feedback', 0)}")
            print(f"   Average Rating: {summary.get('average_rating', 0):.1f}/5.0")
            print("   Feedback by Type:")
            
            for feedback_type, count in summary.get('feedback_by_type', {}).items():
                print(f"     {feedback_type}: {count}")
                
        except Exception as e:
            print(f"❌ Failed to get feedback summary: {str(e)}")
    
    async def demo_contact_form(self):
        """Demonstrate contact form functionality."""
        print("\n" + "="*60)
        print("CONTACT FORM DEMO")
        print("="*60)
        
        # Sample contact form submissions
        contact_submissions = [
            {
                "name": "Alice Johnson",
                "email": "alice@enterprise.com",
                "company": "Enterprise Corp",
                "subject": "Enterprise pricing inquiry",
                "message": "We're interested in ScrollIntel for our 500-person organization. Can you provide enterprise pricing and deployment options?",
                "phone": "555-0123",
                "inquiry_type": "sales"
            },
            {
                "name": "Bob Smith",
                "email": "bob@startup.io",
                "company": "Startup Inc",
                "subject": "Partnership opportunity",
                "message": "We're building a data analytics platform and would like to explore partnership opportunities with ScrollIntel.",
                "phone": "555-0456",
                "inquiry_type": "partnership"
            },
            {
                "name": "Carol Davis",
                "email": "carol@media.com",
                "company": "Tech Media",
                "subject": "Press inquiry",
                "message": "I'm writing an article about AI in business intelligence. Would someone from your team be available for an interview?",
                "inquiry_type": "media"
            }
        ]
        
        print("📧 Processing contact form submissions...")
        for contact_data in contact_submissions:
            try:
                contact_form = ContactFormSubmission(**contact_data)
                ticket = self.contact_manager.submit_contact_form(contact_form)
                
                print(f"✅ Contact form processed: {contact_form.subject}")
                print(f"   From: {contact_form.name} ({contact_form.email})")
                print(f"   Company: {contact_form.company}")
                print(f"   Ticket Created: {ticket.ticket_number}")
                print()
                
            except Exception as e:
                print(f"❌ Failed to process contact form: {str(e)}")
    
    async def demo_support_analytics(self):
        """Demonstrate support analytics and reporting."""
        print("\n" + "="*60)
        print("SUPPORT ANALYTICS DEMO")
        print("="*60)
        
        # Simulate support metrics
        print("📈 Support Performance Metrics:")
        
        metrics = {
            "ticket_volume": {
                "today": 23,
                "this_week": 156,
                "this_month": 642,
                "trend": "+12% vs last month"
            },
            "response_times": {
                "first_response": "1.2 hours",
                "resolution_time": "4.8 hours",
                "sla_compliance": "94.2%"
            },
            "customer_satisfaction": {
                "average_rating": 4.6,
                "response_rate": "78%",
                "nps_score": 67
            },
            "agent_performance": {
                "tickets_resolved": 89,
                "avg_resolution_time": "3.2 hours",
                "customer_rating": 4.8
            }
        }
        
        print(f"\n🎫 Ticket Volume:")
        print(f"   Today: {metrics['ticket_volume']['today']}")
        print(f"   This Week: {metrics['ticket_volume']['this_week']}")
        print(f"   This Month: {metrics['ticket_volume']['this_month']}")
        print(f"   Trend: {metrics['ticket_volume']['trend']}")
        
        print(f"\n⏱️ Response Times:")
        print(f"   First Response: {metrics['response_times']['first_response']}")
        print(f"   Resolution Time: {metrics['response_times']['resolution_time']}")
        print(f"   SLA Compliance: {metrics['response_times']['sla_compliance']}")
        
        print(f"\n😊 Customer Satisfaction:")
        print(f"   Average Rating: {metrics['customer_satisfaction']['average_rating']}/5.0")
        print(f"   Response Rate: {metrics['customer_satisfaction']['response_rate']}")
        print(f"   NPS Score: {metrics['customer_satisfaction']['nps_score']}")
        
        print(f"\n👨‍💼 Agent Performance:")
        print(f"   Tickets Resolved: {metrics['agent_performance']['tickets_resolved']}")
        print(f"   Avg Resolution Time: {metrics['agent_performance']['avg_resolution_time']}")
        print(f"   Customer Rating: {metrics['agent_performance']['customer_rating']}/5.0")
        
        # Knowledge base analytics
        print(f"\n📚 Knowledge Base Analytics:")
        kb_metrics = {
            "total_articles": 45,
            "total_views": 12847,
            "most_viewed": "Getting Started with ScrollIntel",
            "search_success_rate": "87%",
            "user_satisfaction": 4.3
        }
        
        for metric, value in kb_metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value}")
    
    async def run_complete_demo(self):
        """Run the complete support system demonstration."""
        print("🚀 ScrollIntel Support System Demo")
        print("=" * 60)
        print("Demonstrating comprehensive customer support capabilities")
        print("including tickets, knowledge base, feedback, and analytics.")
        
        try:
            await self.demo_ticket_management()
            await self.demo_knowledge_base()
            await self.demo_feedback_system()
            await self.demo_contact_form()
            await self.demo_support_analytics()
            
            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY")
            print("="*60)
            print("\nScrollIntel Support System Features Demonstrated:")
            print("• ✅ Support ticket creation and management")
            print("• ✅ Knowledge base with articles and FAQs")
            print("• ✅ User feedback collection and analysis")
            print("• ✅ Contact form processing")
            print("• ✅ Support analytics and reporting")
            print("• ✅ Multi-channel customer communication")
            print("• ✅ Agent workflow management")
            print("• ✅ Customer satisfaction tracking")
            
            print("\n🎯 Key Benefits:")
            print("• Reduced support ticket volume through self-service")
            print("• Faster resolution times with knowledge base")
            print("• Improved customer satisfaction with multiple channels")
            print("• Data-driven support optimization")
            print("• Scalable support operations")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            raise

async def main():
    """Run the support system demo."""
    demo = SupportSystemDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())