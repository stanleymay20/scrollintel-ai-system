import json
import markdown
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)

class ConversationExporter:
    """Export conversations in various formats"""
    
    def __init__(self):
        self.markdown_processor = markdown.Markdown(
            extensions=['codehilite', 'fenced_code', 'tables', 'toc']
        )
    
    async def export(
        self,
        conversation: Any,
        messages: List[Any],
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """Export conversation in the specified format"""
        if format == "markdown":
            return await self._export_markdown(conversation, messages)
        elif format == "pdf":
            return await self._export_pdf(conversation, messages)
        elif format == "json":
            return await self._export_json(conversation, messages)
        elif format == "txt":
            return await self._export_txt(conversation, messages)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _export_markdown(self, conversation: Any, messages: List[Any]) -> Dict[str, Any]:
        """Export conversation as Markdown"""
        try:
            content = []
            
            # Header
            content.append(f"# {conversation.title or 'Conversation'}")
            content.append(f"**Created:** {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            content.append(f"**Updated:** {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if conversation.agent_id:
                content.append(f"**Agent:** {conversation.agent_id}")
            content.append("\n---\n")
            
            # Messages
            for message in messages:
                role_emoji = "🧑" if message.role == "user" else "🤖" if message.role == "assistant" else "ℹ️"
                role_name = message.role.title()
                
                content.append(f"## {role_emoji} {role_name}")
                content.append(f"*{message.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")
                content.append("")
                content.append(message.content)
                content.append("")
                
                # Add attachments if any
                if hasattr(message, 'attachments') and message.attachments:
                    content.append("**Attachments:**")
                    for attachment in message.attachments:
                        content.append(f"- [{attachment['file_name']}]({attachment['file_url']})")
                    content.append("")
                
                # Add metadata if significant
                if hasattr(message, 'metadata') and message.metadata:
                    metadata = message.metadata
                    if metadata.get('token_count') or metadata.get('model_used'):
                        content.append("<details>")
                        content.append("<summary>Message Details</summary>")
                        content.append("")
                        if metadata.get('model_used'):
                            content.append(f"- **Model:** {metadata['model_used']}")
                        if metadata.get('token_count'):
                            content.append(f"- **Tokens:** {metadata['token_count']}")
                        if metadata.get('execution_time'):
                            content.append(f"- **Response Time:** {metadata['execution_time']:.2f}s")
                        content.append("")
                        content.append("</details>")
                        content.append("")
                
                content.append("---\n")
            
            # Footer
            content.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
            markdown_content = "\n".join(content)
            
            return {
                "format": "markdown",
                "content": markdown_content,
                "filename": f"{conversation.title or 'conversation'}_{conversation.id[:8]}.md",
                "mime_type": "text/markdown",
                "size": len(markdown_content.encode('utf-8'))
            }
        
        except Exception as e:
            logger.error(f"Error exporting to markdown: {e}")
            raise
    
    async def _export_pdf(self, conversation: Any, messages: List[Any]) -> Dict[str, Any]:
        """Export conversation as PDF"""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            user_style = ParagraphStyle(
                'UserMessage',
                parent=styles['Normal'],
                leftIndent=20,
                rightIndent=20,
                spaceBefore=10,
                spaceAfter=10,
                borderColor=colors.lightblue,
                borderWidth=1,
                borderPadding=10
            )
            
            assistant_style = ParagraphStyle(
                'AssistantMessage',
                parent=styles['Normal'],
                leftIndent=20,
                rightIndent=20,
                spaceBefore=10,
                spaceAfter=10,
                borderColor=colors.lightgrey,
                borderWidth=1,
                borderPadding=10
            )
            
            # Title and metadata
            story.append(Paragraph(conversation.title or "Conversation", title_style))
            story.append(Spacer(1, 12))
            
            # Metadata table
            metadata_data = [
                ['Created', conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')],
                ['Updated', conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')],
                ['Messages', str(len(messages))]
            ]
            
            if conversation.agent_id:
                metadata_data.append(['Agent', conversation.agent_id])
            
            metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 20))
            
            # Messages
            for i, message in enumerate(messages):
                # Message header
                role_name = message.role.title()
                timestamp = message.created_at.strftime('%Y-%m-%d %H:%M:%S')
                header_text = f"<b>{role_name}</b> - {timestamp}"
                story.append(Paragraph(header_text, styles['Heading3']))
                
                # Message content
                # Clean content for PDF (remove markdown formatting)
                clean_content = self._clean_content_for_pdf(message.content)
                
                # Choose style based on role
                message_style = user_style if message.role == "user" else assistant_style
                story.append(Paragraph(clean_content, message_style))
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            pdf_content = buffer.getvalue()
            buffer.close()
            
            return {
                "format": "pdf",
                "content": base64.b64encode(pdf_content).decode('utf-8'),
                "filename": f"{conversation.title or 'conversation'}_{conversation.id[:8]}.pdf",
                "mime_type": "application/pdf",
                "size": len(pdf_content)
            }
        
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            raise
    
    async def _export_json(self, conversation: Any, messages: List[Any]) -> Dict[str, Any]:
        """Export conversation as JSON"""
        try:
            export_data = {
                "conversation": {
                    "id": conversation.id,
                    "title": conversation.title,
                    "agent_id": conversation.agent_id,
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                    "tags": getattr(conversation, 'tags', []),
                    "metadata": getattr(conversation, 'metadata', {})
                },
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "content_type": msg.content_type,
                        "created_at": msg.created_at.isoformat(),
                        "updated_at": msg.updated_at.isoformat() if msg.updated_at else None,
                        "parent_message_id": msg.parent_message_id,
                        "regeneration_count": msg.regeneration_count,
                        "metadata": getattr(msg, 'metadata', {}),
                        "attachments": getattr(msg, 'attachments', []),
                        "reactions": getattr(msg, 'reactions', [])
                    }
                    for msg in messages
                ],
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "format": "json",
                    "version": "1.0"
                }
            }
            
            json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            return {
                "format": "json",
                "content": json_content,
                "filename": f"{conversation.title or 'conversation'}_{conversation.id[:8]}.json",
                "mime_type": "application/json",
                "size": len(json_content.encode('utf-8'))
            }
        
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    async def _export_txt(self, conversation: Any, messages: List[Any]) -> Dict[str, Any]:
        """Export conversation as plain text"""
        try:
            content = []
            
            # Header
            content.append(f"Conversation: {conversation.title or 'Untitled'}")
            content.append(f"Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            content.append(f"Updated: {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if conversation.agent_id:
                content.append(f"Agent: {conversation.agent_id}")
            content.append("=" * 50)
            content.append("")
            
            # Messages
            for message in messages:
                role_name = message.role.upper()
                timestamp = message.created_at.strftime('%Y-%m-%d %H:%M:%S')
                
                content.append(f"[{timestamp}] {role_name}:")
                content.append("-" * 30)
                
                # Clean content for plain text
                clean_content = self._clean_content_for_text(message.content)
                content.append(clean_content)
                content.append("")
            
            # Footer
            content.append("=" * 50)
            content.append(f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            text_content = "\n".join(content)
            
            return {
                "format": "txt",
                "content": text_content,
                "filename": f"{conversation.title or 'conversation'}_{conversation.id[:8]}.txt",
                "mime_type": "text/plain",
                "size": len(text_content.encode('utf-8'))
            }
        
        except Exception as e:
            logger.error(f"Error exporting to text: {e}")
            raise
    
    def _clean_content_for_pdf(self, content: str) -> str:
        """Clean content for PDF export"""
        # Remove markdown formatting
        import re
        
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '[Code Block]', content)
        
        # Remove inline code
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Remove markdown links
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove markdown formatting
        content = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', content)
        
        # Remove headers
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        
        return content
    
    def _clean_content_for_text(self, content: str) -> str:
        """Clean content for plain text export"""
        # Similar to PDF but even simpler
        import re
        
        # Remove code blocks but preserve content
        content = re.sub(r'```(\w+)?\n([\s\S]*?)```', r'[Code]\n\2\n[/Code]', content)
        
        # Remove inline code backticks
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Convert markdown links to plain text
        content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1 (\2)', content)
        
        # Remove markdown formatting
        content = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', content)
        
        # Convert headers to uppercase
        content = re.sub(r'^(#+)\s*(.+)$', lambda m: m.group(2).upper(), content, flags=re.MULTILINE)
        
        return content