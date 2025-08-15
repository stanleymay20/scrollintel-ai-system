"""
ScrollIntel Agents Module
Contains all AI agents that provide specialized capabilities.
"""

from ..core.interfaces import BaseAgent, AgentType, AgentStatus
from .proxy import AgentProxy, AgentMessage
from .proxy_manager import ProxyManager
from .scroll_analyst import ScrollAnalyst
from .scroll_bi_agent import ScrollBIAgent

__all__ = [
    "BaseAgent",
    "AgentType", 
    "AgentStatus",
    "AgentProxy",
    "AgentMessage",
    "ProxyManager",
    "ScrollAnalyst",
    "ScrollBIAgent",
]