"""
Proxy Manager for ScrollIntel Agent Communication.
Singleton manager for all agent proxies.
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class ProxyManager:
    """Singleton manager for all agent proxies."""
    
    _instance: Optional['ProxyManager'] = None
    
    def __init__(self):
        self.proxies: Dict[str, 'AgentProxy'] = {}
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> 'ProxyManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def register_proxy(self, proxy: 'AgentProxy') -> None:
        """Register an agent proxy."""
        async with self._lock:
            self.proxies[proxy.agent.agent_id] = proxy
            await proxy.start_message_processing()
            logger.info(f"Registered proxy for agent {proxy.agent.agent_id}")
    
    async def unregister_proxy(self, agent_id: str) -> None:
        """Unregister an agent proxy."""
        async with self._lock:
            if agent_id in self.proxies:
                proxy = self.proxies[agent_id]
                await proxy.stop_message_processing()
                del self.proxies[agent_id]
                logger.info(f"Unregistered proxy for agent {agent_id}")
    
    async def route_message(self, message: 'AgentMessage') -> None:
        """Route a message to the appropriate agent proxy."""
        from ..core.interfaces import AgentError
        
        recipient_proxy = self.proxies.get(message.recipient_id)
        if recipient_proxy:
            await recipient_proxy.receive_message(message)
        else:
            logger.error(f"No proxy found for agent {message.recipient_id}")
            raise AgentError(f"Agent {message.recipient_id} not found")
    
    def get_proxy(self, agent_id: str) -> Optional['AgentProxy']:
        """Get a proxy by agent ID."""
        return self.proxies.get(agent_id)
    
    def get_all_proxies(self) -> List['AgentProxy']:
        """Get all registered proxies."""
        return list(self.proxies.values())
    
    async def broadcast_message(
        self,
        sender_id: str,
        message_type: str,
        payload: Dict[str, Any],
        exclude_agents: Optional[List[str]] = None,
    ) -> List[str]:
        """Broadcast a message to all agents except excluded ones."""
        exclude_agents = exclude_agents or []
        message_ids = []
        
        sender_proxy = self.proxies.get(sender_id)
        if not sender_proxy:
            from ..core.interfaces import AgentError
            raise AgentError(f"Sender agent {sender_id} not found")
        
        for agent_id in self.proxies.keys():
            if agent_id != sender_id and agent_id not in exclude_agents:
                message_id = await sender_proxy.send_message(
                    recipient_id=agent_id,
                    message_type=message_type,
                    payload=payload,
                )
                message_ids.append(message_id)
        
        return message_ids
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get statistics about the proxy manager."""
        return {
            "total_proxies": len(self.proxies),
            "active_proxies": len([p for p in self.proxies.values() if p._running]),
            "proxy_stats": {
                agent_id: proxy.get_message_stats()
                for agent_id, proxy in self.proxies.items()
            },
        }