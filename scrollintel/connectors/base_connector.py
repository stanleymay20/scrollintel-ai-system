"""
Base connector interface for data sources.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

class BaseConnector(ABC):
    """Base class for all data source connectors."""
    
    @abstractmethod
    async def test_connection(self, connection_config: Dict[str, Any], 
                            auth_config: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Test connection to the data source.
        
        Returns:
            Tuple of (success, error_message, test_details)
        """
        pass
    
    @abstractmethod
    async def create_connection(self, connection_config: Dict[str, Any], 
                              auth_config: Dict[str, Any]) -> Any:
        """Create and return a connection object."""
        pass
    
    @abstractmethod
    async def discover_schema(self, connection: Any) -> List[Dict[str, Any]]:
        """
        Discover schema information from the data source.
        
        Returns:
            List of schema dictionaries with structure:
            {
                "schema_name": str,
                "table_name": str,
                "columns": [
                    {
                        "name": str,
                        "type": str,
                        "nullable": bool,
                        "primary_key": bool
                    }
                ]
            }
        """
        pass
    
    @abstractmethod
    async def read_data(self, connection: Any, query_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read data from the source based on query configuration."""
        pass
    
    def validate_config(self, connection_config: Dict[str, Any]) -> List[str]:
        """
        Validate connection configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        return []