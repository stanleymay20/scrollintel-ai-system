"""
Optimized Configuration Module with Lazy Loading
"""
import os
from functools import lru_cache
from typing import Optional

class LazySettings:
    """Lazy-loaded settings class"""
    
    def __init__(self):
        self._settings = {}
        self._loaded = False
    
    def _load_settings(self):
        """Load settings only when needed"""
        if self._loaded:
            return
            
        # Load environment variables
        self._settings = {
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
            'DEBUG': os.getenv('DEBUG', 'true').lower() == 'true',
            'API_HOST': os.getenv('API_HOST', '0.0.0.0'),
            'API_PORT': int(os.getenv('API_PORT', 8000)),
            'DATABASE_URL': os.getenv('DATABASE_URL', 'sqlite:///scrollintel.db'),
            'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY', 'default_secret_key'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
        }
        self._loaded = True
    
    def __getattr__(self, name):
        """Lazy attribute access"""
        self._load_settings()
        if name in self._settings:
            return self._settings[name]
        raise AttributeError(f"Setting '{name}' not found")

# Global lazy settings instance
_settings = LazySettings()

@lru_cache(maxsize=1)
def get_settings():
    """Get cached settings instance"""
    return _settings
