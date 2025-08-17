"""
TensorFlow configuration and lazy loading to prevent memory issues.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Configure TensorFlow before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Keep oneDNN optimizations

# Global flag to track if TF is loaded
_tf_loaded = False
_tf_module = None

def configure_tensorflow():
    """Configure TensorFlow for optimal memory usage."""
    global _tf_loaded, _tf_module
    
    if _tf_loaded:
        return _tf_module
    
    try:
        import tensorflow as tf
        
        # Configure GPU memory growth if available
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
        else:
            # Force CPU-only in development to save memory
            tf.config.set_visible_devices([], "GPU")
            logger.info("Running TensorFlow in CPU-only mode")
        
        _tf_module = tf
        _tf_loaded = True
        logger.info("TensorFlow configured successfully")
        return tf
        
    except ImportError:
        logger.warning("TensorFlow not available")
        return None
    except Exception as e:
        logger.error(f"Failed to configure TensorFlow: {e}")
        return None

def get_tensorflow():
    """Get TensorFlow module with lazy loading."""
    return configure_tensorflow()

def is_reloader_process() -> bool:
    """Check if this is the reloader process to avoid double initialization."""
    return os.getenv("WATCHFILES_RELOADER") == "true"