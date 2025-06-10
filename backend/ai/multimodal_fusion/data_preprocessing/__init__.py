"""
Data Preprocessing Module Initialization
"""

from .metadata_processor import MetadataProcessor

# Import text processors with fallback
try:
    from .minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced text processor not available: {e}")
    ENHANCED_PROCESSOR_AVAILABLE = False

try:
    from .text_processor import TextProcessor
    BASIC_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Basic text processor not available: {e}")
    BASIC_PROCESSOR_AVAILABLE = False
    # Use minimal processor as fallback
    if ENHANCED_PROCESSOR_AVAILABLE:
        TextProcessor = MinimalEnhancedTextProcessor

__all__ = ["MetadataProcessor"]

if ENHANCED_PROCESSOR_AVAILABLE:
    __all__.append("MinimalEnhancedTextProcessor")
if BASIC_PROCESSOR_AVAILABLE:
    __all__.append("TextProcessor")
