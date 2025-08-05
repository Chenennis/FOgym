import logging
import os
from enum import Enum

class LogVerbosity(Enum):
    """Log verbosity level enumeration"""
    MINIMAL = "minimal"      # Minimal information: only show critical progress
    BRIEF = "brief"          # Brief information: combine similar information into one line
    DETAILED = "detailed"    # Detailed information: current complete logs
    DEBUG = "debug"          # Debug information: includes all debug information

class LogConfig:
    """Global logging configuration manager"""
    
    _verbosity = LogVerbosity.BRIEF  # Default to brief mode
    _initialized = False
    
    @classmethod
    def set_verbosity(cls, verbosity: LogVerbosity):
        """Set log verbosity level"""
        cls._verbosity = verbosity
        cls._update_logging_levels()
    
    @classmethod
    def get_verbosity(cls) -> LogVerbosity:
        """Get current log verbosity level"""
        return cls._verbosity
    
    @classmethod
    def is_minimal(cls) -> bool:
        """Check if in minimal mode"""
        return cls._verbosity == LogVerbosity.MINIMAL
    
    @classmethod
    def is_brief(cls) -> bool:
        """Check if in brief mode"""
        return cls._verbosity == LogVerbosity.BRIEF
    
    @classmethod
    def is_detailed(cls) -> bool:
        """Check if in detailed mode"""
        return cls._verbosity == LogVerbosity.DETAILED
    
    @classmethod
    def is_debug(cls) -> bool:
        """Check if in debug mode"""
        return cls._verbosity == LogVerbosity.DEBUG
    
    @classmethod
    def _update_logging_levels(cls):
        """Update logging levels based on verbosity"""
        if cls._verbosity == LogVerbosity.MINIMAL:
            # Minimal mode: only show WARNING and above
            logging.getLogger().setLevel(logging.WARNING)
        elif cls._verbosity == LogVerbosity.BRIEF:
            # Brief mode: show INFO, but some modules use DEBUG
            logging.getLogger().setLevel(logging.INFO)
            # Set repetitive log modules to DEBUG level
            logging.getLogger("FlexScheduler").setLevel(logging.WARNING)
            logging.getLogger("fo_generate.multi_agent_env").setLevel(logging.INFO)
        elif cls._verbosity == LogVerbosity.DETAILED:
            # Detailed mode: show all INFO
            logging.getLogger().setLevel(logging.INFO)
        elif cls._verbosity == LogVerbosity.DEBUG:
            # Debug mode: show all DEBUG
            logging.getLogger().setLevel(logging.DEBUG)
    
    @classmethod
    def init_from_env(cls):
        """Initialize log configuration from environment variables"""
        if cls._initialized:
            return
            
        verbosity_str = os.environ.get("FO_LOG_VERBOSITY", "brief").lower()
        try:
            verbosity = LogVerbosity(verbosity_str)
            cls.set_verbosity(verbosity)
        except ValueError:
            print(f"Warning: Invalid log verbosity level '{verbosity_str}', using default value 'brief'")
            cls.set_verbosity(LogVerbosity.BRIEF)
        
        cls._initialized = True

def log_info_brief(logger, message: str, condition: bool = True):
    """Brief mode INFO log - only shown in brief, detailed, or debug mode"""
    if condition and (LogConfig.is_brief() or LogConfig.is_detailed() or LogConfig.is_debug()):
        logger.info(message)

def log_info_detailed(logger, message: str, condition: bool = True):
    """Detailed mode INFO log - only shown in detailed or debug mode"""
    if condition and (LogConfig.is_detailed() or LogConfig.is_debug()):
        logger.info(message)

def log_debug_conditional(logger, message: str, condition: bool = True):
    """Conditional debug log"""
    if condition and LogConfig.is_debug():
        logger.debug(message)

def log_progress(logger, message: str):
    """Progress log - shown in all modes"""
    if LogConfig.is_minimal():
        # Use WARNING level in minimal mode to ensure display
        logger.warning(f"[Progress] {message}")
    else:
        logger.info(message)

# Initialize log configuration
LogConfig.init_from_env() 