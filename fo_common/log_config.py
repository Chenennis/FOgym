import logging
import os
from enum import Enum

class LogVerbosity(Enum):
    """log verbosity enumeration"""
    MINIMAL = "minimal"      # minimal information: only show critical progress
    BRIEF = "brief"          # brief information: merge similar information into one line
    DETAILED = "detailed"    # detailed information: current complete log
    DEBUG = "debug"          # debug information: include all debug information

class LogConfig:
    """global log configuration manager"""
    
    _verbosity = LogVerbosity.BRIEF  # default use brief mode
    _initialized = False
    
    @classmethod
    def set_verbosity(cls, verbosity: LogVerbosity):
        """set log verbosity"""
        cls._verbosity = verbosity
        cls._update_logging_levels()
    
    @classmethod
    def get_verbosity(cls) -> LogVerbosity:
        """get current log verbosity"""
        return cls._verbosity
    
    @classmethod
    def is_minimal(cls) -> bool:
        """is minimal mode"""
        return cls._verbosity == LogVerbosity.MINIMAL
    
    @classmethod
    def is_brief(cls) -> bool:
        """is brief mode"""
        return cls._verbosity == LogVerbosity.BRIEF
    
    @classmethod
    def is_detailed(cls) -> bool:
        """is detailed mode"""
        return cls._verbosity == LogVerbosity.DETAILED
    
    @classmethod
    def is_debug(cls) -> bool:
        """is debug mode"""
        return cls._verbosity == LogVerbosity.DEBUG
    
    @classmethod
    def _update_logging_levels(cls):
        """update log levels according to verbosity"""
        if cls._verbosity == LogVerbosity.MINIMAL:
            # minimal mode: only show WARNING and above
            logging.getLogger().setLevel(logging.WARNING)
        elif cls._verbosity == LogVerbosity.BRIEF:
            # brief mode: show INFO, but some modules use DEBUG
            logging.getLogger().setLevel(logging.INFO)
            # set repetitive log modules to DEBUG level
            logging.getLogger("FlexScheduler").setLevel(logging.WARNING)
            logging.getLogger("fo_generate.multi_agent_env").setLevel(logging.INFO)
        elif cls._verbosity == LogVerbosity.DETAILED:
            # detailed mode: show all INFO
            logging.getLogger().setLevel(logging.INFO)
        elif cls._verbosity == LogVerbosity.DEBUG:
            # debug mode: show all DEBUG
            logging.getLogger().setLevel(logging.DEBUG)
    
    @classmethod
    def init_from_env(cls):
        """initialize log configuration from environment variables"""
        if cls._initialized:
            return
            
        verbosity_str = os.environ.get("FO_LOG_VERBOSITY", "brief").lower()
        try:
            verbosity = LogVerbosity(verbosity_str)
            cls.set_verbosity(verbosity)
        except ValueError:
            print(f"warning: invalid log verbosity '{verbosity_str}', using default 'brief'")
            cls.set_verbosity(LogVerbosity.BRIEF)
        
        cls._initialized = True

def log_info_brief(logger, message: str, condition: bool = True):
    """brief mode INFO log - only show in brief or detailed mode"""
    if condition and (LogConfig.is_brief() or LogConfig.is_detailed() or LogConfig.is_debug()):
        logger.info(message)

def log_info_detailed(logger, message: str, condition: bool = True):
    """detailed mode INFO log - only show in detailed mode"""
    if condition and (LogConfig.is_detailed() or LogConfig.is_debug()):
        logger.info(message)

def log_debug_conditional(logger, message: str, condition: bool = True):
    """conditional debug log"""
    if condition and LogConfig.is_debug():
        logger.debug(message)

def log_progress(logger, message: str):
    """progress log - show in all modes"""
    if LogConfig.is_minimal():
        # minimal mode use WARNING level to ensure display
        logger.warning(f"[progress] {message}")
    else:
        logger.info(message)

# initialize log configuration
LogConfig.init_from_env() 