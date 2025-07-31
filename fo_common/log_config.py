import logging
import os
from enum import Enum

class LogVerbosity(Enum):
    """日志详细程度枚举"""
    MINIMAL = "minimal"      # 最简信息：只显示关键进度
    BRIEF = "brief"          # 简略信息：合并同类信息到一行
    DETAILED = "detailed"    # 详细信息：当前的完整日志
    DEBUG = "debug"          # 调试信息：包含所有调试信息

class LogConfig:
    """全局日志配置管理器"""
    
    _verbosity = LogVerbosity.BRIEF  # 默认使用简略模式
    _initialized = False
    
    @classmethod
    def set_verbosity(cls, verbosity: LogVerbosity):
        """设置日志详细程度"""
        cls._verbosity = verbosity
        cls._update_logging_levels()
    
    @classmethod
    def get_verbosity(cls) -> LogVerbosity:
        """获取当前日志详细程度"""
        return cls._verbosity
    
    @classmethod
    def is_minimal(cls) -> bool:
        """是否为最简模式"""
        return cls._verbosity == LogVerbosity.MINIMAL
    
    @classmethod
    def is_brief(cls) -> bool:
        """是否为简略模式"""
        return cls._verbosity == LogVerbosity.BRIEF
    
    @classmethod
    def is_detailed(cls) -> bool:
        """是否为详细模式"""
        return cls._verbosity == LogVerbosity.DETAILED
    
    @classmethod
    def is_debug(cls) -> bool:
        """是否为调试模式"""
        return cls._verbosity == LogVerbosity.DEBUG
    
    @classmethod
    def _update_logging_levels(cls):
        """根据详细程度更新日志级别"""
        if cls._verbosity == LogVerbosity.MINIMAL:
            # 最简模式：只显示WARNING及以上
            logging.getLogger().setLevel(logging.WARNING)
        elif cls._verbosity == LogVerbosity.BRIEF:
            # 简略模式：显示INFO，但某些模块使用DEBUG
            logging.getLogger().setLevel(logging.INFO)
            # 将重复性日志模块设为DEBUG级别
            logging.getLogger("FlexScheduler").setLevel(logging.WARNING)
            logging.getLogger("fo_generate.multi_agent_env").setLevel(logging.INFO)
        elif cls._verbosity == LogVerbosity.DETAILED:
            # 详细模式：显示所有INFO
            logging.getLogger().setLevel(logging.INFO)
        elif cls._verbosity == LogVerbosity.DEBUG:
            # 调试模式：显示所有DEBUG
            logging.getLogger().setLevel(logging.DEBUG)
    
    @classmethod
    def init_from_env(cls):
        """从环境变量初始化日志配置"""
        if cls._initialized:
            return
            
        verbosity_str = os.environ.get("FO_LOG_VERBOSITY", "brief").lower()
        try:
            verbosity = LogVerbosity(verbosity_str)
            cls.set_verbosity(verbosity)
        except ValueError:
            print(f"警告: 无效的日志详细程度 '{verbosity_str}'，使用默认值 'brief'")
            cls.set_verbosity(LogVerbosity.BRIEF)
        
        cls._initialized = True

def log_info_brief(logger, message: str, condition: bool = True):
    """简略模式的INFO日志 - 只在简略或详细模式下显示"""
    if condition and (LogConfig.is_brief() or LogConfig.is_detailed() or LogConfig.is_debug()):
        logger.info(message)

def log_info_detailed(logger, message: str, condition: bool = True):
    """详细模式的INFO日志 - 只在详细模式下显示"""
    if condition and (LogConfig.is_detailed() or LogConfig.is_debug()):
        logger.info(message)

def log_debug_conditional(logger, message: str, condition: bool = True):
    """条件调试日志"""
    if condition and LogConfig.is_debug():
        logger.debug(message)

def log_progress(logger, message: str):
    """进度日志 - 在所有模式下都显示"""
    if LogConfig.is_minimal():
        # 最简模式使用WARNING级别确保显示
        logger.warning(f"[进度] {message}")
    else:
        logger.info(message)

# 初始化日志配置
LogConfig.init_from_env() 