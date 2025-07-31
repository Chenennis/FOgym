"""
灵活性报价(FlexOffer)框架

该框架实现了完整的灵活性报价流程，包括：
1. 使用强化学习生成FO (fo_generate)
2. 聚合FO (fo_aggregate)
3. 交易FO (fo_trading)
4. 分解聚合后的FO (fo_schedule - disaggregator)
5. 调度执行FO (fo_schedule - scheduler)

主程序入口: run_fo_pipeline.py
"""

import logging

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 版本信息
__version__ = '0.1.0' 