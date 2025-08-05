"""
FOgym Framework

This framework implements a complete flexibility offer process, including:
1. Using reinforcement learning to generate FO (fo_generate)
2. Aggregating FO (fo_aggregate)
3. Trading FO (fo_trading)
4. Disaggregating the aggregated FO (fo_schedule - disaggregator)


Main program entry: run_fo_pipeline.py
"""

import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Version information
__version__ = '0.1.0' 