"""
Data processing package for Scientific Aspect Extraction project.

This package contains modules for:
- Data preparation and preprocessing
- Statistical analysis of datasets
- Data format conversions
"""

from src.data.data_preparation import DataPreparator
from src.data.calculate_statistics import StatisticsCalculator

__all__ = ["DataPreparator", "StatisticsCalculator"]