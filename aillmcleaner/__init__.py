"""
LLMClean: An AI-powered Data Cleaning Library.

This package provides high-level utilities to clean, standardize, 
and impute pandas DataFrames using local Large Language Models (LLMs).
"""

from .core import (
    clean_text,
    clean_column,
    standardize_column,
    fill_missing,
    detect_anomalies,
    clean_dataframe,
)

__version__ = "0.2.0"
__author__ = "spanigrahidev"

# Define public API
__all__ = [
    "clean_text",
    "clean_column",
    "standardize_column",
    "fill_missing",
    "detect_anomalies",
    "clean_dataframe",
]
