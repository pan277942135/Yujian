"""Model Intelligence v1: evaluation-driven data production helpers.

The package is deliberately independent from the training and Dataset Freeze
state machines.  It consumes immutable evaluation artifacts and produces
analysis/task artifacts that an operator can review before starting a batch.
"""

from .confusion_analyzer import (
    analyze_confusion,
    analyze_confusion_matrix,
    build_confusion_report,
    generate_confusion_report,
    write_confusion_report,
)
from .data_gap_analyzer import analyze_data_gaps, analyze_gaps, build_data_gap_report
from .hard_case_miner import build_hard_case_set, mine_hard_cases
from .task_generator import build_collection_task, generate_collection_task, generate_task

__all__ = [
    "analyze_confusion",
    "analyze_confusion_matrix",
    "analyze_data_gaps",
    "analyze_gaps",
    "build_collection_task",
    "build_confusion_report",
    "build_data_gap_report",
    "build_hard_case_set",
    "generate_collection_task",
    "generate_confusion_report",
    "generate_task",
    "mine_hard_cases",
    "write_confusion_report",
]
