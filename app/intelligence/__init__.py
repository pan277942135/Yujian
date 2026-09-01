"""Model Intelligence v1: evaluation-driven data production helpers.

The package is deliberately independent from the training and Dataset Freeze
state machines.  It consumes immutable evaluation artifacts and produces
analysis/task artifacts that an operator can review before starting a batch.
"""

from .confusion_analyzer import analyze_confusion, build_confusion_report, write_confusion_report

__all__ = ["analyze_confusion", "build_confusion_report", "write_confusion_report"]
