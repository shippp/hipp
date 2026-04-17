"""
Copyright (c) 2025 HIPP developers
Description: Shared result types for the KH-9 PC pipeline and QC report.
"""

from dataclasses import dataclass
from datetime import datetime

from hipp.kh9pc.restitution.strategy import RectificationStrategy


@dataclass
class StepResult:
    name: str
    status: str  # "ran" | "skipped" | "failed"
    started_at: datetime
    duration: float  # seconds
    error: str | None = None


@dataclass
class StrategyAttempt:
    strategy: RectificationStrategy | None
    success: bool
    failure_reason: str | None
