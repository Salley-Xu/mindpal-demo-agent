from typing import Any, Optional


LEVEL_0 = "level_0"
LEVEL_1 = "level_1"
LEVEL_2 = "level_2"
LEVEL_3 = "level_3"

RISK_LEVEL_ORDER = {
    LEVEL_0: 0,
    LEVEL_1: 1,
    LEVEL_2: 2,
    LEVEL_3: 3,
}

_ALIASES = {
    LEVEL_0: LEVEL_0,
    LEVEL_1: LEVEL_1,
    LEVEL_2: LEVEL_2,
    LEVEL_3: LEVEL_3,
    "0": LEVEL_0,
    "1": LEVEL_1,
    "2": LEVEL_2,
    "3": LEVEL_3,
    "low": LEVEL_0,
    "medium": LEVEL_1,
    "high": LEVEL_3,
    "normal": LEVEL_0,
    "warning": LEVEL_1,
    "warning_low": LEVEL_1,
    "warning_high": LEVEL_2,
    "urgent": LEVEL_3,
    None: LEVEL_0,
}

_LEVEL_TO_BAND = {
    LEVEL_0: "low",
    LEVEL_1: "medium",
    LEVEL_2: "medium",
    LEVEL_3: "high",
}

_LEVEL_TO_LABEL = {
    LEVEL_0: "Level 0",
    LEVEL_1: "Level 1",
    LEVEL_2: "Level 2",
    LEVEL_3: "Level 3",
}


def normalize_risk_level(level: Optional[Any], default: str = LEVEL_0) -> str:
    if isinstance(level, str):
        normalized = level.strip().lower()
    else:
        normalized = level
    return _ALIASES.get(normalized, default)


def risk_level_index(level: Optional[Any]) -> int:
    canonical = normalize_risk_level(level)
    return RISK_LEVEL_ORDER[canonical]


def risk_level_band(level: Optional[Any]) -> str:
    canonical = normalize_risk_level(level)
    return _LEVEL_TO_BAND[canonical]


def risk_level_label(level: Optional[Any]) -> str:
    canonical = normalize_risk_level(level)
    return _LEVEL_TO_LABEL[canonical]


def is_emergency_risk(level: Optional[Any]) -> bool:
    return normalize_risk_level(level) == LEVEL_3


def is_high_support_risk(level: Optional[Any]) -> bool:
    return risk_level_index(level) >= 2


def is_non_low_risk(level: Optional[Any]) -> bool:
    return risk_level_index(level) >= 1
