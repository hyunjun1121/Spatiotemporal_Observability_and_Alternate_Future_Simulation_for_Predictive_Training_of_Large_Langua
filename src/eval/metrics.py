from typing import Dict

def instability_events_per_100k(events: int, steps: int) -> float:
    """Normalize instability events per 100k steps."""
    return 1e5 * events / max(1, steps)

def time_to_recover(trigger_step: int, recover_step: int) -> int:
    """Return steps to recover after a trigger."""
    return max(0, recover_step - trigger_step)

