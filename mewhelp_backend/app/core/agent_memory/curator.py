from __future__ import annotations


class MemoryCurator:
    """Phase 1 placeholder for dedupe, conflict detection, and version policy."""

    def should_keep(self, *_args: object, **_kwargs: object) -> bool:
        return True
