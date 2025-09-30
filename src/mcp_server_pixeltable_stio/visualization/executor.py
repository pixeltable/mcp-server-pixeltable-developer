"""Process pool executor for Pixeltable operations in visualization server."""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable


class ExecutorManager:
    """Manages process pool execution for Pixeltable operations."""

    def __init__(self):
        # Pool for Pixeltable operations (isolated processes)
        self.pool = ProcessPoolExecutor(max_workers=2)

    async def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run a Pixeltable operation in an isolated worker process.

        This prevents async/threading conflicts with Pixeltable.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, func, *args, **kwargs)

    def shutdown(self):
        """Shutdown the executor cleanly."""
        self.pool.shutdown(wait=True)


# Global executor manager
_manager = None


def get_executor_manager() -> ExecutorManager:
    """Get or create the global executor manager."""
    global _manager
    if _manager is None:
        _manager = ExecutorManager()
    return _manager
