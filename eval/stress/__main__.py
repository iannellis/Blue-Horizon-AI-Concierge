"""Entry point for ``python -m eval.stress``."""

import asyncio
import platform

from eval.stress.runner import run_stress

if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_stress())
