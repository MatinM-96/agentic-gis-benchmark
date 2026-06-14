import os
import sys
import asyncio
import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_core.tools import StructuredTool
from backend.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _server_params() -> StdioServerParameters:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    server_script = os.path.join(os.path.dirname(current_dir), "server", "run.py")

    return StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=dict(os.environ),
        cwd=project_root,
    )


class MCPClientManager:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    async def _run(self) -> None:
        try:
            async with stdio_client(_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    logger.info("MCP session ready")
                    await self._stop.wait()

        except Exception:
            logger.exception("MCP session crashed")
            self._ready.set()

        finally:
            self._session = None
            async with self._lock:
                self._task = None

    async def connect(self) -> None:
        while True:
            async with self._lock:
                needs_new_task = (
                    self._task is None or self._task.done()
                )

                if needs_new_task:
                    self._ready = asyncio.Event()
                    self._stop = asyncio.Event()
                    self._task = asyncio.create_task(self._run())

                ready = self._ready

            await ready.wait()

            if self._session is not None:
                return

            async with self._lock:
                if self._task is not None and self._task.done():
                    self._task = None

            raise RuntimeError("MCP session not available")

    async def close(self) -> None:
        async with self._lock:
            task = self._task
            self._stop.set()
            self._ready.set()

        if task is not None:
            try:
                await task
            finally:
                async with self._lock:
                    self._task = None
                    self._session = None
                    self._ready = asyncio.Event()
                    self._stop = asyncio.Event()
        else:
            async with self._lock:
                self._session = None
                self._ready = asyncio.Event()
                self._stop = asyncio.Event()

    async def call(self, tool_name: str, args: dict) -> str:
        if self._session is None:
            await self.connect()

        if self._session is None:
            raise RuntimeError("MCP session not available")

        try:
            res = await self._session.call_tool(tool_name, arguments=args)
        except Exception:
            logger.exception("Tool call failed: %s", tool_name)
            async with self._lock:
                self._session = None
                if self._task is not None and self._task.done():
                    self._task = None
            raise

        return res.content[0].text if res.content else ""


mcp_client = MCPClientManager()


async def mcp_call(tool_name: str, args: dict) -> str:
    return await mcp_client.call(tool_name, args)


























