import asyncpg
import uuid
import logging
from typing import Any
import asyncio

logger = logging.getLogger(__name__)


class PoolManager:
    def __init__(self):
        # create a dict to store pool objects
        self._pools: dict[str, asyncpg.Pool] = {}
        self._dsn_by_name: dict[str, str] = {}
        self._conn_id_by_name: dict[str, str] = {}
        self._connect_lock = asyncio.Lock()

    def register(self, database_name: str, connection_string: str | None) -> None:
        if connection_string:
            self._dsn_by_name[database_name] = connection_string

    async def ensure_connection(self, database_name: str) -> str:
        existing = self._conn_id_by_name.get(database_name)
        if existing and existing in self._pools:
            return existing

        connection_string = self._dsn_by_name.get(database_name)
        if not connection_string:
            raise ValueError(f"No connection string registered for database: {database_name}")

        async with self._connect_lock:
            existing = self._conn_id_by_name.get(database_name)
            if existing and existing in self._pools:
                return existing

            conn_id = await self.connect(connection_string)
            self._conn_id_by_name[database_name] = conn_id
            return conn_id

    async def prefetch(self, database_names: list[str]) -> list[str]:
        opened: list[str] = []
        for database_name in database_names:
            if not database_name:
                continue
            try:
                await self.ensure_connection(database_name)
                opened.append(database_name)
            except Exception:
                logger.exception("Failed to prefetch database=%s", database_name)
        return opened

    async def connect(self, connection_string: str) -> str:
        conn_id = str(uuid.uuid4())[:8]
        try:
            logger.info("Creating pool for %s", connection_string)

            self._pools[conn_id] = await asyncpg.create_pool(
                dsn=connection_string,
                min_size=1,
                max_size=3,
                timeout=10,
                command_timeout=300,
                max_inactive_connection_lifetime=300,
            )

            logger.info("Pool created for conn_id=%s", conn_id)
            return conn_id
        except Exception:
            logger.exception("Failed to create pool=%s", conn_id)
            raise

    async def disconnect(self, conn_id: str) -> None:
        try:
            # find the pool by conn id
            pool = self._pools.pop(conn_id, None)

            # close the pool if it exists
            if pool:
                await pool.close()
        except Exception as e:
            logger.error(f"Failed to close pool for conn_id={conn_id}: {e}")
            raise

    async def query(
        self,
        conn_id: str,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict]:
        if conn_id not in self._pools and conn_id in self._dsn_by_name:
            conn_id = await self.ensure_connection(conn_id)

        # get the pool from the dict
        pool = self._pools.get(conn_id)

        # check if the pool exists
        if not pool:
            raise ValueError(f"No connection found for ID: {conn_id}")

        # retry once if the connection is stale
        for attempt in range(2):
            try:
                # take a connection from pool
                async with pool.acquire() as conn:
                    # open a read only transaction
                    tr = conn.transaction(readonly=True)
                    await tr.start()

                    try:
                        # run the sql query
                        rows = await conn.fetch(sql, *(params or ()))

                        # return rows as list of dict
                        return [dict(row) for row in rows]

                    finally:
                        # rollback because it is read only query
                        await tr.rollback()

            except (
                asyncpg.ConnectionDoesNotExistError,
                asyncpg.ConnectionFailureError,
                asyncpg.TooManyConnectionsError,
                OSError,
            ) as e:
                if attempt == 0:
                    logger.warning(
                        "query: connection error on attempt 1, retrying. conn_id=%s error=%r",
                        conn_id, e,
                    )
                    continue
                raise

    async def close_all(self) -> None:
        # loop through all conn ids
        for conn_id in list(self._pools):
            await self.disconnect(conn_id)
