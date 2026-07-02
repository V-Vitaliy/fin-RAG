from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class IngestDocumentJob:
    document_id: UUID


class IngestionQueue:
    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue[IngestDocumentJob] = asyncio.Queue(maxsize=maxsize)

    async def enqueue(self, job: IngestDocumentJob) -> None:
        await self._queue.put(job)

    async def get(self) -> IngestDocumentJob:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def qsize(self) -> int:
        return self._queue.qsize()