from __future__ import annotations

import logging

from app.services.jobs.ingestion_queue import IngestionQueue

logger = logging.getLogger(__name__)


class IngestionRunner:
    def __init__(self, queue: IngestionQueue, ingest_use_case):
        self._queue = queue
        self._ingest_use_case = ingest_use_case
        self._running = False

    async def run_forever(self) -> None:
        self._running = True
        logger.info("Ingestion runner started.")

        while self._running:
            job = await self._queue.get()

            try:
                await self._ingest_use_case.ingest(job)
            except Exception:
                logger.exception("Unhandled ingestion job failure: document_id=%s", job.document_id)
            finally:
                self._queue.task_done()

        logger.info("Inestion runner stopped.")

    async def run_once(self) -> None:
        job = await self._queue.get()

        try:
            await self._ingest_use_case.ingest(job)
        except Exception:
            logger.exception("Unhandled ingestion job failure: document_id=%s", job.document_id)
        finally:
            self._queue.task_done()

    def stop(self) -> None:
        self._running = False