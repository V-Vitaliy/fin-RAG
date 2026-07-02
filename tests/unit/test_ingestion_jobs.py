import uuid

import pytest

from app.services.jobs.ingestion_queue import IngestDocumentJob, IngestionQueue
from app.services.jobs.ingestion_runner import IngestionRunner


class FakeIngestUseCase:
    def __init__(self):
        self.jobs = []

    async def ingest(self, job):
        self.jobs.append(job)


@pytest.mark.asyncio
async def test_ingestion_queue_enqueue_and_get():
    queue = IngestionQueue()
    job = IngestDocumentJob(document_id=uuid.uuid4())

    await queue.enqueue(job)

    assert queue.qsize() == 1
    assert await queue.get() == job

    queue.task_done()
    await queue.join()


@pytest.mark.asyncio
async def test_ingestion_runner_run_once_processes_job():
    queue = IngestionQueue()
    use_case = FakeIngestUseCase()
    runner = IngestionRunner(queue=queue, ingest_use_case=use_case)

    job = IngestDocumentJob(document_id=uuid.uuid4())
    await queue.enqueue(job)

    await runner.run_once()
    await queue.join()

    assert use_case.jobs == [job]