from io import BytesIO
from typing import BinaryIO
from types_aiobotocore_s3.client import S3Client


class S3StorageRepository:
    """ wrapper around an already-opened aioboto3 S3 client."""

    def __init__(self, client: "S3Client", bucket_name: str):
        self._client = client
        self._bucket_name = bucket_name

    async def upload_file(self, file_obj: BinaryIO, object_key: str) -> str:
        await self._client.upload_fileobj(
            file_obj,
            self._bucket_name,
            object_key,
        )
        return object_key

    async def get_bytes(self, object_key: str) -> bytes:
        buffer = BytesIO()
        await self._client.download_fileobj(
            self._bucket_name,
            object_key,
            buffer,
        )
        return buffer.getvalue()

    async def delete_file(self, object_key: str) -> None:
        await self._client.delete_object(
            Bucket=self._bucket_name,
            Key=object_key,
        )
