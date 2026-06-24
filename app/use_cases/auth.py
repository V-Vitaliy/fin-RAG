from typing import Protocol
import uuid
from app.models.domain import User, WorkspaceType
from app.repositories.uow import SqlAlchemyUnitOfWork


class PasswordHasherProtocol(Protocol):
    def hash_password(self, password: str) -> str:
        ...

    def verify_password(self, plain_password: str, password_hash: str) -> bool:
        ...


class AuthUseCase:
    def __init__(
        self,
        uow: SqlAlchemyUnitOfWork,
        password_hasher: PasswordHasherProtocol,
    ):
        self._uow = uow
        self._password_hasher = password_hasher

    async def register_user(
        self,
        email: str,
        password: str,
        workspace_name: str | None = None,
    ) -> User:
        async with self._uow as uow:
            existing_user = await uow.users.get_user_by_email(email)
            if existing_user:
                raise ValueError("User with this email already exists")

            workspace = await uow.workspaces.create_workspace(
                name=workspace_name or f"{email}'s workspace",
                ws_type=WorkspaceType.PRIVATE,
            )

            password_hash = self._password_hasher.hash_password(password)

            return await uow.users.create_user(
                email=email,
                password_hash=password_hash,
                workspace_id=workspace.id,
            )

    async def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> User:
        async with self._uow as uow:
            user = await uow.users.get_user_by_email(email)
            if not user:
                raise ValueError("Invalid email or password")

            if not self._password_hasher.verify_password(
                plain_password=password,
                password_hash=user.password_hash,
            ):
                raise ValueError("Invalid email or password")

            return user

    async def get_user_by_id(self, user_id: uuid.UUID) -> User:
        async with self._uow as uow:
            user = await uow.users.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")

            return user

    async def get_user_by_email(self, email: str) -> User:
        async with self._uow as uow:
            user = await uow.users.get_user_by_email(email)
            if not user:
                raise ValueError("User not found")

            return user