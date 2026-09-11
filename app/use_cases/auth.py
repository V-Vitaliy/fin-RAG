from typing import Protocol
import uuid
import asyncio

from sqlalchemy.exc import IntegrityError

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
        *,
        enable_company_domain_workspaces: bool = True,
        public_email_domains: set[str] | None = None,
    ):
        self._uow = uow
        self._password_hasher = password_hasher
        self._enable_company_domain_workspaces = enable_company_domain_workspaces
        self._public_email_domains = public_email_domains or set()

    @staticmethod
    def _normalize_email(email: str) -> str:
        normalized = str(email or "").strip().lower()

        if "@" not in normalized:
            raise ValueError("Invalid email")

        return normalized

    @staticmethod
    def _email_domain(email: str) -> str:
        return email.rsplit("@", 1)[1].strip().lower()

    def _should_use_company_domain_workspace(self, domain: str) -> bool:
        if not self._enable_company_domain_workspaces:
            return False

        if not domain:
            return False

        return domain not in self._public_email_domains

    async def register_user(
        self,
        email: str,
        password: str,
        workspace_name: str | None = None,
    ) -> User:
        email = self._normalize_email(email)
        domain = self._email_domain(email)

        use_company_workspace = self._should_use_company_domain_workspace(domain)

        async with self._uow as uow:
            existing_user = await uow.users.get_user_by_email(email)
            if existing_user:
                raise ValueError("User with this email already exists")

            if use_company_workspace:
                workspace = await uow.companies.get_workspace_by_email_domain(domain)

                if not workspace:
                    company_name = workspace_name or domain

                    try:
                        workspace = await uow.companies.create_company_workspace_for_domain(
                            company_name=company_name,
                            domain=domain,
                        )
                    except IntegrityError:
                        # Handles concurrent registration for the same company domain.
                        await uow.session.rollback()

                        workspace = await uow.companies.get_workspace_by_email_domain(domain)
                        if not workspace:
                            raise
            else:
                workspace = await uow.workspaces.create_workspace(
                    name=workspace_name or f"{email}'s workspace",
                    ws_type=WorkspaceType.PRIVATE,
                )

            password_hash = await asyncio.to_thread(
                self._password_hasher.hash_password,
                password,
            )

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
        email = self._normalize_email(email)

        async with self._uow as uow:
            user = await uow.users.get_user_by_email(email)
            if not user:
                raise ValueError("Invalid email or password")

            password_ok = await asyncio.to_thread(
                self._password_hasher.verify_password,
                plain_password=password,
                password_hash=user.password_hash,
            )

            if not password_ok:
                raise ValueError("Invalid email or password")

            return user

    async def get_user_by_id(self, user_id: uuid.UUID) -> User:
        async with self._uow as uow:
            user = await uow.users.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")

            return user

    async def get_user_by_email(self, email: str) -> User:
        email = self._normalize_email(email)

        async with self._uow as uow:
            user = await uow.users.get_user_by_email(email)
            if not user:
                raise ValueError("User not found")

            return user