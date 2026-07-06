from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.models.schemas import TokenPayload


class PasswordHasher:
    def __init__(self) -> None:
        self._pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
        )

    def hash_password(self, password: str) -> str:
        return self._pwd_context.hash(password)

    def verify_password(self, plain_password: str, password_hash: str) -> bool:
        return self._pwd_context.verify(plain_password, password_hash)


def create_access_token(
    *,
    subject: str | UUID,
    workspace_id: str | UUID,
    expires_delta: timedelta | None = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload = {
        "sub": str(subject),
        "workspace_id": str(workspace_id),
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc

    sub = payload.get("sub")
    workspace_id = payload.get("workspace_id")

    if not sub:
        raise ValueError("Token subject is missing")

    return TokenPayload(
        sub=str(sub),
        workspace_id=UUID(str(workspace_id)) if workspace_id else None,
    )