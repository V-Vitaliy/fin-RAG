from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_auth_use_case, get_current_user
from app.core.security import create_access_token
from app.models.domain import User
from app.models.schemas import (
    LoginRequest,
    RegisterRequest,
    Token,
    UserResponse,
)
from app.use_cases.auth import AuthUseCase

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    auth_use_case: Annotated[AuthUseCase, Depends(get_auth_use_case)],
) -> UserResponse:
    try:
        user = await auth_use_case.register_user(
            email=str(payload.email),
            password=payload.password,
            workspace_name=payload.workspace_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return UserResponse.model_validate(user)


@router.post("/login", response_model=Token)
async def login(
    payload: LoginRequest,
    auth_use_case: Annotated[AuthUseCase, Depends(get_auth_use_case)],
) -> Token:
    try:
        user = await auth_use_case.authenticate_user(
            email=str(payload.email),
            password=payload.password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    token = create_access_token(
        subject=user.id,
        workspace_id=user.workspace_id,
    )

    return Token(
        access_token=token,
        token_type="bearer",
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    return UserResponse.model_validate(current_user)