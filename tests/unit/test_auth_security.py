from __future__ import annotations

from uuid import uuid4

from app.core.security import create_access_token, decode_access_token


def test_create_and_decode_access_token():
    user_id = uuid4()
    workspace_id = uuid4()

    token = create_access_token(
        subject=user_id,
        workspace_id=workspace_id,
    )

    payload = decode_access_token(token)

    assert payload.sub == str(user_id)
    assert payload.workspace_id == workspace_id