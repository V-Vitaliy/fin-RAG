from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from app.core.config import settings
from app.infrastructure.postgres import build_postgres_engine, build_sessionmaker
from app.models.domain import Workspace, WorkspaceType
from app.repositories.uow import SqlAlchemyUnitOfWork


def resolve_workspace_id(raw_value: str | None) -> uuid.UUID:
    value = raw_value or settings.RAG_GLOBAL_WORKSPACE_ID

    if not value:
        raise ValueError(
            "Global workspace id is required. "
            "Set RAG_GLOBAL_WORKSPACE_ID or pass --workspace-id."
        )

    return uuid.UUID(str(value))


async def ensure_global_workspace(
    *,
    workspace_id: uuid.UUID,
    name: str,
    output_path: Path | None,
) -> dict:
    engine = build_postgres_engine()
    sessionmaker = build_sessionmaker(engine)

    try:
        uow_factory = lambda: SqlAlchemyUnitOfWork(sessionmaker)

        async with uow_factory() as uow:
            existing = await uow.workspaces.get_workspace(workspace_id)

            if existing:
                result = {
                    "created": False,
                    "workspace_id": str(existing.id),
                    "name": existing.name,
                    "type": existing.type.value,
                }
            else:
                workspace = Workspace(
                    id=workspace_id,
                    name=name,
                    type=WorkspaceType.GLOBAL,
                    company_id=None,
                )

                uow.session.add(workspace)
                await uow.session.flush()
                await uow.session.refresh(workspace)

                result = {
                    "created": True,
                    "workspace_id": str(workspace.id),
                    "name": workspace.name,
                    "type": workspace.type.value,
                }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result

    finally:
        await engine.dispose()


async def amain() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure that the configured global workspace exists."
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Global workspace UUID. Defaults to RAG_GLOBAL_WORKSPACE_ID.",
    )
    parser.add_argument(
        "--name",
        default="Global FinanceBench Workspace",
        help="Workspace display name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    workspace_id = resolve_workspace_id(args.workspace_id)

    result = await ensure_global_workspace(
        workspace_id=workspace_id,
        name=args.name,
        output_path=args.output,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())