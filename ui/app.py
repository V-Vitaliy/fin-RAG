from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import chainlit as cl

from ui.backend_client import BackendClient


backend = BackendClient()


def _format_tool_arguments(arguments: dict[str, Any] | None) -> str:
    if not arguments:
        return "No arguments."

    return "\n".join(
        f"- **{html.escape(str(key))}:** `{html.escape(str(value))}`"
        for key, value in arguments.items()
    )


def _format_json_block(value: Any) -> str:
    try:
        return (
            "```json\n"
            + json.dumps(value, ensure_ascii=False, indent=2, default=str)
            + "\n```"
        )
    except TypeError:
        return f"```text\n{value}\n```"

def _format_source_documents(source_documents: list[dict[str, Any]] | None) -> str:
    if not source_documents:
        return ""

    lines = ["### Sources"]

    for source in source_documents:
        filename = html.escape(str(source.get("filename") or "document"))
        url = str(source.get("url") or "")
        markers = ", ".join(str(marker) for marker in source.get("markers") or [])
        pages = source.get("pages") or []
        pages_text = ", ".join(f"p.{page}" for page in pages) if pages else ""

        details: list[str] = []
        if markers:
            details.append(f"markers: {html.escape(markers)}")
        if pages_text:
            details.append(html.escape(pages_text))

        suffix = f" — {'; '.join(details)}" if details else ""

        if url:
            lines.append(f"- [{filename}]({url}){suffix}")
        else:
            lines.append(f"- {filename}{suffix}")

    return "\n".join(lines)


def _get_uploaded_file_info(element: Any) -> tuple[Path | None, str | None, str | None]:
    path_raw = getattr(element, "path", None)
    name = getattr(element, "name", None)
    mime = (
        getattr(element, "mime", None)
        or getattr(element, "mime_type", None)
        or getattr(element, "content_type", None)
    )

    if not path_raw:
        return None, name, mime

    path = Path(str(path_raw))
    if not name:
        name = path.name

    return path, str(name), str(mime or "application/pdf")


async def _upload_message_files_stream(
    *,
    access_token: str,
    message: cl.Message,
) -> list[str]:
    ready_document_ids: list[str] = []

    for element in message.elements or []:
        path, filename, content_type = _get_uploaded_file_info(element)

        if not path or not path.exists():
            await cl.Message(
                content=f"Could not read uploaded file: {filename or 'unknown file'}"
            ).send()
            continue

        if filename and not filename.lower().endswith(".pdf"):
            await cl.Message(
                content=f"Skipped `{filename}`. Only PDF files are supported."
            ).send()
            continue

        status_message = cl.Message(
            content=f"_Uploading and processing `{filename or path.name}`..._"
        )
        await status_message.send()

        created_document_id: str | None = None

        async for event in backend.upload_document_stream(
            access_token=access_token,
            file_path=path,
            filename=filename,
            content_type=content_type,
        ):
            event_name = event.event
            data = event.data or {}

            if event_name == "stage":
                message_text = str(data.get("message") or data.get("stage") or "Processing")
                status_message.content = f"_{message_text}..._"
                await status_message.update()

            elif event_name == "document_created":
                created_document_id = str(data.get("document_id") or "")
                status_message.content = (
                    f"_Document uploaded. Processing started: `{filename or path.name}`..._"
                )
                await status_message.update()

            elif event_name == "ready":
                document_id = str(data.get("document_id") or created_document_id or "")
                if document_id:
                    ready_document_ids.append(document_id)

                status_message.content = (
                    f"_Document `{filename or path.name}` is ready._"
                )
                await status_message.update()

            elif event_name == "error":
                error_message = (
                    data.get("message")
                    or data.get("detail")
                    or "Document processing failed."
                )
                status_message.content = f"Document processing failed: {error_message}"
                await status_message.update()
                raise RuntimeError(str(error_message))

            elif event_name == "done":
                continue

    return ready_document_ids


def _format_uploaded_documents(uploaded: list[dict[str, Any]]) -> str:
    if not uploaded:
        return ""

    lines = ["Uploaded document(s):"]

    for document in uploaded:
        doc_id = document.get("id") or document.get("document_id")
        filename = document.get("filename") or document.get("doc_name") or "document"
        status = document.get("status") or "unknown"

        lines.append(f"- `{filename}` — status: `{status}`, id: `{doc_id}`")

    lines.append("")
    lines.append(
        "Indexing may take some time. Questions use documents that are already READY in your workspace."
    )

    return "\n".join(lines)


async def _send_documents_status(access_token: str) -> None:
    documents = await backend.list_documents(access_token=access_token)

    if not documents:
        await cl.Message(content="No documents found in this workspace.").send()
        return

    lines = ["Documents in your workspace:"]

    for document in documents:
        filename = document.get("filename") or document.get("doc_name") or "document"
        status = document.get("status") or "unknown"
        doc_id = document.get("id")
        lines.append(f"- `{filename}` — `{status}` — `{doc_id}`")

    await cl.Message(content="\n".join(lines)).send()


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    try:
        token = await backend.login(
            email=username,
            password=password,
        )
    except Exception:
        return None

    return cl.User(
        identifier=username,
        metadata={
            "access_token": token.access_token,
            "token_type": token.token_type,
        },
    )


@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")

    if not user:
        await cl.Message(
            content="Authentication failed. Please refresh and log in again."
        ).send()
        return

    token = (user.metadata or {}).get("access_token")
    if not token:
        await cl.Message(
            content="Backend token is missing. Please refresh and log in again."
        ).send()
        return

    cl.user_session.set("access_token", token)

    await cl.Message(
        content=(
            "Hi. Ask a question about indexed financial documents in your workspace.\n\n"
            "You can also upload PDF files here. After upload, they need to be indexed before the agent can use them.\n\n"
            "Commands:\n"
            "- `/documents` — show workspace documents and statuses"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    access_token = cl.user_session.get("access_token")

    if not access_token:
        await cl.Message(
            content="You are not authenticated. Please refresh and log in again."
        ).send()
        return

    question = (message.content or "").strip()

    if question == "/documents":
        await _send_documents_status(access_token)
        return

    ready_document_ids = await _upload_message_files_stream(
        access_token=access_token,
        message=message,
    )

    if not question:
        if ready_document_ids:
            await cl.Message(
                content="Document is ready. Send a question to analyze it."
            ).send()
        else:
            await cl.Message(content="Please enter a question or upload a PDF.").send()
        return

    status_message: cl.Message | None = None
    answer_message: cl.Message | None = None
    final_answer_shown = False
    source_documents: list[dict[str, Any]] = []

    try:
        async for event in backend.ask_stream(
                access_token=access_token,
                question=question,
                document_ids=ready_document_ids or None,
        ):
            event_name = event.event
            data = event.data or {}

            if event_name == "stage":
                stage = str(data.get("stage", "stage"))
                stage_message = str(data.get("message", stage))

                if stage in {"resolving_access", "planning", "thinking"}:
                    text = f"_{stage_message}..._"
                elif stage == "finalizing":
                    text = "_Preparing final answer and sources..._"
                else:
                    text = f"_{stage_message}_"

                if status_message is None:
                    status_message = cl.Message(content=text)
                    await status_message.send()
                else:
                    status_message.content = text
                    await status_message.update()

            elif event_name == "tool_call":
                tool_name = str(data.get("name", "tool"))

                text = f"_Using tool: `{tool_name}`..._"

                if status_message is None:
                    status_message = cl.Message(content=text)
                    await status_message.send()
                else:
                    status_message.content = text
                    await status_message.update()


            elif event_name == "tool_result":

                tool_name = str(data.get("name", "tool"))

                ok = data.get("ok", False)

                status = "finished" if ok else "returned an error"

                text = f"_Tool `{tool_name}` {status}._"

                if status_message is None:

                    status_message = cl.Message(content=text)

                    await status_message.send()

                else:

                    status_message.content = text

                    await status_message.update()



            elif event_name == "final_answer":

                answer = str(data.get("answer") or "")

                source_documents = data.get("source_documents") or []

                if status_message is not None:

                    try:

                        await status_message.remove()

                    except Exception:

                        status_message.content = "_Done._"

                        await status_message.update()

                if answer:
                    answer_message = cl.Message(content=answer)

                    await answer_message.send()

                    final_answer_shown = True



            elif event_name == "done":

                done_answer = data.get("answer")

                done_sources = data.get("source_documents") or source_documents

                if not final_answer_shown and done_answer:

                    if status_message is not None:

                        try:

                            await status_message.remove()

                        except Exception:

                            status_message.content = "_Done._"

                            await status_message.update()

                    answer_message = cl.Message(content=str(done_answer))

                    await answer_message.send()

                    final_answer_shown = True

                sources_block = _format_source_documents(done_sources)

                if sources_block:
                    await cl.Message(content=sources_block).send()

            elif event_name == "error":
                error_message = (
                        data.get("message")
                        or data.get("detail")
                        or "Unknown backend error."
                )

                await cl.Message(
                    content=f"Backend error: {error_message}"
                ).send()

            else:
                async with cl.Step(name=f"Backend event: {event_name}", type="run") as step:
                    step.output = _format_json_block(data)

    except Exception as exc:
        await cl.Message(
            content=f"UI/backend communication error: {exc}"
        ).send()

    finally:

        if answer_message is not None and final_answer_shown:
            await answer_message.update()