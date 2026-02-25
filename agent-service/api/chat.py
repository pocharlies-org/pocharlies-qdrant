"""Chat API — REST and WebSocket endpoints for the supervisor agent."""

import logging
import uuid

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger("agent.api.chat")

router = APIRouter(tags=["Chat"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat message."""

    message: str = Field(..., description="User message text")
    thread_id: str | None = Field(
        None, description="Conversation thread ID. Auto-generated if omitted."
    )


class ChatResponse(BaseModel):
    """Agent response payload."""

    thread_id: str = Field(..., description="Conversation thread ID")
    message: str = Field(..., description="Agent reply text")
    tool_calls: list[dict] = Field(
        default_factory=list, description="Tool calls made during processing"
    )


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------

@router.post("/api/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Send a message to the agent and receive the full response (non-streaming)."""
    graph = request.app.state.supervisor
    thread_id = body.thread_id or str(uuid.uuid4())

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        logger.exception("Graph invocation failed for thread %s", thread_id)
        raise exc

    # Extract the last AI message from the result
    ai_message = ""
    tool_calls: list[dict] = []
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            ai_message = msg.content if isinstance(msg.content, str) else ""
            tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["args"],
                }
                for tc in (msg.tool_calls or [])
            ]
            break

    return ChatResponse(
        thread_id=thread_id,
        message=ai_message,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/api/chat/ws")
async def chat_ws(websocket: WebSocket) -> None:
    """Stream agent responses over a WebSocket connection.

    Client sends: ``{"type": "message", "content": "...", "thread_id": "..."}``

    Server streams frames with ``type`` in
    ``token | tool_call | tool_result | done | error``.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            # Validate incoming message
            if data.get("type") != "message":
                await websocket.send_json(
                    {"type": "error", "message": "Expected type 'message'"}
                )
                continue

            content = data.get("content", "")
            thread_id = data.get("thread_id") or str(uuid.uuid4())

            graph = websocket.app.state.supervisor

            try:
                async for event in graph.astream_events(
                    {"messages": [HumanMessage(content=content)]},
                    config={"configurable": {"thread_id": thread_id}},
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk and isinstance(chunk.content, str) and chunk.content:
                            await websocket.send_json(
                                {"type": "token", "content": chunk.content}
                            )

                    elif kind == "on_tool_start":
                        await websocket.send_json(
                            {
                                "type": "tool_call",
                                "name": event.get("name", ""),
                                "args": event["data"].get("input", {}),
                            }
                        )

                    elif kind == "on_tool_end":
                        output = event["data"].get("output", "")
                        # output may be a ToolMessage or a string; normalise
                        if hasattr(output, "content"):
                            output = output.content
                        await websocket.send_json(
                            {
                                "type": "tool_result",
                                "name": event.get("name", ""),
                                "content": str(output),
                            }
                        )

                # Signal that the agent turn is complete
                await websocket.send_json(
                    {"type": "done", "thread_id": thread_id}
                )

            except Exception as exc:
                logger.exception(
                    "Streaming error for thread %s: %s", thread_id, exc
                )
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("Unexpected WebSocket error: %s", exc)
