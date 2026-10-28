"""Milestone 1: complete the typed decisions described in ASSIGNMENT.md."""

# These imports are intentionally supplied for TODOs 1.1 through 1.3.
# ruff: noqa: F401

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .models import AgentResponse

ToolName = Literal[
    "search_lab_documents",
    "get_user_authorizations",
    "get_resource_status",
    "reserve_resource",
]


class DecisionModel(BaseModel):
    """All decision variants reject fields not declared by their schema."""

    model_config = ConfigDict(extra="forbid")


class ToolCallDecision(DecisionModel):
    """Worked example: a typed proposal to call one allowed tool."""

    kind: Literal["tool_call"]
    tool: ToolName
    arguments: dict[str, Any]
    reasoning_summary: str


class ClarificationDecision(DecisionModel):
    kind: Literal["clarification"]
    # TODO 1.1: Add question: str and missing_fields: list[str].
    question: str
    missing_fields: list[str]


class FinalDecision(DecisionModel):
    kind: Literal["final"]
    # TODO 1.2: Add response: AgentResponse.
    response: AgentResponse


# TODO 1.3: Wrap this union in Annotated[..., Field(discriminator="kind")].
AgentDecision = Annotated[
    ToolCallDecision | ClarificationDecision | FinalDecision,
    Field(discriminator="kind"),
]

DecisionAdapter = TypeAdapter(AgentDecision)
