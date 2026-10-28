"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context
from .tools import MUTATING_TOOLS

fields_to_check = ("project_id", "resource_id", "start", "end")

checks = (
    ("arguments_complete", "reservation_arguments_incomplete"),
    ("policy_retrieved", "policy_not_retrieved"),
    ("authorizations_checked", "authorizations_not_checked"),
    ("resource_status_checked", "resource_status_not_checked"),
    ("project_authorized", "project_not_authorized"),
    ("qualification_current", "required_qualification_missing"),
    ("resource_available", "resource_unavailable"),
    ("calibration_current", "calibration_not_current"),
    ("maintenance_clear", "resource_under_maintenance"),
    ("after_hours_approval_satisfied", "after_hours_approval_missing"),
)


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    """TODO 3.1: Prevent identity and resolved-field substitution."""
    args = decision.arguments
    errors: list[str] = []

    if "user_id" in args and args["user_id"] != request.authenticated_user_id:
        errors.append("user_id_must_match_authenticated_user")

    for field in fields_to_check:
        if field not in args:
            continue
        expected = getattr(request, field)
        if expected is None:
            if decision.tool in MUTATING_TOOLS:
                errors.append(f"{field}_unresolved_in_request")
        elif args[field] != expected:
            errors.append(f"{field}_does_not_match_request")

    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    """TODO 3.2: Return an error for every required context fact that is false."""
    return [
        msg for name, msg in checks if not getattr(context, name)
    ]


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    """TODO 3.3: Validate citations, claimed actions, and final status."""
    errors: list[str] = []

    bad_citations = sorted(set(response.citations) - retrieved_passage_ids(trace))
    if bad_citations:
        errors.append(f"citations_not_retrieved: {', '.join(bad_citations)}")

    succeeded = [
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    ]
    if any(a not in succeeded for a in response.completed_actions):
        errors.append("completed_action_not_in_successful_trace")

    if response.status == "completed":
        if not succeeded:
            errors.append("completed_without_successful_mutation")
        if not response.completed_actions:
            errors.append("completed_without_claimed_action")
    else:
        if response.completed_actions:
            errors.append("completed_actions_without_completed_status")
        if succeeded:
            errors.append("successful_mutation_not_reported")

    return errors
