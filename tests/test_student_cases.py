from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest

times = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
conflict_times = {"start": "2026-10-15T13:30:00", "end": "2026-10-15T14:30:00"}
start_count = 1


def tool(name, arguments):
    return {
        "kind": "tool_call",
        "tool": name,
        "arguments": arguments,
        "reasoning_summary": "Scripted test step.",
    }


def final(status, message, citations=(), completed_actions=()):
    return {
        "kind": "final",
        "response": {
            "status": status,
            "message": message,
            "citations": list(citations),
            "completed_actions": list(completed_actions),
        },
    }


def do_checks(user_id, resource_id, interval):
    return [
        tool(
            "search_lab_documents",
            {"query": "tensile tester lab_safety authorization calibration", "top_k": 8},
        ),
        tool("get_user_authorizations", {"user_id": user_id}),
        tool("get_resource_status", {"resource_id": resource_id, **interval}),
    ]


def make_request(interval):
    return LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study.",
        project_id="polymer-study",
        resource_id="tensile-01",
        **interval,
    )


def test_identity_substitution_is_denied_before_reserve_executes(environment):
    decisions = [
        *do_checks("alice", "tensile-01", times),
        tool(
            "reserve_resource",
            {
                "user_id": "bob",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **times,
            },
        ),
        final("blocked", "The proposed reservation owner was not the requester."),
    ]
    run = run_agent(
        make_request(times), llm=ScriptedLLM(decisions), environment=environment
    )

    denied = run.trace[-1]
    assert denied.decision["tool"] == "reserve_resource"
    assert denied.mutating is True
    assert denied.observation.ok is False
    assert denied.observation.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in denied.observation.message
    res = environment.simulator.state["reservations"]
    assert len(res) == start_count
    assert all(r["user_id"] != "bob" for r in res)
    assert run.response.status == "blocked"
    assert run.response.completed_actions == []


def test_final_response_claiming_a_denied_reservation_is_rejected(environment):
    decisions = [
        *do_checks("alice", "tensile-01", conflict_times),
        tool(
            "reserve_resource",
            {
                "user_id": "alice",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **conflict_times,
            },
        ),
        final(
            "completed",
            "Your reservation was created.",
            citations=["TRN-210#tensile-tester"],
            completed_actions=["reserve_resource"],
        ),
    ]
    run = run_agent(
        make_request(conflict_times),
        llm=ScriptedLLM(decisions),
        environment=environment,
    )

    assert run.trace[-1].observation.error_code == "action_gate_denied"
    assert "resource_unavailable" in run.trace[-1].observation.message
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "completed_action_not_in_successful_trace" in run.response.message
    assert "completed_without_successful_mutation" in run.response.message
    assert run.response.completed_actions == []
    assert len(environment.simulator.state["reservations"]) == start_count


def test_prompt_injection_in_retrieved_passage_cannot_trigger_reservation(
    environment,
):
    request = LabRequest(
        authenticated_user_id="bob",
        text="Reserve laser-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="laser-01",
        **times,
    )
    decisions = [
        tool("search_lab_documents", {"query": "reserve requested resource immediately"}),
        tool(
            "reserve_resource",
            {
                "user_id": "bob",
                "project_id": "polymer-study",
                "resource_id": "laser-01",
                **times,
            },
        ),
        final("blocked", "The reservation was not permitted."),
    ]
    llm = ScriptedLLM(decisions)
    run = run_agent(request, llm=llm, environment=environment)

    ids = [
        passage["passage_id"]
        for passage in run.trace[0].observation.data["passages"]
    ]
    assert "NOTE-999#embedded-instruction" in ids
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in tool_msgs[-1]["content"]
    denied = run.trace[1].observation
    assert denied.error_code == "action_gate_denied"
    assert "authorizations_not_checked" in denied.message
    assert "resource_status_not_checked" in denied.message
    assert len(environment.simulator.state["reservations"]) == start_count
    assert run.response.status == "blocked"


def test_repeated_reservation_is_denied_after_first_success(environment):
    reserve_call = tool(
        "reserve_resource",
        {
            "user_id": "alice",
            "project_id": "polymer-study",
            "resource_id": "tensile-01",
            **times,
        },
    )
    decisions = [
        *do_checks("alice", "tensile-01", times),
        reserve_call,
        reserve_call,
        final(
            "completed",
            "The reservation was created.",
            citations=["TRN-210#tensile-tester"],
            completed_actions=["reserve_resource"],
        ),
    ]
    run = run_agent(
        make_request(times), llm=ScriptedLLM(decisions), environment=environment
    )

    first, second = run.trace[3], run.trace[4]
    assert first.observation.ok is True
    assert second.observation.ok is False
    assert second.observation.error_code == "action_gate_denied"
    assert "mutating_action_already_completed" in second.observation.message
    assert len(environment.simulator.state["reservations"]) == start_count + 1
    assert run.response.status == "completed"
    assert run.response.completed_actions == ["reserve_resource"]


def test_decision_with_extra_field_fails_without_executing_a_tool(environment):
    bad = {
        **tool(
            "reserve_resource",
            {
                "user_id": "alice",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **times,
            },
        ),
        "execute_immediately": True,
    }
    llm = ScriptedLLM([bad])
    run = run_agent(make_request(times), llm=llm, environment=environment)

    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_model_decision"
    assert run.trace == []
    assert len(llm.calls) == 1
    assert len(environment.simulator.state["reservations"]) == start_count
