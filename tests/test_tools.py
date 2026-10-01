import pytest

from app.tools import check_documents, check_policy_flags, estimate_processing
from app.trace import run_trace


@pytest.mark.parametrize(
    "categories,international,has_guests,attachments,expected",
    [
        (["lodging"], False, False, ["card_slip"], ["R1"]),
        (["meals"], False, True, ["meal_receipt"], ["R4"]),
        (["mileage"], True, False, ["mileage_log"], ["R5"]),
        (["lodging"], False, True, ["itemized_hotel_receipt"], []),
        (["meals"], False, True, ["meal_receipt", "attendee_list"], []),
        (["airfare"], True, False, ["flight_itinerary", "pre_approval"], []),
        (["airfare", "meals", "mileage"], False, False, [], ["R2", "R3", "R6"]),
    ],
)
def test_check_documents(categories, international, has_guests, attachments, expected):
    findings = check_documents(categories=categories, international=international,
                               has_guests=has_guests, attachments=attachments)
    assert [finding.rule_id for finding in findings] == expected
    assert all(finding.severity == "block" for finding in findings)
    assert all(finding.message and finding.source for finding in findings)


@pytest.mark.parametrize(
    "alcohol,total,attachments,expected",
    [
        (True, 100, [], [("R8", "flag")]),
        (False, 2500, [], [("R7", "remind")]),
        (False, 2500, ["manager_approval"], []),
        (False, 2000, [], []),
        (True, 2500, [], [("R7", "remind"), ("R8", "flag")]),
        (True, 2500, ["manager_approval"], [("R8", "flag")]),
    ],
)
def test_check_policy_flags(alcohol, total, attachments, expected):
    findings = check_policy_flags(mentions_alcohol=alcohol, total=total,
                                  attachments=attachments)
    assert [(finding.rule_id, finding.severity) for finding in findings] == expected


def test_estimate_processing_mileage():
    assert estimate_processing(categories=["mileage"]) == {
        "similar_claims": 2, "median_days": 2.5,
    }


def test_estimate_processing_category_sets():
    assert estimate_processing(categories=["lodging", "airfare", "lodging"]) == {
        "similar_claims": 2, "median_days": 7.5,
    }
    assert estimate_processing(categories=[]) == {
        "similar_claims": 0, "median_days": None,
    }


def test_tools_are_traced():
    with run_trace() as run:
        check_documents(categories=["mileage"], international=False,
                        has_guests=False, attachments=["mileage_log"])
        check_policy_flags(mentions_alcohol=False, total=100, attachments=[])
        estimate_processing(categories=["mileage"])
    assert run.tools_called() == [
        "check_documents", "check_policy_flags", "estimate_processing",
    ]
    assert all(step["ok"] for step in run.steps)
