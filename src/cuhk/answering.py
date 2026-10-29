"""Safe question inputs and shared answer validation for the investigator."""

import itertools


def question_input(question):
    return {
        "id": "q1",
        "category": question["category"],
        "question": question["question"],
        "options": {k: v for k, v in question["options"].items() if v.strip()},
    }


def answer_choices(question):
    letters = sorted(question["options"])
    if not letters or not set(letters) <= set("ABCD"):
        raise ValueError("Invalid question options")
    if question["category"] == "sequence":
        return ["".join(p) for p in itertools.permutations(letters)]
    if question["category"] == "multi":
        return [
            "".join(p)
            for n in range(1, len(letters) + 1)
            for p in itertools.combinations(letters, n)
        ]
    if question["category"] not in {"single", "combination", "emotion", "object_interaction"}:
        raise ValueError("Unknown question category")
    return letters


def validate_answer(value, question):
    if (
        not isinstance(value, dict)
        or set(value) != {"scene", "answer", "evidence"}
        or value["answer"] not in answer_choices(question)
        or any(not isinstance(value[k], str) or not value[k].strip() for k in ("scene", "evidence"))
    ):
        raise ValueError("Invalid structured answer")
    return value


def validate_joint_answer(value, questions):
    if (
        not isinstance(value, dict)
        or set(value) != {"scene", "answers"}
        or not isinstance(value["answers"], dict)
        or set(value["answers"]) != {q["id"] for q in questions}
    ):
        raise ValueError("Invalid joint answer coverage")
    for question in questions:
        entry = value["answers"][question["id"]]
        if not isinstance(entry, dict) or set(entry) != {"answer", "evidence"}:
            raise ValueError("Invalid joint answer entry")
        validate_answer({"scene": value["scene"], **entry}, question)
    return value
