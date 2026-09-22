from app.classifier import _normalize_result


def test_normalize_json_result():
    result = _normalize_result(
        '{"classification":"Toxic","explanation":"Contains abusive language."}'
    )
    assert result.classification == "Toxic"
    assert result.explanation == "Contains abusive language."


def test_normalize_markdown_json_result():
    result = _normalize_result(
        '```json\n{"classification":"Non-Toxic","explanation":"A harmless statement."}\n```'
    )
    assert result.classification == "Non-Toxic"


def test_normalize_plain_text_fallback():
    result = _normalize_result("Classification: Non-Toxic\nExplanation: harmless.")
    assert result.classification == "Non-Toxic"
