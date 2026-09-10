from toxic_classifier.utils import clean_text


def test_clean_text():
    assert clean_text("HELLO <b>World</b>!") == "hello world !"
