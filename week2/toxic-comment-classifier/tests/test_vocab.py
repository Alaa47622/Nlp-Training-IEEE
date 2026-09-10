from toxic_classifier.features.vocab import Vocabulary


def test_vocab_encode():
    v = Vocabulary()
    v.build(["hello world hello"])
    encoded = v.encode("hello unknown", 5)
    assert len(encoded) == 5
    assert encoded[0] != 0
