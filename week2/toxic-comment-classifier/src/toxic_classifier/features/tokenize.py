import re


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[!?]", str(text).lower())
