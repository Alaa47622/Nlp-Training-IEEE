import re
import html
import unicodedata


ARABIC_DIACRITICS = re.compile(
    r"""
    ّ    | # Tashdid
    َ    | # Fatha
    ً    | # Tanwin Fath
    ُ    | # Damma
    ٌ    | # Tanwin Damm
    ِ    | # Kasra
    ٍ    | # Tanwin Kasr
    ْ    | # Sukun
    ـ      # Tatweel
    """,
    re.VERBOSE,
)


def _basic_cleanup(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_english(text: str) -> str:
    text = _basic_cleanup(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_arabic(text: str) -> str:
    text = _basic_cleanup(text)
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic diacritics and tatweel.
    text = re.sub(ARABIC_DIACRITICS, "", text)

    # Normalize common Arabic letter variants.
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)

    # Keep Arabic letters, digits and spaces.
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_language_detection(text: str) -> str:
    """
    Keep both Arabic and Latin letters. Character n-grams work very well
    for classical language identification.
    """
    text = _basic_cleanup(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\u0600-\u06FFa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
