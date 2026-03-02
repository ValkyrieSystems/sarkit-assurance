import string
import unicodedata

ALLOWED_CHARS = string.ascii_letters + string.digits + "-_ "
ESC_CHAR = "+"


def sanitize_char(c):
    try:
        return c if c in ALLOWED_CHARS else f"{ESC_CHAR}{unicodedata.name(c)}{ESC_CHAR}"
    except ValueError:
        return f"{ESC_CHAR}u{ord(c):04x}{ESC_CHAR}"


def sanitize_name(name):
    assert ESC_CHAR not in ALLOWED_CHARS
    return "".join([sanitize_char(c) for c in name])
