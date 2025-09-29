import re
from typing import List


def camelize(string: str, uppercase_first_letter: bool = True) -> str:
    if uppercase_first_letter:
        return re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), string)
    else:
        return string[0].lower() + camelize(string)[1:]


def underscore(word: str) -> str:
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    return word.lower()


def strip_comma(target: str) -> List[str]:
    if target is None:
        return []

    return [x.strip() for x in target.split(',')]


def convert_any_item(target: str) -> str:
    return None if not target else target


def search_invalid_whitespace(value: str):
    return re.search(r"(^\s+\"|^\u3000+\")", value)
