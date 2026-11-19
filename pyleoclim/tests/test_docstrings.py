"""Static checks on the package source and its docstrings.

These parse the source with ``ast`` instead of importing modules, so they are
fast and do not depend on optional dependencies.
"""
import ast
import pathlib
import re
import warnings

import pytest

PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
SOURCE_FILES = sorted(PKG_DIR.rglob("*.py"))
SOURCE_IDS = [p.relative_to(PKG_DIR).as_posix() for p in SOURCE_FILES]

# Control characters that a backslash sequence in a non-raw string silently
# produces, e.g. '\frac' -> form feed + 'rac', '\tau' -> tab + 'au',
# '\beta' -> backspace + 'eta'.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f]")


def _docstrings(tree, src):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                const = body[0].value
                yield getattr(node, "name", "<module>"), const, ast.get_source_segment(src, const) or ""


@pytest.mark.parametrize("path", SOURCE_FILES, ids=SOURCE_IDS)
def test_no_escape_corruption_in_docstrings(path):
    """Docstrings with LaTeX must be raw strings (issue #716)."""
    src = path.read_text(encoding="utf-8")
    problems = []
    for name, const, segment in _docstrings(ast.parse(src), src):
        text = const.value
        # a tab is only a problem if it did not come from a literal tab in the source
        if "\t" not in segment:
            text = text.replace("\t", "\x00")
        for m in CONTROL_CHARS.finditer(text):
            context = const.value[max(0, m.start() - 20):m.end() + 10]
            problems.append(f"line {const.lineno} ({name}): {context!r}")
    assert not problems, (
        "Docstrings contain control characters from backslash escapes; "
        "use a raw docstring (r'''...'''):\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=SOURCE_IDS)
def test_no_invalid_escape_sequences(path):
    """Invalid escapes such as '\\%' warn today and will be errors in future Python."""
    src = path.read_text(encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(src, str(path), "exec")
    msgs = [f"line {w.lineno}: {w.message}" for w in caught
            if issubclass(w.category, (SyntaxWarning, DeprecationWarning)) and "escape" in str(w.message)]
    assert not msgs, "\n".join(msgs)
