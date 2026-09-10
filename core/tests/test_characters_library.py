"""The character library: naming, resolution, listing, and import validation."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from inline_core.characters import charfile as cf
from inline_core.characters import library


@pytest.fixture(autouse=True)
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.delenv("INLINE_CHARACTERS_DIR", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)
    return root


def _doc(name: str = "Ada") -> cf.CharDoc:
    manifest = cf.Manifest(
        char_id="7f1c0d2e-0000-4000-8000-000000000001",
        name=name,
        created_at=1755000000,
        modified_at=1755000000,
        text={"path": "text/description.md", "sha256": cf.sha256_bytes(b"green jacket")},
    )
    manifest.refs.append({"path": "refs/000.png", "sha256": cf.sha256_bytes(b"x")})
    return cf.CharDoc(
        manifest=manifest,
        members={"refs/000.png": b"x", "text/description.md": b"green jacket"},
    )


def test_file_name_reduces_a_display_name_to_something_openable() -> None:
    assert library.file_name("Ada") == "Ada.char"
    assert library.file_name("Ada Løvelace") == "Ada Lvelace.char"
    assert library.file_name("../../etc/passwd") == "etcpasswd.char"
    assert library.file_name("天") == "character.char"


def test_saving_two_characters_with_one_name_does_not_overwrite() -> None:
    first = library.save(_doc("Ada"))
    second = library.save(_doc("Ada"))
    assert first.name == "Ada.char"
    assert second.name == "Ada-2.char"
    assert first.exists() and second.exists()


def test_resolve_accepts_a_bare_name_and_a_path() -> None:
    library.save(_doc("Ada"))
    assert library.resolve("Ada.char") is not None
    assert library.resolve("characters/Ada.char") is not None
    assert library.resolve("Nobody.char") is None
    assert library.resolve("") is None


def test_summaries_carry_the_manifest_and_the_description() -> None:
    library.save(_doc("Ada"))
    rows = library.summaries()
    assert len(rows) == 1
    assert rows[0]["name"] == "Ada"
    assert rows[0]["refs"] == 1
    assert rows[0]["description"] == "green jacket"


def test_an_unreadable_character_is_reported_rather_than_hidden() -> None:
    """A corrupt file the user can see is fixable; one we hide is a file that vanished."""
    library.root().mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(library.root() / "Broken.char", "w") as archive:
        archive.writestr("hello.txt", "hi")
    rows = library.summaries()
    assert len(rows) == 1
    assert rows[0]["error"]


def test_delete_removes_the_file_and_reports_a_miss() -> None:
    library.save(_doc("Ada"))
    assert library.delete("Ada.char") is True
    assert library.delete("Ada.char") is False


def test_import_bytes_validates_before_it_lands(tmp_path: Path) -> None:
    staged = tmp_path / "incoming.char"
    cf.write(staged, _doc("Imported"))
    landed = library.import_bytes(staged.read_bytes(), "whatever.char")
    assert landed.name == "Imported.char"

    with pytest.raises(cf.CharFileError):
        library.import_bytes(b"not a zip", "bad.char")
    assert [p.name for p in library.list_files()] == ["Imported.char"]


def test_content_hash_changes_when_the_file_changes() -> None:
    path = library.save(_doc("Ada"))
    before = library.content_hash(path)

    edited = _doc("Ada")
    edited.members["text/description.md"] = b"red jacket"
    cf.write(path, edited)
    assert library.content_hash(path) != before


def test_with_no_characters_dir_set_saving_is_unchanged(models_root: Path) -> None:
    assert library.save(_doc("Ada")) == models_root / "characters" / "Ada.char"


def test_characters_dir_is_where_a_character_is_written(
    models_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", str(private))
    saved = library.save(_doc("Ada"))
    assert saved == private / "Ada.char"
    # The point of the knob: a shared models root must not receive one user's character.
    assert not (models_root / "characters" / "Ada.char").exists()


def test_resolve_prefers_the_characters_dir_over_a_models_root(
    models_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = library.save(_doc("Ada"))
    private = tmp_path / "private"
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", str(private))
    private.mkdir()
    cf.write(private / "Ada.char", _doc("Ada"))
    assert library.resolve("Ada.char") == private / "Ada.char"
    assert shared.is_file()


def test_resolve_still_finds_a_character_only_in_a_models_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library.save(_doc("Ada"))
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", str(tmp_path / "private"))
    assert library.resolve("Ada.char") is not None


def test_list_files_covers_both_without_listing_a_name_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library.save(_doc("Ada"))
    library.save(_doc("Bo"))
    private = tmp_path / "private"
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", str(private))
    library.save(_doc("Cy"))
    cf.write(private / "Ada.char", _doc("Ada"))
    names = [p.name for p in library.list_files()]
    assert sorted(names) == ["Ada.char", "Bo.char", "Cy.char"]
    assert [p for p in library.list_files() if p.name == "Ada.char"][0].parent == private
