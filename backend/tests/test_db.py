from __future__ import annotations

from datetime import UTC

import numpy as np
import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy.exc import IntegrityError

from tribunal.cli import main
from tribunal.db.migrate import upgrade
from tribunal.db.models import Base, Document, Passage, Subject, User
from tribunal.db.session import make_engine, make_session_factory


def _factory(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    upgrade(url)
    return make_session_factory(make_engine(url))


def _seed(session):
    user = User(email="a@example.org")
    session.add(user)
    session.flush()
    s1, s2 = Subject(name="s1", owner_id=user.id), Subject(name="s2", owner_id=user.id)
    session.add_all([s1, s2])
    session.flush()
    return user, s1, s2


def _doc(user, subject, h="h"):
    return Document(
        subject_id=subject.id, uploaded_by=user.id, filename="f.pdf", content_hash=h,
        storage_key="k", size_bytes=1, ocr_languages=["eng"],
    )


def test_upgrade_matches_models(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    upgrade(url)
    with make_engine(url).connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []


def test_upgrade_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    upgrade(url)
    upgrade(url)


def test_document_hash_unique_per_subject(tmp_path):
    factory = _factory(tmp_path)
    with factory.begin() as session:
        user, s1, s2 = _seed(session)
        session.add_all([_doc(user, s1), _doc(user, s2)])
    with pytest.raises(IntegrityError), factory.begin() as session:
        user, s1 = session.query(User).one(), session.query(Subject).filter_by(name="s1").one()
        session.add(_doc(user, s1))


def test_embedding_roundtrip(tmp_path):
    factory = _factory(tmp_path)
    vec = np.arange(4, dtype=np.float32)
    with factory.begin() as session:
        user, s1, _ = _seed(session)
        doc = _doc(user, s1)
        session.add(doc)
        session.flush()
        session.add(Passage(document_id=doc.id, subject_id=s1.id, page_number=1, char_start=0,
                            char_end=4, text="abcd", embedding=vec, extraction_version="v1"))
    with factory.begin() as session:
        got = session.query(Passage).one().embedding
    assert np.array_equal(got, vec)


def test_utc_timestamps(tmp_path):
    factory = _factory(tmp_path)
    with factory.begin() as session:
        _seed(session)
    with factory.begin() as session:
        assert session.query(User).one().created_at.tzinfo is UTC


def test_defaults(tmp_path):
    factory = _factory(tmp_path)
    with factory.begin() as session:
        user, s1, _ = _seed(session)
        doc = _doc(user, s1)
        session.add(doc)
        session.flush()
        assert (len(doc.id), doc.status, user.is_admin) == (32, "queued", False)


def test_cli_db_upgrade(test_env, capsys):
    assert main(["db-upgrade"]) == 0
    assert "database at head" in capsys.readouterr().out
