from datetime import UTC
from pathlib import Path

import pytest

from tgmem.config import Settings, load_env_file
from tgmem.embeddings import HashingEmbedder, OpenAICompatibleEmbedder


def test_defaults_need_no_environment():
    settings = Settings.from_env({})
    assert settings.db_path == Path("memory.db")
    assert settings.vault_dir == Path("vault")
    assert settings.embedder == "auto"
    assert settings.build_embedder() is None  # no key: keyword + graph search only
    assert settings.tz() is UTC


def test_auto_embedder_uses_a_key_when_one_is_present():
    settings = Settings.from_env({"OPENAI_API_KEY": "placeholder-key",
                                  "TGMEM_EMBED_DIMENSIONS": "256"})
    embedder = settings.build_embedder()
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model_id == "api.openai.com/text-embedding-3-small@256"


def test_explicit_embedders():
    hashing = Settings.from_env({"TGMEM_EMBEDDER": "hash", "TGMEM_HASH_DIM": "64"})
    assert isinstance(hashing.build_embedder(), HashingEmbedder)
    assert hashing.build_embedder().dim == 64
    none = Settings.from_env({"TGMEM_EMBEDDER": "none", "OPENAI_API_KEY": "placeholder-key"})
    assert none.build_embedder() is None
    local = Settings.from_env({"TGMEM_EMBEDDER": "openai", "TGMEM_EMBED_API_KEY": "local",
                               "TGMEM_EMBED_BASE_URL": "http://127.0.0.1:8000/v1",
                               "TGMEM_EMBED_MODEL": "nomic-embed-text"})
    assert local.build_embedder().model_id == "127.0.0.1:8000/nomic-embed-text"


def test_openai_embedder_without_a_key_is_a_configuration_error():
    with pytest.raises(ValueError, match="API key"):
        Settings.from_env({"TGMEM_EMBEDDER": "openai"}).build_embedder()


@pytest.mark.parametrize(("env", "message"), [
    ({"TGMEM_EMBEDDER": "magic"}, "TGMEM_EMBEDDER"),
    ({"TGMEM_HASH_DIM": "lots"}, "TGMEM_HASH_DIM"),
    ({"TGMEM_EMBED_TIMEOUT": "-1"}, "TGMEM_EMBED_TIMEOUT"),
    ({"TGMEM_DEFAULT_COUNTRY_CODE": "+1"}, "TGMEM_DEFAULT_COUNTRY_CODE"),
    ({"TGMEM_TIMEZONE": "Mars/Olympus"}, "TGMEM_TIMEZONE"),
])
def test_invalid_settings_name_the_variable(env, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env(env)


def test_named_time_zones():
    settings = Settings.from_env({"TGMEM_TIMEZONE": "America/New_York"})
    assert settings.tz().key == "America/New_York"


def test_env_file_parsing(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nTGMEM_DB=data/memory.db\nexport TGMEM_EMBEDDER=hash\n"
        "TGMEM_VAULT=\"my vault\"\nEMPTY=\nWITH_EQUALS=a=b # not a comment\n",
        encoding="utf-8")
    assert load_env_file(path) == {"TGMEM_DB": "data/memory.db", "TGMEM_EMBEDDER": "hash",
                                   "TGMEM_VAULT": "my vault", "EMPTY": "",
                                   "WITH_EQUALS": "a=b # not a comment"}
    with pytest.raises(ValueError, match="line 2"):
        bad = tmp_path / "bad.env"
        bad.write_text("OK=1\nthis line has no equals sign\n", encoding="utf-8")
        load_env_file(bad)
