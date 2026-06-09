from importlib import import_module


def test_import_app_package() -> None:
    assert import_module("app") is not None
