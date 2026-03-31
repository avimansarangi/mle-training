import importlib.metadata


def test_package_is_installed():
    """
    Verify that the ml-workflow package is installed in the environment.
    """
    version = importlib.metadata.version("ml-workflow")
    assert version is not None
