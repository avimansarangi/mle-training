def test_package_import():
    """
    Verify that the ml_workflow package can be imported.
    """
    import ml_workflow
    assert ml_workflow is not None


def test_core_modules_import():
    """
    Verify that core workflow modules are importable.
    """
    from ml_workflow import ingest_data
    from ml_workflow import train
    from ml_workflow import score

    assert ingest_data is not None
    assert train is not None
    assert score is not None
