import pytest

@pytest.fixture(scope="session", autouse=True)
def invoke():
    print("starting")