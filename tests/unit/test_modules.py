import pytest

from tests.helpers.runif import RunIf

from torchtext.datasets import WikiText2



def test_something1():
    """Some test description."""
    assert True is True


def test_something2():
    """Some test description."""
    assert 1 + 1 == 2


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
def test_something3(arg1: float):
    """Some test description."""
    assert arg1 > 0




if __name__ == "__main__":
    print(WikiText2)
    