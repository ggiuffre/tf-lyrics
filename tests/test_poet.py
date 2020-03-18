from .context import tflyrics

def test_poet_creation():
    """It is possible to instantiate a Poet object."""

    p1 = tflyrics.Poet()
    assert type(p1) is tflyrics.Poet

    p2 = tflyrics.Poet(vocabulary=['a', 'b', 'c'])
    assert type(p2) is tflyrics.Poet
