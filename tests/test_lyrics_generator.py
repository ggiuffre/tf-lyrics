from .context import LyricsGenerator
import itertools
import tensorflow as tf



def test_creation():
    """A LyricsGenerator object can be instantiated, and all of its arguments
    are optional."""

    # create mock values for each argument to the LyricsGenerator constructor:
    test_args = {
        'artists': ['Testament', 'Testonius Monk', 'Test Buckley'],
        'per_artist': 3,
        'vocabulary': ['a', 'b', 'c'],
        'token': 'mock_access_token'
        }

    # verify that arguments to the constructor are all optional:
    args_as_tuple = zip(test_args.keys(), test_args.values())
    for l in range(len(test_args)):
        for args in itertools.combinations(args_as_tuple, l):
            p = LyricsGenerator(**dict(args_as_tuple))
            assert type(p) is LyricsGenerator
