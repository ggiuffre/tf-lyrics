from tflyrics import LyricsGenerator, default_vocab
from tflyrics.genius import Genius
import itertools
import pytest
import tensorflow as tf



@pytest.fixture(scope='module')
def artists():
    return ['Bob Dylan', 'Shabazz Palaces']

@pytest.fixture(scope='module')
def per_artist():
    return 3

@pytest.fixture(scope='module')
def example_correct_gen(artists, per_artist):
    return LyricsGenerator(artists=artists, per_artist=per_artist)



def test_creation():
    """A LyricsGenerator object can be instantiated, and all of its arguments
    are optional."""

    # create mock values for each argument to the LyricsGenerator constructor:
    test_args = {
        'artists': ['Testament', 'Testonius Monk', 'Test Buckley'],
        'per_artist': 3,
        'vocabulary': ['a', 'b', 'c'],
        'text_provider': Genius()
        }

    # verify that arguments to the constructor are all optional:
    args_as_tuple = zip(test_args.keys(), test_args.values())
    for l in range(len(test_args)):
        for args in itertools.combinations(args_as_tuple, l):
            p = LyricsGenerator(**dict(args_as_tuple))
            assert isinstance(p, LyricsGenerator)

def test_vocabulary():
    """A LyricsGenerator can be assigned a non-empty vocabulary, and will otherwise have a default vocabulary of recurring characters."""

    # verify that the LyricsGenerator has a default vocabulary:
    gen1 = LyricsGenerator()
    assert gen1.vocabulary == default_vocab

    # verify that an empty vocabulary is ignored by the LyricsGenerator:
    gen2 = LyricsGenerator(vocabulary=[])
    assert gen2.vocabulary == default_vocab

    # verify that it's possible to assign a custom vocabulary:
    custom_vocab = ['a', 'b', 'c']
    gen3 = LyricsGenerator(vocabulary=custom_vocab)
    assert gen3.vocabulary == custom_vocab

def test_num_songs(example_correct_gen, artists, per_artist):
    """A LyricsGenerator has a list of songs whose length is at most the
    product of the number of artists and the number of songs per artist."""

    gen = example_correct_gen
    assert len(gen.songs) <= len(artists) * per_artist

def test_fake_artist():
    """A LyricsGenerator has an empty list of songs, if asked to collect
    songs by a non-existing artist."""

    artists = ['tRqZDkVuppLYZcSplOOn']
    per_artist = 5
    gen = LyricsGenerator(artists=artists, per_artist=per_artist)
    assert len(gen.songs) == 0

def test_as_dataset(example_correct_gen, artists, per_artist):
    """A LyricsGenerator can provide a TensorFlow Dataset object with
    a predictable shape."""

    gen = example_correct_gen

    batch_size = 2
    seq_length = 50
    ds1 = gen.as_dataset(batch_size=batch_size, seq_length=seq_length)

    assert isinstance(ds1, tf.data.Dataset)
    assert len(ds1.element_spec) == 2
    assert ds1.element_spec[0].shape == (batch_size, seq_length)
    assert ds1.element_spec[1].shape == (batch_size, seq_length)

@pytest.mark.parametrize('seq_length', [51, 100, 123])
def test_preprocess(seq_length):
    """A LyricsGenerator can turn a string into multiple sequences of integers
    whose character representations are subsequences of the original string."""

    gen = LyricsGenerator()
    string = 'abcdefghijklmnopqrstuvwxyz' * 234
    print(type(seq_length))
    subsequences = gen.preprocess(string, seq_length=seq_length)

    assert isinstance(subsequences, tf.Tensor)

    expected_shape = [len(string) // (seq_length + 1), (seq_length + 1)]
    assert subsequences.shape.as_list() == expected_shape

    for seq in subsequences:
        assert ''.join([gen.vocabulary[i] for i in seq]) in string
