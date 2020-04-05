from tflyrics import Poet, LyricsGenerator, default_vocab
import itertools
import pytest
import shutil
import tensorflow as tf



@pytest.fixture(scope='module')
def example_correct_gen():
    artists = ['Bob Dylan', 'Shabazz Palaces']
    per_artist = 3
    return LyricsGenerator(artists=artists, per_artist=per_artist)



def test_creation():
    """A Poet object can be instantiated, and all of its arguments are
    optional."""

    # create mock values for each argument to the Poet constructor:
    test_args = {
        'name': 'test_name',
        'vocabulary': ['a', 'b', 'c'],
        'embdedding_dim': 23,
        'rnn_units': 17,
        'batch_size': 4
        }

    # verify that arguments to the constructor are all optional:
    args_as_tuple = zip(test_args.keys(), test_args.values())
    for l in range(len(test_args)):
        for args in itertools.combinations(args_as_tuple, l):
            p = Poet(**dict(args_as_tuple))
            assert isinstance(p, Poet)

def test_vocabulary():
    """A Poet object can be assigned a non-empty vocabulary, and will otherwise have a default vocabulary of recurring characters."""

    # verify that the Poet has a default vocabulary:
    p1 = Poet()
    assert p1.vocabulary == default_vocab

    # verify that an empty vocabulary is ignored by the Poet:
    p2 = Poet(vocabulary=[])
    assert p2.vocabulary == default_vocab

    # verify that it's possible to assign a custom vocabulary:
    custom_vocab = ['a', 'b', 'c']
    p3 = Poet(vocabulary=custom_vocab)
    assert p3.vocabulary == custom_vocab

    # verify that the input size of the model reflects the vocabulary size:
    assert p1.model.layers[0].input_dim == len(default_vocab)
    assert p2.model.layers[0].input_dim == len(default_vocab)
    assert p3.model.layers[0].input_dim == len(custom_vocab)

def test_batch_size():
    """Changing the batch size of a Poet changes the model input size."""

    p = Poet()

    sizes = [1, 2, 3, 4, 8, 16, 17, 32, 33, 64, 128]
    for s in sizes:
        p.batch_size = s
        assert p.model.input_shape == (s, None)

def test_train(example_correct_gen):
    """It is possible to train a Poet on a TensorFlow Dataset object."""

    # create a mock dataset:
    ds = tf.data.Dataset.from_tensors((tf.range(20), tf.range(20) + 1))
    ds = ds.repeat(33).batch(4, drop_remainder=True)

    # train a Poet on the dataset:
    p1 = Poet(rnn_units=256)
    hist_1 = p1.train_on(ds)
    assert isinstance(hist_1, dict)
    assert 'loss' in hist_1
    assert isinstance(hist_1['loss'], list)
    text_1 = p1.generate('Abc', n_gen_chars=26)

    # train a Poet on the dataset for multiple epochs:
    p2 = Poet(rnn_units=256)
    n_epochs = 2
    hist_2 = p2.train_on(ds, n_epochs=n_epochs)
    assert isinstance(hist_2, dict)
    assert 'loss' in hist_2
    assert isinstance(hist_2['loss'], list)
    assert len(hist_2['loss']) == n_epochs

    # train a Poet on a LyricsGenerator:
    p1 = Poet(rnn_units=256, batch_size=16)
    hist_1 = p1.train_on(example_correct_gen)
    assert isinstance(hist_1, dict)
    assert 'loss' in hist_1
    assert isinstance(hist_1['loss'], list)
    text_1 = p1.generate('Abc', n_gen_chars=26)

def test_restore():
    """It is possible to save a checkpoint at each epoch when training a Poet.
    In this case, the latest checkpoint can be retrieved."""

    # create a mock dataset:
    ds = tf.data.Dataset.from_tensors((tf.range(20), tf.range(20) + 1))
    ds = ds.repeat(33).batch(4, drop_remainder=True)

    # train a Poet on the dataset, saving checkpoints:
    p = Poet()
    p.train_on(ds, checkpoints=True)
    w1 = p.weights
    p.restore()
    w2 = p.weights
    assert w1 == w2
    shutil.rmtree(p.checkpoint_dir)

    # train a Poet on the dataset, without saving checkpoints:
    p = Poet()
    p.train_on(ds, checkpoints=False)
    with pytest.warns(ResourceWarning):
        p.restore()

def test_generate():
    """A Poet can generate text."""

    p = Poet()
    start_string = 'Abc'
    n_chars = 26
    text = p.generate(start_string, n_gen_chars=n_chars)
    assert len(text) == len(start_string) + n_chars
    assert start_string in text
