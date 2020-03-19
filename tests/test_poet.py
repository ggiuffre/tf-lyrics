from .context import Poet, default_vocab
import itertools
import tensorflow as tf



def test_creation():
    """A Poet object can be instantiated, and all of its arguments are
    optional."""

    # create mock values for each argument to the Poet constructor:
    test_args = {
        'name': 'test_name',
        'vocabulary': ['a', 'b', 'c'],
        'embdedding_dim': 23,
        'rnn_units': 17
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

def test_train():
    """It is possible to train a Poet on a TensorFlow Dataset object."""

    # create a mock dataset:
    ds = tf.data.Dataset.from_tensors((tf.range(20), tf.range(20) + 1))
    ds = ds.repeat(42).batch(4, drop_remainder=True)

    # train a Poet on the dataset:
    p = Poet(vocabulary=default_vocab)
    p.train_on(ds)
    text = p.generate('Abc', n_gen_chars=26)

    # train a Poet on the dataset for multiple epochs:
    p = Poet(vocabulary=default_vocab)
    p.train_on(ds, n_epochs=2)

def test_generate():
    """A Poet can generate text."""

    p = Poet()
    start_string = 'Abc'
    n_chars = 26
    text = p.generate(start_string, n_gen_chars=n_chars)
    assert len(text) == len(start_string) + n_chars
    assert start_string in text
