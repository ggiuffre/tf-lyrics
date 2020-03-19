from .context import Poet, default_vocab
import itertools



def test_poet_creation():
    """A Poet object can be instantiated, and all of its arguments are optional."""

    # create one test value for each argument to the Poet constructor:
    test_args = {
        'name': 'test_name',
        'vocabulary': ['a', 'b', 'c'],
        'embdedding_dim': 23,
        'rnn_units': 17
        }

    # verify that arguments to the Poet constructor are all optional:
    args_as_tuple = zip(test_args.keys(), test_args.values())
    for l in range(len(test_args)):
        for args in itertools.combinations(args_as_tuple, l):
            p = Poet(**dict(args_as_tuple))
            assert type(p) is Poet

def test_poet_vocabulary():
    """A Poet object can be assigned a non-empty vocabulary, and will otherwise have a default vocabulary of recurring characters."""

    # verify that the Poet has a default vocabulary:
    p1 = Poet()
    assert p1.vocabulary == default_vocab
    assert p1.model.layers[0].input_dim == len(default_vocab)

    # verify that an empty vocabulary is ignored by the Poet:
    p2 = Poet(vocabulary=[])
    assert p2.vocabulary == default_vocab
    assert p2.model.layers[0].input_dim == len(default_vocab)

    # verify that it's possible to assign a custom vocabulary:
    custom_vocab = ['a', 'b', 'c']
    p3 = Poet(vocabulary=custom_vocab)
    assert p3.vocabulary == custom_vocab
    assert p3.model.layers[0].input_dim == len(custom_vocab)

def test_poet_batch_size():
    """Changing the batch size of a Poet changes the model input size."""

    p = Poet()

    sizes = [1, 2, 3, 4, 8, 16, 17, 32, 33, 64, 128]
    for s in sizes:
        p.batch_size = s
        assert p.model.input_shape == (s, None)
