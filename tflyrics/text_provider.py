from abc import ABC, abstractmethod



class TextProvider(ABC):
    """An abstract proxy to a collection of text resources.

        TextProvider is the base class for any concrete class that exposes a
        `resources` generator and a `get_text` method. A Genius object, for
        example, is also a TextProvider: it provides a `resources` generator
        that yields song identifiers, and a `get_text` method that maps a
        song  identifier to the lyrics of that song.

        In general, the `resources` generator of a TextProvider is a generator
        that yields resource identifiers; resources can be songs on the Genius
        database, songs in a text file, text contained in other documents,
        etcetera. The `get_text` method, on the other side, is what allows to
        get the actual of a resource from an identifier. This way, a client
        needs only traverse a generator of IDs and can delegate the I/O burden
        of retrieving the actual content of each resource to a some other part
        od the program, that will use the `get_text` method.
        """

    def __init__(self):
        """Create a TextProvider object."""

        super(ABC, self).__init__()

    @abstractmethod
    def resources(self, **params)-> int:
        """Generate the IDs of a specified set of resources.

        One at a time, yield the unique identifiers of some resources that
        meet a set of criteria specified by the user.

        :param params: criteria that resources should meet
        :yield: an integer that uniquely identifies the resource
        """

        pass

    @abstractmethod
    def get_text(self, index: int) -> str:
        """Get the text contained in a resource identified by a string.

        Get the text contained in a resource, by providing its identifier.

        :param song_id: the identifier of a resource
        :return: the text contained in that resource
        """

        pass
