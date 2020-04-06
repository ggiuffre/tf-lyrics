from tflyrics.text_provider import TextProvider
import pytest



class MockProvider(TextProvider):
    """A mock TextProvider."""

    def __init__(self, mock_arg: object):
        """Create a MockProvider object."""

        super(MockProvider, self).__init__()
        self.mock_arg = mock_arg

    def resources(self, limit: int = 10) -> int:
        for i in range(limit):
            yield i

    def get_text(self, index: int) -> str:
        return str(index) * 200




def test_abstractness():
    """A TextProvider object cannot be instantiated."""

    with pytest.raises(TypeError):
        abstract_prov = TextProvider()

def test_concreteness():
    """A concrete sub-class of TextProvider can be instantiated."""

    mock_arg = 'a'
    concrete_prov = MockProvider(mock_arg)
