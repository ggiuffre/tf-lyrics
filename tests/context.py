import sys
import os
pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, pkg_path)

from tflyrics import Poet, LyricsGenerator, default_vocab
