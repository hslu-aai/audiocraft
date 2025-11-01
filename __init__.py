# Audiocraft package wrapper
# This makes the nested audiocraft.audiocraft structure work properly

from audiocraft.audiocraft import *
from audiocraft.audiocraft import __version__

# Re-export all submodules
from audiocraft import audiocraft as _inner
modules = _inner.modules
models = _inner.models
data = _inner.data
