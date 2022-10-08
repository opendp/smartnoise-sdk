from snsynth import *
from snsynth.base import synth_map

class TestFactory:
    def test_create_empty(self):
        for synth in synth_map.keys():
            _ = Synthesizer.create(synth, epsilon=1.0)
