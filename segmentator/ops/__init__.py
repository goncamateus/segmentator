"""Pure image operations, free of the stage protocol and of any config object.

Every function here takes explicit arguments and returns arrays or plain data.
The stage classes in :mod:`segmentator.stages` are thin wrappers that hold the
parameters and move the result onto the :class:`~segmentator.pipeline.Ctx`.

Ported from goncanalyser's ``features/*``, which drove the same operators from a
single frozen ``Settings`` bundle because a Qt panel needed one flat object. A
registry-built stage already owns its parameters, so ``Settings`` is not ported.
"""
