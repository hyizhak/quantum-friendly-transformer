"""Access the recorded experiment evidence distributed with the library.

Compact records and tables ship in the wheel. Raw measurements and checkpoints
can be read from a companion archive or an existing experiment directory.
"""

from .store import Artifact, EvidenceStore

__all__ = ["Artifact", "EvidenceStore"]
