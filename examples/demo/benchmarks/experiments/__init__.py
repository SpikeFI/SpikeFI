"""Concrete Experiment subclasses. Each module is imported here purely for
its side effect: importing it runs its @register decorator, which adds the
class to experiment.EXPERIMENTS so cli.py picks it up as a subcommand.
"""

from . import equivalence  # noqa: F401
