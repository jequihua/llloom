"""Opt-in, read-only OKF / framework-profile observation.

Nothing in llloom calls this package. A caller opts in explicitly::

    from llloom.okf import observe_page_okf

It is deliberately absent from ``llloom.__all__``: adding it to the root
re-export set would change a frozen public inventory for a surface no default
path uses.
"""

from llloom.okf.observation import PageOkfObservation, observe_page_okf

__all__ = ["PageOkfObservation", "observe_page_okf"]
