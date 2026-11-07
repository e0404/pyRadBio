"""Biophysical models. LEM I-IV implemented here; the MKM family runs on a
pluggable backend (the native, pyMKM-free engine by default; pyMKM optional, for
cross-checks).

Import models by name to keep optional deps lazy, e.g.::

    from pyradbio.models.lem.lem1 import LEM1
    from pyradbio.models.mkm import MKMModel   # no optional extra needed
"""
