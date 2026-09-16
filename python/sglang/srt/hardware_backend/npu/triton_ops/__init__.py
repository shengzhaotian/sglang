"""In-tree Triton-Ascend kernels for the Kimi-K3 NPU decode path.

Importing this package pulls in ``triton``; import the modules lazily from the
call sites that are gated on the corresponding switch, so the CPU unit tests
and non-Ascend builds never need it.
"""
