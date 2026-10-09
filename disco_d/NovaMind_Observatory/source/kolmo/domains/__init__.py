"""Task domains other than ARC.

Everything here must produce the SAME (GridPair, Task) types grids.py
defines, because the library lives in the weight space of one fixed
architecture: two domains can only share a basis B if their fits live in
the same theta. That is a real constraint on what may be called a
"domain" here, and it is disclosed rather than worked around -- a domain
that only fits by distorting its data into a grid would be measuring the
distortion, not transfer.
"""
