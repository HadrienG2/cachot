# A cache simulator, used for studying pairwise iterations

There are well-known techniques for performing cache-efficient 2D iteration
over square domains, including blocking and space-filling curves.

But how applicable are those techniques to iteration over unordered pairs of
elements in a 1D set, which is more or less equivalent to 2D iteration over the
lower/upper triangular part of a square lattice? And are there more efficient
iteration schemes in this configuration?

This program was built to answer this sort of question.
