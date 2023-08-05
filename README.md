# RefCountedArrays.jl - Free your arrays!

RefCountedArrays.jl provides a reference-counted array, the `RCArray`, which
uses `malloc`-backed memory and supports being freed eagerly with
`RefCountedArrays.unsafe_free!`, just like CUDA.jl and other GPU array types.

### Why use refcounting?

Unlike Julia's venerable `Array`, which is GC-managed and thus lazily
collected, the `RCArray` can be freed the moment that the user or library that
allocated it knows that it's no longer used. When used correctly, this feature
can allow for very tight control of memory, which is especially valuable for
library authors who know the "lifetime" of the arrays that their library
allocates.
