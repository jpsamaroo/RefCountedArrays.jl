module RefCountedArrays

using Mmap

export RCArray, RCVector, RCMatrix

abstract type AbstractArrayBuffer end

#= TODO: Justify this
mutable struct MmapBuffer <: AbstractArrayBuffer
    ptr::Ptr{Cvoid}
    size::Int
    @atomic ctr::Int
    arr::Union{Vector{UInt8},Nothing}
end
function MmapBuffer(sz::Integer)
    arr = Mmap.mmap(Mmap.Anonymous(), Vector{UInt8}, sz, 0)
    buf = MmapBuffer(pointer(arr), sz, 0, arr)
    return buf
end
function unsafe_free!(buf::MmapBuffer)
    finalize(buf.arr)
    buf.arr = nothing
end
=#

mutable struct MallocBuffer <: AbstractArrayBuffer
    ptr::Ptr{Cvoid}
    size::Int
    @atomic ctr::Int
end
function MallocBuffer(sz::Integer)
    buf = MallocBuffer(Libc.malloc(sz), sz, 1)
    return buf
end
unsafe_free!(buf::MallocBuffer) = Libc.free(buf.ptr)
Base.pointer(buf::MallocBuffer) = buf.ptr

function release!(buf::AbstractArrayBuffer)
    if (@atomic :monotonic buf.ctr -= 1) == 0
        unsafe_free!(buf)
    end
    return
end

mutable struct RCArray{T,N,B<:AbstractArrayBuffer} <: AbstractArray{T,N}
    buf::B
    dims::Dims{N}
    function RCArray{T,N}(buf::B, dims::Dims{N}) where {T,N,B}
        @assert isbitstype(T) "RCArray only supports bitstype elements"
        xs = new{T,N,B}(buf, dims)
        finalizer(release!, xs)
        return xs
    end
end
release!(A::RCArray) = release!(A.buf)
unsafe_free!(A::RCArray) = unsafe_free!(A.buf)

# aliases
const RCVector{T} = RCArray{T,1}
const RCMatrix{T} = RCArray{T,2}
const RCVecOrMat{T} = Union{RCVector{T},RCMatrix{T}}

function RCArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    buf = MallocBuffer(prod(dims) * sizeof(T))
    return RCArray{T,N}(buf, dims)
end

# type and dimensionality specified, accepting dims as series of Ints
RCArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = RCArray{T,N}(undef, dims)

# type but not dimensionality specified
RCArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = RCArray{T,N}(undef, dims)
RCArray{T}(::UndefInitializer, dims::Integer...) where {T} =
    RCArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# from Base arrays
function RCArray{T,N}(x::Array{T,N}, dims::Dims{N}) where {T,N}
    r = RCArray{T,N}(undef, dims)
    copyto!(r, x)
    return r
end

# empty vector constructor
RCArray{T,1}() where {T} = RCArray{T,1}(undef, 0)

# array interface
Base.similar(a::RCArray{T,N}) where {T,N} = RCArray{T,N}(undef, size(a))
Base.similar(::RCArray{T}, dims::Base.Dims{N}) where {T,N} = RCArray{T,N}(undef, dims)
Base.similar(::RCArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = RCArray{T,N}(undef, dims)

Base.elsize(::Type{<:RCArray{T}}) where {T} = sizeof(T)
Base.size(x::RCArray) = x.dims
Base.sizeof(x::RCArray) = Base.elsize(x) * length(x)

# interop with Julia arrays
RCArray{T,N}(x::AbstractArray{S,N}) where {T,N,S} =
    RCArray{T,N}(convert(Array{T}, x), size(x))

# underspecified constructors
RCArray(A::AbstractArray{T,N}) where {T,N} = RCArray{T,N}(A)
RCArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = RCArray{T,N}(xs)
(::Type{RCArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = RCArray{S,N}(x)

# idempotency
RCArray{T,N}(xs::RCArray{T,N}) where {T,N} = xs

# conversions
Base.convert(::Type{T}, x::T) where T <: RCArray = x

# pointer
Base.pointer(A::RCArray{T}) where T = 
    reinterpret(Ptr{T}, pointer(A.buf))

# copies
function Base.copyto!(
    dst::Array{T}, d_offset::Integer,
    src::RCArray{T}, s_offset::Integer,
    amount::Integer) where T
    amount == 0 && return dst
    @boundscheck checkbounds(dst, d_offset+amount-1)
    @boundscheck checkbounds(src, s_offset+amount-1)
    GC.@preserve dst src begin
        dst_ptr = pointer(dst)
        src_ptr = pointer(src)
        ccall(:memcpy, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
              dst_ptr, src_ptr, amount)
    end
    return dst
end
#Base.copyto!(
function Base.copyto!(
    dst::RCArray{T}, d_offset::Integer,
    src::Array{T}, s_offset::Integer,
    amount::Integer) where T
    amount == 0 && return dst
    @boundscheck checkbounds(dst, d_offset+amount-1)
    @boundscheck checkbounds(src, s_offset+amount-1)
    GC.@preserve dst src begin
        dst_ptr = pointer(dst)
        src_ptr = pointer(src)
        ccall(:memcpy, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
              dst_ptr, src_ptr, amount)
    end
    return dst
end
function Base.copyto!(
    dst::RCArray{T}, d_offset::Integer,
    src::RCArray{T}, s_offset::Integer,
    amount::Integer) where T
    amount == 0 && return dst
    @boundscheck checkbounds(dst, d_offset+amount-1)
    @boundscheck checkbounds(src, s_offset+amount-1)
    GC.@preserve dst src begin
        dst_ptr = pointer(dst)
        src_ptr = pointer(src)
        ccall(:memcpy, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
              dst_ptr, src_ptr, amount)
    end
    return dst
end

# indexing
function Base.getindex(A::RCArray{T}, idx::Integer) where T
    @boundscheck checkbounds(A, idx)
    unsafe_load(pointer(A), idx)
end
function Base.getindex(A::RCArray{T}, idxs::Integer...) where T
    @boundscheck checkbounds(A, idxs...)
    idx = Base._to_linear_index(A, idxs...)
    unsafe_load(pointer(A), idx)
end
function Base.setindex!(A::RCArray{T}, value, idx::Integer) where T
    @boundscheck checkbounds(A, idx)
    unsafe_store!(pointer(A), convert(T, value), idx)
end
function Base.setindex!(A::RCArray{T}, value, idxs::Integer...) where T
    @boundscheck checkbounds(A, idxs...)
    idx = Base._to_linear_index(A, idxs...)
    unsafe_store!(pointer(A), convert(T, value), idx)
end

end # module RefCountedArrays
