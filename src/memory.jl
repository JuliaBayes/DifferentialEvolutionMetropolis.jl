#no memory
struct DEMMemoryless{T} <: AbstractDEMMemory{T}
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

#abstract memory-full type
abstract type AbstractDEMMemoryFormat{
    T, VV <: AbstractVector{<:AbstractVector{T}},
} <:
AbstractDEMMemory{T} end

#abstract type for methods of filling memory
abstract type AbstractDEMMemoryFillMethod end

# Per-chain block of the newest memory entries that each chain may not pick
struct DEMMemoryStagger
    enabled::Bool
    first_excluded::Vector{Int}
    n_excluded::Vector{Int}
end

DEMMemoryStagger(enabled::Bool, n_chains::Int) = DEMMemoryStagger(enabled, zeros(Int, n_chains), zeros(Int, n_chains))

# Hides the entries of chains `i + 1:n` from chain `i`, given the block stores chain `j` at `block_start + n - j`
function stagger_block!(stagger::DEMMemoryStagger, block_start::Int)
    stagger.enabled || return stagger
    n_chains = length(stagger.n_excluded)
    for i in 1:n_chains
        stagger.first_excluded[i] = block_start
        stagger.n_excluded[i] = n_chains - i
    end
    return stagger
end

# Hides the retained initial chain positions of chains `i + 1:n` from chain `i`, given chain `j > n_dropped` is stored at `j - n_dropped`
function stagger_initial_positions!(stagger::DEMMemoryStagger, n_dropped::Int)
    stagger.enabled || return stagger
    n_chains = length(stagger.n_excluded)
    for i in 1:n_chains
        first_chain = max(i, n_dropped) + 1
        stagger.first_excluded[i] = first_chain - n_dropped
        stagger.n_excluded[i] = max(n_chains - first_chain + 1, 0)
    end
    return stagger
end

clear_stagger!(stagger::DEMMemoryStagger) = (fill!(stagger.n_excluded, 0); stagger)

# Maps indices sampled from the archive with the excluded block removed back onto the archive
function skip_excluded!(indices::Vector{Int}, n::Int, stagger::DEMMemoryStagger, chain::Int)
    n_excluded = stagger.n_excluded[chain]
    n_excluded == 0 && return indices
    first_excluded = stagger.first_excluded[chain]
    for i in 1:n
        if indices[i] >= first_excluded
            indices[i] += n_excluded
        end
    end
    return indices
end

function write_block!(memory::AbstractDEMMemoryFormat, x)
    if update_position!(memory.fill)
        for i in 1:memory.fill.n_chains
            memory.mem_x[memory.fill.position - i + 1] .= x[i]
        end
        stagger_block!(memory.stagger, memory.fill.position - memory.fill.n_chains + 1)
    else
        clear_stagger!(memory.stagger)
    end
    return memory
end

#full memory, refreshing from the start
struct DEMMemoryRefill{T, VV <: AbstractVector{<:AbstractVector{T}}, F <: AbstractDEMMemoryFillMethod} <:
    AbstractDEMMemoryFormat{T, VV}
    mem_x::VV
    fill::F
    stagger::DEMMemoryStagger
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DEMMemoryRefill{T, VV},
        x::VV
    ) where {T <: Real, VV <: AbstractVector{<:AbstractVector{T}}}
    write_block!(memory, x)
    if memory.fill.position == length(memory.mem_x)
        memory.fill.position = 0
    end
    return memory
end

#non-full memory, filling up to a max size then extending or refilling
struct DEMMemoryFill{T, VV <: AbstractVector{<:AbstractVector{T}}, F <: AbstractDEMMemoryFillMethod} <:
    AbstractDEMMemoryFormat{T, VV}
    mem_x::VV
    fill::F
    refill::Bool
    memory_size::Int
    stagger::DEMMemoryStagger
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DEMMemoryFill{T, VV},
        x::VV
    ) where {T <: Real, V_elem <: AbstractVector{T}, VV <: AbstractVector{V_elem}}
    write_block!(memory, x)

    if memory.fill.position == length(memory.mem_x)
        if memory.refill
            memory = DEMMemoryRefill(memory.mem_x, memory.fill, memory.stagger, memory.indices_INTERNAL, memory.ordered_indices_INTERNAL)
            memory.fill.position = 0
        else
            #increase memory size
            resize!(memory.mem_x, length(memory.mem_x) + memory.memory_size)
            dims = size(memory.mem_x[1])
            for i in (length(memory.mem_x) - memory.memory_size + 1):length(memory.mem_x)
                memory.mem_x[i] = V_elem(undef, dims)
            end
        end
    end
    return memory
end

mutable struct DEMMemoryFillEvery <:
    AbstractDEMMemoryFillMethod
    position::Int
    n_chains::Int
end

function update_position!(method::DEMMemoryFillEvery)
    method.position += method.n_chains
    return true
end

mutable struct DEMMemoryFillThin <:
    AbstractDEMMemoryFillMethod
    position::Int
    n_chains::Int
    count::Int
    max_count::Int
end

function update_position!(method::DEMMemoryFillThin)
    method.count -= 1
    if method.count == 0
        method.position += method.n_chains
        method.count = method.max_count
        return true
    else
        return false
    end
end
