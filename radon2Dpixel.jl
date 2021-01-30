using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using MAT

struct ray2d
    origin::Union{SVector{2,Float64},Vector{Float64}}
    dir::Union{SVector{2,Float64},Vector{Float64}}
    inv_direction::Union{SVector{2,Float64},Vector{Float64}}
    sign::Union{Array{Int64,1},SVector{2,Int64}}
end


function ray2d(p1::T, p2::T) where {T}
    dir::T = p2 - p1
    invdir::T = 1.0 ./ dir
    sign = (invdir .< 0) .* 1
    return ray2d(p1, dir, invdir, sign)
end

function ray2ddir(p1::T, dir::T) where {T}
    invdir::T = 1.0 ./ dir
    sign = (invdir .< 0) .* 1
    return ray2d(p1, dir, invdir, sign)
end


@inline @inbounds function intersects(
    r::ray2d,
    b::HyperRectangle{2,Float64},
)::Float64
    #parameters = @SVector[b.origin, b.origin + b.widths]
    #tmin = (parameters[r.sign[1]+1][1] - r.origin[1]) * r.inv_direction[1]
    #tmax = (parameters[1-r.sign[1]+1][1] - r.origin[1]) * r.inv_direction[1]
    #tymin = (parameters[r.sign[2]+1][2] - r.origin[2]) * r.inv_direction[2]
    #tymax = (parameters[1-r.sign[2]+1][2] - r.origin[2]) * r.inv_direction[2]
    eps = 1e-16

    if (r.inv_direction[1] >= 0.0)
        #tmin = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmin = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmax = (prevfloat(b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
    else
        #tmin = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmin = (prevfloat(b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmax = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
    end

    if (r.inv_direction[2] >= 0.0)
        #tymin = (parameters[1][2] - r.origin[2]) * r.inv_direction[2]
        tymin = (b.origin[2]  - r.origin[2]) * r.inv_direction[2]      
        #tymax = (parameters[2][2]- r.origin[2]) * r.inv_direction[2]
        tymax = (prevfloat(b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]
    else
        #tymin = (parameters[2][2] - r.origin[2]) * r.inv_direction[2]
        tymin = (prevfloat(b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]      
        #tymax = (parameters[1][2] - r.origin[2]) * r.inv_direction[2]
        tymax = (b.origin[2]  - r.origin[2]) * r.inv_direction[2]
    end

    if ((tmin > tymax) || (tymin > tmax))
        return -1.0
    end
     if (tymin > tmin)
         tmin = tymin
     end
    if (tymax < tmax)
         tmax = tymax
    end

    #println(tmin,",",tmax)
    return norm(r.dir*(tmax-tmin))

end

@inline function intersects(
    a::HyperRectangle{N,Float64},
    b::HyperRectangle{N,Float64},
) where {N}
    maxA = a.origin + a.widths
    minA = a.origin
    maxB = b.origin + b.widths
    minB = b.origin
    return (all(maxA .>= minB) && all(maxB .> minA))

end


function inittree2d(n)
    empty::Array{Int64,1} = []
    oct = Cell(SVector(-1.0, -1), SVector(2.0, 2), empty)
    for _ = 1:n
        for leaf in allleaves(oct)
            split!(leaf)
        end
    end
    for node in allcells(oct)
        node.data = copy(node.data)
    end
    return oct
end


function inserttotree!(
    box::HyperRectangle{N,Float64},
    tree::Cell,
    index::Int64,
) where {N}
    if (isleaf(tree))
        if (intersects(box, tree.boundary))
            push!(tree.data, index)
        end
    elseif (intersects(box, tree.boundary))
        for i = 1:length(tree.children)
            inserttotree!(box, tree[i], index)
        end
    end
    return nothing
end

function possiblevoxels(r::ray2d, tree::Cell)::Vector{Int64}
    if (isleaf(tree) && intersects(r, tree.boundary) > 0.0)
        return tree.data

    elseif (~(tree.parent == nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblevoxels(r, tree[i]))
        end
        return v

    elseif ((tree.parent == nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblevoxels(r, tree[i]))
        end
        return collect(Set(v))

    else
        return []
    end

end


function pixelray(
    r::ray2d,
    vv::Vector{HyperRectangle{2,Float64}},
    checklist::Vector{Int64},
)
    N = length(checklist)
    indices = Vector{Int64}()
    tlens = Vector{Float64}()
    for i = 1:N
        q = intersects(r, vv[checklist[i]])
        if (q>0.0)
            push!(tlens, q)
            push!(indices, checklist[i])
        end
    end
    return (indices, tlens)
end


function constructmatrix(tree::Cell, vv::Vector{HyperRectangle{2,Float64}}, Nproj::Int64, theta)
    r = sqrt(2) # Radius of the transform.
    Na = length(theta)
    if (~isa(theta, Union{SArray,Array}))
        theta = [theta]
    end
    if (Nproj > 1)
        span = range(r, stop = -r, length = Nproj)
    else
        span = [0.0]
    end
    Ntotal = Na * Nproj
    rows = Vector{Int64}()
    cols = Vector{Int64}()
    vals = Vector{Float64}()
    Nofpixels = length(vv)

    Nth = Threads.nthreads()
    rows = Vector{Vector{Int64}}(undef, Nth)
    cols = Vector{Vector{Int64}}(undef, Nth)
    vals = Vector{Vector{Float64}}(undef, Nth)
    for p = 1:Nth
        rows[p] = []
        cols[p] = []
        vals[p] = []
    end

    t = time()
    for a = 1:Na
        dir = [cos(theta[a]), sin(theta[a])]
        aux = [-sin(theta[a]), cos(theta[a])]
        Threads.@threads for i = 1:Nproj
            rayindex = (a - 1) * Nproj + i
            p1 = dir + aux * span[i]
            p2 = p1 - 2 * dir * r
            or = ray2d(p1, p2)
            checklist = possiblevoxels(or, tree) #Possible elements.
            indices, tlens = pixelray(or, vv, checklist)
            Nel = length(indices)

            append!(rows[Threads.threadid()],rayindex * ones(Int64, length(tlens)))
            append!(cols[Threads.threadid()], indices)
            append!(vals[Threads.threadid()], tlens)
  
        end
    end
    rows = vcat(rows...)
    cols = vcat(cols...)
    vals = vcat(vals...)
    M = sparse(rows, cols, vals, Ntotal, Nofpixels)
    println("Matrix constructed in ", time() - t, " seconds.")
    return M
end


function sub2ind(sizes, i::Int64, j::Int64, k::Int64)
    @assert i > 0 && i <= sizes[1]
    @assert j > 0 && j <= sizes[2]
    @assert k > 0 && k <= sizes[3]
    return (k - 1) * sizes[1] * sizes[2] + sizes[1] * (j - 1) + i
end


function setuppixels(sizex::Int64,sizey::Int64,octsize::Int64)
    oct = inittree2d(octsize)
    l = Array([ [0.0, 0] [1, 0] [1, 1] [0, 1.0]]')
    pm = mean(l, dims = 1)
    size = [sizex, sizey]
    cellsize = 2.0 ./ size
    width = @SVector[cellsize[1], cellsize[2]]
    for i = 1:2
        l[:, i] = cellsize[i] .* (l[:, i] .- pm[i]) .- cellsize[i] / 2.0 * size[i] .+ 0.5 * cellsize[i]
    end
    voxelvector = Vector{HyperRectangle{2,Float64}}(undef, sizex * sizey)
    t = time()

    for j = 1:sizey
        for i = 1:sizex
            ll = l .+ reshape(cellsize .* [i - 1, j - 1], 1, 2)
            origin = @SVector[
                minimum(ll[:, 1]),
                minimum(ll[:, 2]),
            ]
            voxel = HyperRectangle(origin, width)
            index = j + (i - 1) * sizey 
            inserttotree!(voxel, oct, index)
            voxelvector[index] = voxel
        end
    end

    println("Voxels initialized in ", time() - t, " seconds.")
    return (oct, voxelvector)
end



function test()
    fn = "phanpix.mat"
    Nx = 128; Ny = 128; Os = 4; 
    Npr = 185; Na = 64
    (qt,pixelvector)=setuppixels(Nx,Ny,Os)
    M = constructmatrix(
        qt,
        pixelvector,
        Npr,
        Array(range(0/ 2, stop = pi, length = Na)),
    )
    file = matopen(fn)
    zn = read(file, "m")
    zn = zn[:]
    y = M * zn
    rad = reshape(y, Npr, Na)
    imshow(rad, aspect = "auto")
    return nothing
end

test()
