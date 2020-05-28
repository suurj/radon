using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using MAT


struct ray3d{T}
    origin::T
    dir::T
    inv_direction::T
    sign::Union{Array{Int64,1},SArray{Tuple{3},Int64,1,3}}
end


function ray3d(p1::T, p2::T) where {T}
    dir::T = p2 - p1
    invdir::T = 1.0 ./ dir
    sign = (invdir .< 0) .* 1
    return ray3d(p1, dir, invdir, sign)
end

#An Efficient and Robust Ray–Box Intersection Algorithm by
#Amy Williams Steve Barrus R.Keith Morley Peter Shirley.
@inline @inbounds function intersects(
    r::ray3d,
    b::HyperRectangle{3,Float64},
)::Float64
    Q = [b.origin, b.origin + b.widths]
    tmin = (Q[r.sign[1]+1][1] - r.origin[1]) * r.inv_direction[1]
    tmax = (Q[1-r.sign[1]+1][1] - r.origin[1]) * r.inv_direction[1]
    tymin = (Q[r.sign[2]+1][2] - r.origin[2]) * r.inv_direction[2]
    tymax = (Q[1-r.sign[2]+1][2] - r.origin[2]) * r.inv_direction[2]
    if ((tmin > tymax) || (tymin > tmax))
        return -1.0
    end
    if (tymin > tmin)
        tmin = tymin
    end
    if (tymax < tmax)
        tmax = tymax
    end
    tzmin = (Q[r.sign[3]+1][3] - r.origin[3]) * r.inv_direction[3]
    tzmax = (Q[1-r.sign[3]+1][3] - r.origin[3]) * r.inv_direction[3]

    if ((tmin > tzmax) || (tzmin > tmax))
        return -1.0
    end
    if (tzmin > tmin)
        tmin = tzmin
    end
    if (tzmax < tmax)
        tmax = tzmax
    end
    if (~isfinite(tmax) || ~isfinite(tmin))
        error("An ambiguous ray trajectory detected: ray touches the voxel at its boundary. ")
    end
    return norm((r.origin + r.dir * tmin) - (r.origin + r.dir * tmax))

end

@inline function intersects(a::HyperRectangle{N,Float64}, b::HyperRectangle{N,Float64}) where {N}
    maxA = a.origin + a.widths
    minA = a.origin
    maxB = b.origin + b.widths
    minB = b.origin
    return (all(maxA .>= minB) && all(maxB .> minA))

end

function inittree3d(n)
    empty::Array{Int64,1} = []
    oct = Cell(SVector(-1.0, -1, -1), SVector(2.0, 2, 2), empty)
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

function printdata(tree::Cell)
    for i = 1:length(tree.children)
        if (tree[i].children != nothing)
            for j = 1:length(tree[i].children)
                println(i, " ", j, " ", tree[i][j].data)
            end
        else
            println(i, " ", tree[i].data)
        end
    end
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

function possiblevoxels(r::Union{ray3d,ray2d}, tree::Cell)::Vector{Int64}
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

function voxelray(
    r::ray3d,
    vv::Vector{HyperRectangle{3,Float64}},
    checklist::Vector{Int64},
)
    N = length(checklist)
    tlens = Vector{Float64}(undef, N)
    for i = 1:N
        q = intersects(r, vv[checklist[i]])
        tlens[i] = q
    end
    store = tlens .> 0.0
    tlens = tlens[store]
    indices = checklist[store]
    return (indices, tlens)
end


@inline function rodrigues(k::T,v::T,theta::Float64)::T where {T}
    kn = k/norm(k)
    return v*cos(theta) + cross(kn,v)*sin(theta) + kn*dot(kn,v)*(1.0-cos(theta))
end

function drawrays(N, theta, fii)
    @assert length(theta) == length(fii)
    Na = length(theta)
    if (~isa(theta, Union{SArray,Array}))
        theta = [theta]
        fii = [fii]
    end
    r = 0.5 * sqrt(12)
    v1 = [1.0, 0, 0]
    v2 = [0, 1.0, 0]
    v3 = [0, 0, 1.0]
    span = range(-r, stop = r, length = N)
    using3D()
    fig = figure()
    ax = fig.add_subplot(111, projection = "3d")
    for a = 1:Na
        x1 = r * sin(theta[a]) * cos(fii[a])
        y1 = r * sin(theta[a]) * sin(fii[a])
        z1 = r * cos(theta[a])
        d = [x1, y1, z1]
        fullangle = acos(dot(d, v1) / (norm(d) * norm(v1)))
        k = cross(v1, [x1, y1, z1])
        mainaxle = rodrigues(k, v1, fullangle)
        axle2 = rodrigues(k, v2, fullangle)
        axle3 = rodrigues(k, v3, fullangle)
        center = @SVector [mainaxle[1] * r, mainaxle[2] * r, mainaxle[3] * r]
        for i = 1:N
            for j = 1:N
                p1 = center + axle2 * span[j] + axle3 * span[i]
                p2 = p1 - 2 * mainaxle * r
                ax.plot3D([p1[1], p2[1]], [p1[2], p2[2]], [p1[3], p2[3]])

            end
        end
    end


end

function constructmatrix(
    tree::Cell,
    vv::Vector{HyperRectangle{3,Float64}},
    N::Int64,
    theta,
    fii,
)
    @assert length(theta) == length(fii)
    v1 = [1.0, 0, 0]
    v2 = [0, 1.0, 0]
    v3 = [0, 0, 1.0]
    r = 0.5 * sqrt(12)

    Na = length(theta)
    if (~isa(theta, Union{SArray,Array}))
        theta = [theta]
        fii = [fii]
    end

    span = range(-r, stop = r, length = N)
    Ntotal = Na * N^2
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
        x1 = r * sin(theta[a]) * cos(fii[a])
        y1 = r * sin(theta[a]) * sin(fii[a])
        z1 = r * cos(theta[a])
        d = [x1, y1, z1]
        fullangle = acos(dot(d, v1) / (norm(d) * norm(v1)))
        k = cross(v1, [x1, y1, z1])
        mainaxle = rodrigues(k, v1, fullangle)
        axle2 = rodrigues(k, v2, fullangle)
        axle3 = rodrigues(k, v3, fullangle)
        center = @SVector [mainaxle[1] * r, mainaxle[2] * r, mainaxle[3] * r]
        Threads.@threads for i = 1:N
            for j = 1:N
                rayindex = (a - 1) * N^2 + (i - 1) * N + j
                p1 = center + axle2 * span[j] + axle3 * span[i]
                p2 = p1 - 2 * mainaxle * r
                or = ray3d(p1, p2)
                checklist = possiblevoxels(or, tree)
                indices, tlens = voxelray(or, vv, checklist)
                # append!(rows,rayindex*ones(Int64,length(tlens)))
                # append!(cols,indices)
                # append!(vals,tlens)

                append!(rows[Threads.threadid()],rayindex * ones(Int64, length(tlens)))
                append!(cols[Threads.threadid()], indices)
                append!(vals[Threads.threadid()], tlens)

            end
        end
    end
    rows = vcat(rows...)
    cols = vcat(cols...)
    vals = vcat(vals...)
    M = sparse(rows, cols, vals, Ntotal, length(vv))
    println("Matrix constructed in ", time() - t, " seconds.")
    return M
end



function setupvoxels(sizex::Int64,sizey::Int64,sizez::Int64,octsize::Int64)
    oct = inittree3d(octsize)
    l = Array([[1.0, 0.0, 0.0] [0,0,1] [1, 1, 0] [0, 0, 0] [0, 1, 1] [1, 0, 1] [1, 1, 1] [0, 1, 0]]')
    pm = mean(l, dims = 1)
    size = [sizex, sizey, sizez]
    cellsize = 2.0 ./ size
    width = @SVector[cellsize[1], cellsize[2], cellsize[3]]
    for i = 1:3
        l[:, i] =cellsize[i] .* (l[:, i] .- pm[i]) .- cellsize[i] / 2.0 * size[i] .+ 0.5 * cellsize[i]
    end
    voxelvector = Vector{HyperRectangle{3,Float64}}(undef, sizex * sizey * sizez)
    t = time()
    for k = 1:sizez
        for j = 1:sizey
            for i = 1:sizex
                ll = l .+ reshape(cellsize .* [i - 1, j - 1, k - 1], 1, 3)
                origin = @SVector[
                    minimum(ll[:, 1]),
                    minimum(ll[:, 2]),
                    minimum(ll[:, 3]),
                ]
                voxel = HyperRectangle(origin, width)
                index = j + (i - 1) * sizey + (k - 1) * sizey * sizex
                inserttotree!(voxel, oct, index)
                voxelvector[index] = voxel
            end
        end
    end
    println("Voxels initialized in ", time() - t, " seconds.")
    return (oct, voxelvector)
end

function sub2ind(sizes,i::Int64,j::Int64,k::Int64)
    @assert i > 0 && i<= sizes[1]
    @assert j > 0 && j<= sizes[2]
    @assert k > 0 && k<= sizes[3]
    return (k-1)sizes[1]*sizes[2] + sizes[1]*(j-1) + i
end

function test()
    Nx = 100; Ny = 100; Nz = 100; Os = 6; Ns = 100
    (oct,voxelvector)=setupvoxels(Nx,Ny,Nz,Os)
    theta = [0,pi/2]; fii = [0, pi/2]
    M=constructmatrix(oct,voxelvector,Ns,theta,fii)

    l = Array([[1.0, 0.0, 0.0] [0,0,1] [1, 1, 0] [0, 0, 0] [0, 1, 1] [1, 0, 1] [1, 1, 1] [0, 1, 0]]')
    pm = mean(l, dims = 1)
    size = [Nx,Ny,Nz]
    cellsize = 2.0 ./ size
    qx=Vector(range(-1 + cellsize[1]/2.0,stop=1- cellsize[1]/2.0,length=Nx))
    qy=Vector(range(-1+ cellsize[2]/2.0,stop=1- cellsize[2]/2.0,length=Ny))
    qz=Vector(range(-1+ cellsize[3]/2.0,stop=1- cellsize[3]/2.0,length=Nz))

    mgrid = Matlab.meshgrid(qx,qy,qz)
    x = mgrid[1]; y = mgrid[2]; z = mgrid[3]

    object = @. 5.0.*(x^2/1^2 + (y)^2/0.5^2 + z^2/1^2 < 0.25)
    fig = plt.figure()
    imshow(object[:,:,Nz÷2+1])
    objf = object[:]
    sino = M*objf
    sino1 = sino[1:Ns^2]
    sinor1 = reshape(sino1,Ns,Ns)
    fig = plt.figure()
    imshow(sinor1)

    fig = plt.figure()
    sino2 = sino[Ns^2+1:end]
    sinor2 = reshape(sino2,Ns,Ns)
    imshow(sinor2)
    return  nothing

end

#test()
