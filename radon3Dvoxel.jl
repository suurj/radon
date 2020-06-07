using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using MAT

struct segment3d{T}
    origin::T
    dir::T
end


function segment3dpoint(p1::T, p2::T) where {T}
    dir::T = p2 - p1
    return segment3d(p1, dir)
end


struct plane3d
    origin::Union{SVector{3,Float64},Vector{Float64}}
    dir1::Union{SVector{3,Float64},Vector{Float64}}
    dir2::Union{SVector{3,Float64},Vector{Float64}}
    n::Union{SVector{3,Float64},Vector{Float64}}
    d::Float64
end

function plane3ddir(
    origin::T,
    dir1::T,
    dir2::T,
) where {T}
    A = [dir1 dir2]
    q = svd(A)
    dir1 = q.U[:,1]
    dir2 = q.U[:,2]
    n = cross(dir1,dir2)
    d = dot(origin,n)
    return plane3d(origin,dir1,dir2,n,d)
end

function plane3d(
    p1::T,
    p2::T,
    p3::T,
) where {T}
    origin = p1
    dir1 = p2-p1
    dir2 = p3-p1
    A = [dir1 dir2]
    q = svd(A)
    dir1 = q.U[:,1]
    dir2 = q.U[:,2]
    n = cross(dir1,dir2)
    d = dot(origin,n)
    return plane3d(origin,dir1,dir2,n,d)
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

function possiblevoxels(r::plane3d, tree::Cell)::Vector{Int64}
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


function sub2ind(sizes, i::Int64, j::Int64, k::Int64)
    @assert i > 0 && i <= sizes[1]
    @assert j > 0 && j <= sizes[2]
    @assert k > 0 && k <= sizes[3]
    return (k - 1) * sizes[1] * sizes[2] + sizes[1] * (j - 1) + i
end

function intersects(p::plane3d,r::segment3d)
    t = dot(p.n,r.origin-p.origin)/(-dot(r.dir,p.n))
    #println(t)
    if (0<=t<=1)
        return r.origin + t*r.dir
    else
        return nothing
    end
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


@inline @inbounds function intersects(
    p::plane3d,
    b::HyperRectangle{3,Float64})
    e = b.widths.*0.5
    c = b.origin + e

    r = e[1]*abs(p.n[1]) + e[2]*abs(p.n[2]) +e[3]*abs(p.n[3])
    s = dot(p.n,c) -p.d

    return  -r <= s <r

end

@inbounds @inline function planeintegal(p::plane3d,h::HyperRectangle{3,Float64})::Float64
    e = Vector{segment3d{SArray{Tuple{3},Float64,1,3}}}(undef,12)
    e[1] = segment3d(SVector(h.origin),@SVector[h.widths[1],0,0])
    e[2] = segment3d(SVector(h.origin),@SVector[0,h.widths[2],0])
    e[3] = segment3d(SVector(h.origin + [h.widths[1],0,0]),@SVector[0,h.widths[2],0])
    e[4] = segment3d(SVector(h.origin + [0,h.widths[2],0]),@SVector[h.widths[1],0,0])

    e[5] = segment3d(SVector(h.origin + [0,0,h.widths[3]]),@SVector[h.widths[1],0,0])
    e[6] = segment3d(SVector(h.origin + [0,0,h.widths[3]]),@SVector[0,h.widths[2],0])
    e[7] = segment3d(SVector(h.origin + [h.widths[1],0,h.widths[3]]),@SVector[0,h.widths[2],0])
    e[8] = segment3d(SVector(h.origin + [0,h.widths[2],h.widths[3]]),@SVector[h.widths[1],0,0])

    e[9] = segment3d(SVector(h.origin ),@SVector[0,0,h.widths[3]])
    e[10] = segment3d(SVector(h.origin + [h.widths[1],0,0]),@SVector[0,0,h.widths[3]])
    e[11] = segment3d(SVector(h.origin + [0,h.widths[2],0]),@SVector[0,0,h.widths[3]])
    e[12] = segment3d(SVector(h.origin + [h.widths[1],h.widths[2],0]),@SVector[0,0,h.widths[3]])

    points = Vector{SArray{Tuple{3},Float64,1,3}}()
    for i = 1:12
        q = intersects(p,e[i])
        if(!isnothing(q))
            push!(points,q)
            #scatter3D(q[1],q[2],q[3])
        end
        #plot3D([e[i].origin[1],e[i].origin[1]+ e[i].dir[1]],[e[i].origin[2],e[i].origin[2]+ e[i].dir[2]],[e[i].origin[3],e[i].origin[3]+ e[i].dir[3]])
    end

    N = length(points)

    if(N<=2)
        return 0.0
    end
    pointsplane = Vector{SArray{Tuple{2},Float64,1,2}}(undef,N)
    for i = 1:N
        pointsplane[i] = SVector(dot(p.dir1,points[i]),dot(p.dir2,points[i]))
    end
    c = mean(pointsplane)
    sort!(pointsplane,by=x->atan((x[2]-c[2]),(x[1]-c[1])))
    val = shoelace(pointsplane)
    return val

end

@inbounds @inline function shoelace(v::T)::Float64 where {T}
    N = length(v)
    A = 0.0
    for i = 1:N-1
        A = A+ v[i][1]*v[i+1][2] - v[i+1][1]*v[i][2]
    end
    A = A + v[N][1]*v[1][2] - v[1][1]*v[N][2]
    return 0.5*abs(A)
end

function voxelplane(
    r::plane3d,
    vv::Vector{HyperRectangle{3,Float64}},
    checklist::Vector{Int64},
)
    N = length(checklist)
    indices = Vector{Int64}()
    tlens = Vector{Float64}()
    for i = 1:N
        q = planeintegal(r, vv[checklist[i]])
        if (q>0.0)
            push!(tlens, q)
            push!(indices, checklist[i])
        end

    end
    return (indices, tlens)
end

function onlyintersections(r::plane3d,vv::Vector{HyperRectangle{3,Float64}},checklist::Vector{Int64})
    N = length(checklist)
    indices = Vector{Int64}()
    for i = 1:N
        if( intersects(r, vv[checklist[i]]))
            push!(indices, checklist[i])
        end
    end

    return indices

end

function constructmatrix(
    tree::Cell,
    vv::Vector{HyperRectangle{3,Float64}},
    N::Int64,
    planevector::Vector{plane3d})

    r = 0.5 * sqrt(12)
    Na = length(planevector)
    if (N > 1)
        span = range(r, stop = -r, length = N)
    else
        span = [0.0]
    end
    Ntotal = Na * N

    rows = Vector{Int64}()
    cols = Vector{Int64}()
    vals = Vector{Float64}()

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
        pl = planevector[a]
        Threads.@threads for j = 1:N
            #for j = 1:N
                planeindex = (a - 1) * N + j
                newp = plane3ddir(pl.origin + pl.n.*span[j],pl.dir1,pl.dir2)
                checklist = possiblevoxels(newp, tree)
                checklist = onlyintersections(newp,vv,checklist)
                indices, tlens = voxelplane(newp, vv, checklist)
                # append!(rows,planeindex*ones(Int64,length(tlens)))
                # append!(cols,indices)
                # append!(vals,tlens)

                append!(rows[Threads.threadid()],planeindex * ones(Int64, length(tlens)))
                append!(cols[Threads.threadid()], indices)
                append!(vals[Threads.threadid()], tlens)

            end
        end

    rows = vcat(rows...)
    cols = vcat(cols...)
    vals = vcat(vals...)
    M = sparse(rows, cols, vals, Ntotal, length(vv))
    println("Matrix constructed in ", time() - t, " seconds.")
    return M
end

function test()
    Nx = 100; Ny = 100; Nz =100; Os = 6; Ns = 100
    (oct,voxelvector)=setupvoxels(Nx,Ny,Nz,Os)
    p1=plane3ddir([0,0,0.0],[1,1.0,1.0],[0.0,1,1.0])
    p2=plane3ddir([0,0,0.0],[1,0.0,0.0],[0.0,0,1.0])
    planevector=[p1,p2]

    # Plotting a part of the first plane.
    c1 = p1.origin + p1.dir1*3 + p1.dir2*3
    c2 = p1.origin + p1.dir1*3 - p1.dir2*3
    c3 = p1.origin - p1.dir1*3 + p1.dir2*3
    c4 = p1.origin - p1.dir1*3 - p1.dir2*3
    xx = reshape([c1[1],c2[1],c3[1],c4[1]],2,2)
    yy = reshape([c1[2],c2[2],c3[2],c4[2]],2,2)
    z = reshape([c1[3],c2[3],c3[3],c4[3]],2,2)
    plot_surface(xx,yy,z)


    M=constructmatrix(oct,voxelvector,Ns,planevector)

    l = Array([[1.0, 0.0, 0.0] [0,0,1] [1, 1, 0] [0, 0, 0] [0, 1, 1] [1, 0, 1] [1, 1, 1] [0, 1, 0]]')
    pm = mean(l, dims = 1)
    size = [Nx,Ny,Nz]
    cellsize = 2.0 ./ size
    qx=Vector(range(-1 + cellsize[1]/2.0,stop=1- cellsize[1]/2.0,length=Nx))
    qy=Vector(range(-1+ cellsize[2]/2.0,stop=1- cellsize[2]/2.0,length=Ny))
    qz=Vector(range(-1+ cellsize[3]/2.0,stop=1- cellsize[3]/2.0,length=Nz))

    mgrid = Matlab.meshgrid(qx,qy,qz)
    x = mgrid[1]; y = mgrid[2]; z = mgrid[3]

    object = @. 1.0.*(x^2 + y^2 + z^2 < 1) #Testing Radon transform of a unit ball.
    objf = object[:]
    sino = M*objf
    sino1 = sino[1:Ns]
    sino2 = sino[Ns+1:end]
    fig = plt.figure()
    plot(sino1)
    plot(sino2)

    return  nothing
end

test()
