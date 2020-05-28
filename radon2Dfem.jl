using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using MAT

struct triangle2d
    pts::Array{Array{Float64,1},1}
    nodes::Union{Array{Int64,1},SVector{3,Int64}}
    bbox::HyperRectangle{2,Float64}
    A::Float64
end

struct ray2d
    origin::Union{SVector{2,Float64},Vector{Float64}}
    dir::Union{SVector{2,Float64},Vector{Float64}}
    inv_direction::Union{SVector{2,Float64},Vector{Float64}}
    sign::Union{Array{Int64,1},SVector{2,Int64}}
end

function triangle2d(
    p1::T,
    p2::T,
    p3::T,
    nlist::Union{Array{Int64,1},SArray{Tuple{3},Int64,1,3}},
) where {T}
    origin = [minimum([p1[1], p2[1], p3[1]]), minimum([p1[2], p2[2], p3[2]])]
    final = [maximum([p1[1], p2[1], p3[1]]), maximum([p1[2], p2[2], p3[2]])]
    width = final - origin
    origin = @SVector[origin[1], origin[2]]
    width = @SVector[width[1], width[2]]
    bbox = HyperRectangle(origin, width)
    A =
        0.5 * abs(
            p1[1] * p2[2] + p2[1] * p3[2] + p3[1] * p1[2] - p2[1] * p1[2] -
            p3[1] * p2[2] - p1[1] * p3[2],
        )
    return triangle2d([p1, p2, p3], nlist, bbox, A)
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
    parameters = [b.origin, b.origin + b.widths]
    tmin = (parameters[r.sign[1]+1][1] - r.origin[1]) * r.inv_direction[1]
    tmax = (parameters[1-r.sign[1]+1][1] - r.origin[1]) * r.inv_direction[1]
    tymin = (parameters[r.sign[2]+1][2] - r.origin[2]) * r.inv_direction[2]
    tymax = (parameters[1-r.sign[2]+1][2] - r.origin[2]) * r.inv_direction[2]
    if ((tmin > tymax) || (tymin > tmax))
        return -1.0
    end
    if (tymin > tmin)
        tmin = tymin
    end
    if (tymax < tmax)
        tmax = tymax
    end
    if (~isfinite(tmax) || ~isfinite(tmin))
        error("An ambiguous ray trajectory detected: ray touches the pixel at its boundary. ")
    end
    return norm((r.origin + r.dir * tmin) - (r.origin + r.dir * tmax))

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


function elementray(r::ray2d, vv::Vector{triangle2d}, checklist::Vector{Int64})
    N = length(checklist)
    tlens = Vector{MArray{Tuple{2},MArray{Tuple{2},Float64,1,2},1,2}}()
    indices = Vector{Int64}()
    for i = 1:N
        q = intersects(r, vv[checklist[i]])
        if (~isnothing(q))
            push!(tlens, q)
            push!(indices, checklist[i])
        end
    end
    return (indices, tlens)
end


function constructmatrix(tree::Cell, vv::Vector{triangle2d}, N::Int64, theta)
    r = 1.0 # Radius of the transform.
    Na = length(theta)
    if (~isa(theta, Union{SArray,Array}))
        theta = [theta]
    end
    if (N > 1)
        span = range(r, stop = -r, length = N)
    else
        span = [0.0]
    end
    Ntotal = Na * N
    rows = Vector{Int64}()
    cols = Vector{Int64}()
    vals = Vector{Float64}()
    Nofnodes = length(Set(vcat([i.nodes for i in vv]...)))

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
        Threads.@threads for i = 1:N
            rayindex = (a - 1) * N + i
            p1 = dir + aux * span[i]
            p2 = p1 - 2 * dir * r
            or = ray2d(p1, p2)
            checklist = possiblevoxels(or, tree) #Possible elements.
            indices, tlens = elementray(or, vv, checklist)
            Nel = length(indices)
            for el = 1:Nel
                nodes = vv[indices[el]].nodes
                integrals = elementintegral(tlens[el], vv[indices[el]])
                #
                # append!(rows,rayindex*ones(Int64,3))
                # append!(cols,nodes)
                # append!(vals,integrals)

                append!(rows[Threads.threadid()], rayindex * ones(Int64, 3))
                append!(cols[Threads.threadid()], nodes)
                append!(vals[Threads.threadid()], integrals)

            end
        end
    end
    rows = vcat(rows...)
    cols = vcat(cols...)
    vals = vcat(vals...)
    M = sparse(rows, cols, vals, Ntotal, Nofnodes)
    println("Matrix constructed in ", time() - t, " seconds.")
    return M
end


function sub2ind(sizes, i::Int64, j::Int64, k::Int64)
    @assert i > 0 && i <= sizes[1]
    @assert j > 0 && j <= sizes[2]
    @assert k > 0 && k <= sizes[3]
    return (k - 1) * sizes[1] * sizes[2] + sizes[1] * (j - 1) + i
end


@inbounds function intersects(
    r::ray2d,
    t::triangle2d,
)::Union{MArray{Tuple{2},MArray{Tuple{2},Float64,1,2},1,2},Nothing}
    v = r.dir
    ips = @MVector [@MVector[NaN, NaN], @MVector[NaN, NaN]]
    cnt::Int64 = 0
    for i = 1:3
        j = mod(i, 3) + 1
        u = @SVector[t.pts[j][1] - t.pts[i][1], t.pts[j][2] - t.pts[i][2]]
        c = u[1] * v[2] - u[2] * v[1]
        a = @SVector[t.pts[i][1] - r.origin[1], t.pts[i][2] - r.origin[2]]
        b = a[1] * v[2] - a[2] * v[1]
        if (c == 0 && b == 0)
            println("Ray coincides with an edge. Ray is perturbed to avoid counting the integral twice. ")
            rnew = ray2ddir(r.origin + 1e-10 .* [-v[2], v[1]], r.dir)
            return intersects(rnew, t)
        end

        k = -b / c

        if (0 < k <= 1.0)
            cnt = cnt + 1
            ips[cnt][1] = t.pts[i][1] + u[1] * k
            ips[cnt][2] = t.pts[i][2] + u[2] * k

        end

    end
    if (cnt >= 2)
        return ips
    else
        return nothing

    end
end

@inline @inbounds function  gaussq(fvals,edge)::Float64
    val = fvals[1]*0.555555556 + fvals[2]*0.888888889 + 0.555555556*fvals[3];
    val = val*sqrt(((edge[1][1]-edge[2][1])/2 )^2 + ((edge[1][2]-edge[2][2])/2 )^2);
    return val
end


@inline @inbounds function localtoglobal(s, edge)
    Ns = length(s)
    z = zeros(MMatrix{Ns,2,Float64})
    for i = 1:Ns
        z[i, 1] =
            s[i] * (edge[2][1] - edge[1][1]) / 2 + (edge[1][1] + edge[2][1]) / 2
        z[i, 2] =
            s[i] * (edge[2][2] - edge[1][2]) / 2 + (edge[1][2] + edge[2][2]) / 2
    end
    return z
end


@inbounds function shapefun(x,n1,n2,A::Float64)
    val = 1/(2*A).*(n1[1].*n2[2]-n2[1].*n1[2] .+ (n1[2] - n2[2]).*x[:,1] + (n2[1] - n1[1]).*x[:,2] );
    return val
end

 @inbounds function shapefun!(val,x,n1,n2,A::Float64)
   val .= 1/(2*A).*(n1[1].*n2[2]-n2[1].*n1[2] .+ (n1[2] - n2[2]).*x[:,1] + (n2[1] - n1[1]).*x[:,2] );
   @assert maximum(val) <= 1.00001 && minimum(val) >= -0.00001
   # We do not want use a shapefunction out of its support.
end

function elementintegral(edge,t::triangle2d)
    v = @MVector[0.0,0.0,0.0]
    qp = localtoglobal(@SVector[-0.774596669,0,0.774596669],edge)
    fv = @MVector [0.0,0.0,0.0]
    for i = 1:3
        n1 = mod(i,3) + 1;
        n2 = mod(i+1,3) + 1;
        shapefun!(fv,qp,t.pts[n1],t.pts[n2],t.A)
        v[i] = gaussq(fv,edge)
    end

    return v
end

function nodeslookup(filename)
    file = matopen(filename)
    p=read(file,"p")
    e=read(file,"e")
    t=read(file,"t")
    t = Array{Int64}(t)
    quad = inittree2d(7) # Do 7 divisions for quadtree.
    Nelem = size(t)[2]
    trianglevector = Vector{triangle2d}(undef,Nelem)
    tt = time()
    for i =1:Nelem
        trianglevector[i] = triangle2d(p[:,t[1,i]],p[:,t[2,i]],p[:,t[3,i]],t[1:3,i])
        inserttotree!(trianglevector[i].bbox, quad, i)
    end
    println("Quadtree initialized in " ,time()-tt ," seconds.")

    return quad,trianglevector
end

function test()
    fn = "phantom.mat"
    quad, trianglevector = nodeslookup(fn)
    Npr = 400
    Na = 100
    M = constructmatrix(
        quad,
        trianglevector,
        Npr,
        Array(range(pi / 2, stop = 3 / 2 * pi, length = Na)),
    )
    file = matopen(fn)
    zn = read(file, "zn")
    zn = zn'
    y = M * zn
    rad = reshape(y, Npr, Na)
    imshow(rad, aspect = "auto")
    return nothing
end

#test()
