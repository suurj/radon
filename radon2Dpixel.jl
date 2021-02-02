using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using ProgressBars
using RandomizedLinAlg
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

    if (r.inv_direction[1] >= 0.0)
        #tmin = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmin = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmax = ((b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
    else
        #tmin = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmin = ((b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmax = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
    end

    if (r.inv_direction[2] >= 0.0)
        #tymin = (parameters[1][2] - r.origin[2]) * r.inv_direction[2]
        tymin = (b.origin[2]  - r.origin[2]) * r.inv_direction[2]      
        #tymax = (parameters[2][2]- r.origin[2]) * r.inv_direction[2]
        tymax = ((b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]
    else
        #tymin = (parameters[2][2] - r.origin[2]) * r.inv_direction[2]
        tymin = ((b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]      
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

function possiblepixels(r::ray2d, tree::Cell)::Vector{Int64}
    if (isleaf(tree) && intersects(r, tree.boundary) > 0.0)
        return tree.data

    elseif (~(tree.parent == nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblepixels(r, tree[i]))
        end
        return v

    elseif ((tree.parent == nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblepixels(r, tree[i]))
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

function constructmatrix(tree::Cell, vv::Vector{HyperRectangle{2,Float64}}, Nrays::Int64, theta)
    r = sqrt(2) # Radius of the transform.
    Nproj = length(theta)
    if (~isa(theta, Union{SArray,Array}))
        theta = [theta]
    end

    span = range(r, stop = -r, length = Nrays)

    Ntotal = Nrays * Nproj
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
    pb = ProgressBar(1:Nproj)
    for a in pb
        dir = [cos(theta[a]), sin(theta[a])]
        aux = [-sin(theta[a]), cos(theta[a])]
        Threads.@threads for i = 1:Nrays
            rayindex = (a - 1) * Nrays + i
            p1 = dir + aux * span[i]
            p2 = p1 - 2 * dir * r
            or = ray2d(p1, p2)
            checklist = possiblepixels(or, tree) #Possible pixels.
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



function constructmatrix(tree::Cell, vv::Vector{HyperRectangle{2,Float64}}, rays::Array{Array{ray2d,1},1})
    r = sqrt(2) # Radius of the transform.
    Nproj = length(rays)
    for i = 1:Nproj-1
        @assert length(rays[i]) == length(rays[i+1])
    end
    Nrays = length(rays[1])
    Ntotal = Nrays * Nproj
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
    pb = ProgressBar(1:Nproj)
    for a in pb
        Threads.@threads for i = 1:Nrays
            rayindex = (a - 1) * Nrays + i
            or = rays[a][i]            
            checklist = possiblepixels(or, tree) #Possible pixels.
            indices, tlens = pixelray(or, vv, checklist)
            Nel = length(indices)

            append!(rows[Threads.threadid()],rayindex * ones(Int64, length(tlens)))
            append!(cols[Threads.threadid()], indices)
            append!(vals[Threads.threadid()], tlens)
  
        end
        # if (a==Nproj)
        #     cols = vcat(cols...)
        #     ix = zeros(256^2,)
        #     ix[cols] .= 1
        #     figure()
        #     imshow(reshape(ix,256,256))
        #     error("")
        # end
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
    pixelvector = Vector{HyperRectangle{2,Float64}}(undef, sizex * sizey)
    t = time()

    for j = 1:sizey
        for i = 1:sizex
            ll = l .+ reshape(cellsize .* [i - 1,  sizey - j + 1], 1, 2) # Flip Y-axis so that we have canonical Euclidean coordinate system.
            origin = @SVector[
                minimum(ll[:, 1]),
                minimum(ll[:, 2]),
            ]
            pixel = HyperRectangle(origin, width)
            index = j + (i - 1) * sizey 
            inserttotree!(pixel, oct, index)
            pixelvector[index] = pixel
        end
    end

    println("Pixels initialized in ", time() - t, " seconds.")
    return (oct, pixelvector)
end

function constructrays(Nrays,Nproj;center=[0,0.0])

    rays = Vector{Vector{ray2d}}(undef,Nproj)

    # Parallel beam.
    # r = sqrt(2)
    # rotations = range(0,stop=2*pi,length=Nproj)

    # for i = 1:Nproj
    #     span = range(r, stop = -r, length = Nrays)
    #     rays[i] = Vector{ray2d}(undef,Nrays)
    #     for j = 1:Nrays
    #         dir = [cos(rotations[i]), sin(rotations[i])]
    #         #aux = [-sin(rotations[i]), cos(rotations[i])]
    #         aux = [-sin(rotations[i]), cos(rotations[i])]*span[j] + center
    #         p1 =  aux #* span[j]
    #         #p2 = p1 - 2 * dir * r
    #         #ray = ray2d(p1, p2)
    #         ray = ray2ddir(p1,dir)
    #         rays[i][j] = ray
            
    #         #plot([ray.origin[1], ray.origin[1] +  ray.dir[1]],[ray.origin[2], ray.origin[2] +  ray.dir[2]] )
    #        # plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )

    #     end
    # end


    # # Fan-beam.
    dsource_origo = 5
    ddetect_origo = 5
    r = sqrt(2)*(dsource_origo+ddetect_origo)/dsource_origo
    
    span = range(-2r, stop = r, length = Nrays)
    if (Nproj >1 )
        rotations = range(0,stop=-2*pi,length=Nproj)
    else
        rotations = [0.0,]
    end
    for i = 1:Nproj
        source = [cos(rotations[i]), sin(rotations[i])]*dsource_origo + center
        detcenter = -[cos(rotations[i]), sin(rotations[i])]*ddetect_origo
        rays[i] = Vector{ray2d}(undef,Nrays)
        for j = 1:Nrays           
            aux = [-sin(rotations[i]), cos(rotations[i])]*span[j] + detcenter + center
            p1 = aux
            p2 = source
            ray = ray2d(p1, p2)
            rays[i][j] = ray

            #plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )
            
        end
    #     #plot([rays[i][1].origin[1], rays[i][1].origin[1] +  rays[i][1].dir[1]],[rays[i][1].origin[2], rays[i][1].origin[2] +  rays[i][1].dir[2]] )
    #     #plot([rays[i][end].origin[1], rays[i][end].origin[1] +  rays[i][end].dir[1]],[rays[i][end].origin[2], rays[i][end].origin[2] +  rays[i][end].dir[2]] )
    #     #plot(-sin(rotations[i])*[span[1],span[end]]*(dsource_origo+ddetect_origo)/dsource_origo .+ detcenter[1] .+ center[1] ,cos(rotations[i])*[span[1],span[end]]*(dsource_origo+ddetect_origo)/dsource_origo .+ detcenter[2] .+center[2] )
    #     #error("")
     end
    
    return rays
end


function test()
    #fn = "phanpix256.mat"
    fn = "r.mat"
    Nx = 256; Ny = 256; Os = 6; 
    Nproj = 360; Nrays = 250
    (qt,pixelvector)=setuppixels(Nx,Ny,Os)
    cent = [-0.5,0.5]
    file = matopen(fn)
    #zn = read(file, "m")
    # zn = zn[:]
    close("all")
    #figure()
    zn = read(file, "S")
    
    rays = constructrays(Nrays,Nproj;center=cent)
    #imshow(reshape(zn[:,1],256,256),extent=[-1,1,-1,1])
    #xlim([-5,5])
    #ylim([-5,5])

    # for i = 1:length(rays[2])
    #     ray = rays[2][i]
    #     ro = ray.origin
    #     re = ray.origin + ray.dir
    #     plot([ro[1], re[1]], [ro[2],re[2]])
    # end
    #error("")
    M = constructmatrix(qt, pixelvector, rays)
    # M = constructmatrix(
    #     qt,
    #     pixelvector,
    #     Nrays,
    #     Array(range(0/ 2, stop = pi, length = Nproj)),
    # )
    
    y = M * zn[:,1]

    # Nlog = size(zn)[2]
    # rad = zeros(Nrays,Nlog)
    # for i = 1:Nlog
    #     p = zn[:,i]
    #     y = M * p
    #     rad[:,i] = y
    # end
    figure()
    rad = reshape(y,  Nrays,Nproj)
    imshow(rad, aspect = "auto")
    return y,M
end

y,M=test()

# Simple reconstruction.
figure()
Q = rsvd(M, 400)
x = (sparse(Q.Vt')*(spdiagm(0=>1.0./Q.S))*(sparse(Q.U')*y));
imshow(reshape(x,256,256))



#################################################################################

## MATLAB code to compare (detector&source are not in the same scale, however).
# clear all
# close all
# clc

# N = 256;
# m = phantom(N);
# % theta = (  linspace(0,pi,180))/(2*pi)*360;
# % %m = [zeros(256,256),m, zeros(256,0); zeros(256,512)];
# % I = radon(m,theta)./(N*0.5);
# % 
# % imagesc(I)
# % 
# % return

# D = 256;
# [F,G,H]=fanbeam(m,D,'FanSensorGeometry', 'line',  'FanRotationIncrement',1);
# imagesc(F./(N*0.5));

# %K = F;
# %K(:,180:end) = F(:,1:181); K(:,1:180) = F(:,181:end);
# %K = [K; zeros(20,360)]; K(1:20,:) = []; 

# %figure
# %I = ifanbeam(F,D,'FanSensorGeometry', 'line', 'OutputSize', 200);
# %imagesc(I)
# figure
# imagesc(m)

# %sum(sum((m-I).^2))