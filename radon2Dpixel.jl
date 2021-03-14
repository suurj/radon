using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using ProgressBars
using MAT
using Images

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

    if (r.inv_direction[1] >= 0.0)
        #tmin = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmin = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmax = (prevfloat(b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
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
        tymax = (prevfloat(b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]
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

    if (isnan(tmin))
        #error("NaN detected")
       return b.widths[1]
    end

    if (isnan(tymin))
        #error("NaN detected")
       return  b.widths[2]
    end

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


function inittree2d(n;lowx= -1.0, lowy = -1.0, wx = 2.0, wy = 2.0)
    empty::Array{Int64,1} = []
    oct = Cell(SVector(lowx, lowy), SVector(wx, wx), empty)
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

    elseif (~(tree.parent === nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblepixels(r, tree[i]))
        end
        return v

    elseif ((tree.parent === nothing) && intersects(r, tree.boundary) > 0.0)
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
    if (Nproj > 1)
        pb = ProgressBar(1:Nproj)
    else
        pb = 1:1
    end
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
    if (Nproj > 1)
        pb = ProgressBar(1:Nproj)
    else
        pb = 1:1
    end
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


function setuppixels(sizex::Int64,sizey::Int64,octsize::Int64;lowy = -1.0, lowx = -1.0, widthy = 2, widthx = 2)
    @assert widthy == widthx
    oct = inittree2d(octsize;lowx= lowx, lowy = lowy, wx = widthx, wy = widthy)  
    size = [sizex, sizey]
    cellsize = [widthx,widthy] ./ size
    wds = [widthx,widthy]
    width = @SVector[cellsize[1], cellsize[2]]
    #l = Array([ [0.0, 0] [1, 0] [1, 1] [0, 1.0]]')
    #pm = mean(l, dims = 1)
    # for i = 1:2
    #     l[:, i] = cellsize[i] .* (l[:, i] .- pm[i]) .- cellsize[i] / 2.0 * size[i] .+ 0.5 * cellsize[i]
    # end
    l = [[lowx, lowy] [lowx + cellsize[1], lowy ] [lowx , lowy + cellsize[2]] [lowx + cellsize[1], lowy + cellsize[2]] ]'
    # println(l)
    # error("")
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

function constructparallelrays(Nrays,Nproj;rotations=range(-pi/2,stop=3/4*2*pi,length=Nproj),dete_plate_span=range(-sqrt(2), stop = sqrt(2), length = Nrays))

    @assert length(dete_plate_span) == Nrays
    @assert length(rotations) == Nproj
    rays = Vector{Vector{ray2d}}(undef,Nproj)

    # Parallel beam.

    for i = 1:Nproj
        span = dete_plate_span#range(r, stop = -r, length = Nrays)
        rays[i] = Vector{ray2d}(undef,Nrays)
        for j = 1:Nrays
            dir = [cos(rotations[i]), sin(rotations[i])]
            #aux = [-sin(rotations[i]), cos(rotations[i])]
            aux = [-sin(rotations[i]), cos(rotations[i])]*span[j] + center
            p1 =  aux #* span[j]
            #p2 = p1 - 2 * dir * r
            #ray = ray2d(p1, p2)
            ray = ray2ddir(p1,dir)
            rays[i][j] = ray
            
            #plot([ray.origin[1], ray.origin[1] +  ray.dir[1]],[ray.origin[2], ray.origin[2] +  ray.dir[2]] )
           # plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )

        end
    end

    return rays

end

function constructfanrays(Nrays,Nproj;translation=[0,0.0],src_to_det_init=[0,1],det_axis_init = nothing,det_shift_func=a->[0,0.0],apart=range(0,stop=pi,length=Nproj),det_radius=sqrt(2),src_radius=sqrt(2),dpart=range(-sqrt(2), stop = sqrt(2), length = Nrays))

    @assert length(dpart) == Nrays
    @assert length(apart) == Nproj
    rays = Vector{Vector{ray2d}}(undef,Nproj)

    sdinit = 2*pi*1/4+atan(src_to_det_init[2],src_to_det_init[1]) 
    apart = apart .+ sdinit
    translation = [translation[2], -translation[1]]
    dextra_rot = 0.0

    if (det_axis_init === nothing)
        dextra_rot = 0.0
    else   
        dinit = 2*pi*1/4+atan(det_axis_init[2],det_axis_init[1])
        dextra_rot = mod((2*pi*1/4-(sdinit-dinit)),2*pi)
    end
    
    span = dpart

    for i = 1:Nproj
        source = [cos(apart[i]), sin(apart[i])]*src_radius 
        p2 = source + translation
        detcenter = -[cos(apart[i]), sin(apart[i])]*det_radius
        rays[i] = Vector{ray2d}(undef,Nrays)
        forplots = zeros(2,2)
        det_tangent = [-sin(apart[i]), cos(apart[i])]
        det_orth = [cos(apart[i]), sin(apart[i])]
        detshift = det_shift_func(apart[i])
        totcenter = detcenter - detshift[1]*det_orth - detshift[2]*det_tangent
        for j = 1:Nrays                    
            aux = [-sin(apart[i]+dextra_rot), cos(apart[i]+dextra_rot)]*span[j] + totcenter
            p1 = aux + translation         
            ray = ray2d(p1, p2)
            rays[i][j] = ray

            # if(j==1)
            #     forplots[1,:] = p1
            # elseif(j==Nrays)
            #     forplots[2,:] = p1
            # end

            # plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )
            # scatter(aux[1],aux[2])
            
        end
        ## plot([rays[i][1].origin[1], rays[i][1].origin[1] +  rays[i][1].dir[1]],[rays[i][1].origin[2], rays[i][1].origin[2] +  rays[i][1].dir[2]] )
        ## plot([rays[i][end].origin[1], rays[i][end].origin[1] +  rays[i][end].dir[1]],[rays[i][end].origin[2], rays[i][end].origin[2] +  rays[i][end].dir[2]] ) 
        # plot(forplots[:,1],forplots[:,2])
        # println(forplots)
        # scatter(p2[1],p2[2])
        # xlim([-9,9])
        # ylim([-9,9])
        # axis(:equal)
     end
    #  error("")
    return rays
end


function test()

    close("all")
    
    #(qt,pixelvector)=setuppixels(Nx,Ny,Os)   # Default grid span is a rectangle with extents of [-1,-1] and [1,1].
    # fn = "r.mat"
    # file = matopen(fn)
    # zn = read(file, "S")
    # zn = zn[:,1]

    fn = "simplephantom.png"
    zn = load(fn)
    zn = channelview(zn)
    zn = zn[:]
    

    ## When compared to ODL, the logic of the detector span is the same. The span vector should be increasing, so
    ## detector_partition = odl.uniform_partition(-8, 4, 100) equals dete_plate = range(-8, stop = 4, length = 100)

    ## However, with the coordinate axis are flipped in ODL. Depending on the case, π/2 radians plus a
    ## possbile angle shift must be added to obtain the same 
    ## rotational span due to the inverted coordinate system. 
    ## This is very important if the code is modified, but the current fanbeam function takes care of the axis differences.

    ## The vector src_to_det_init defines the shift to the initial angle:
    ## angle_partition = odl.uniform_partition(0,2*np.pi, 360) 
    ## equals rotations = range(0,stop=2*pi,length=360) .+ 2*pi*1/4+atan(src_to_det_init[2],src_to_det_init[1]) 
    ## Again, this extra angle is added due to the flipped coordinate axis between ODL and this code.

    ## ODL's parameter det_axis_init=[1,-0.5] refers to extra rotation of the detector plate. That is,
    ## det_axis_init=[3,-0.7] equals dextra_rot = atan(-0.7,3)
    ## if the src_to_det_init is left to its default value. 

    ## Source and detector radii have the same logic. 
    ## By default, the center of rotation in ODL is the origin. The vector translation
    ## moves it. This is just added to  initial points of the rays and the detector plate 
    ## position vectors.
    
    ## The function det_shift_func defines a shift for the detector at each projection angle
    ## in the parallel direction of the ray that goes through the COR and in the tangent direction of the ray.
    ## For some reason, the src_shift_func does not work in ODL so it it not implemented in constructfanrays.
    ## Likely it would be as straightforward to do as det_shift_func.
    ## Curved detector plates are not also implemented.

    ## The domain of the object is odl_space and its correspondence to ODL should be clear. Since Astra toolbox and ODL do not seem to support
    ## anisotropic pixels together, the  number of pixels of the object and the geometry dimensions should be the same.


    Nx = 256; Ny = 256;  # Number of X and Y pixels,
    Os = 6 # Splitting factor of the quadtree. Does not affect the theory matrix, only the performance of building it. 6 seems optimal for 256x256.
    Nproj = 360 # Number of projections i.e. no. angles
    Nrays = 400 # Number of rays in a projection

    odl_space = (min_pt=[-1.4, -2], max_pt=[3.6, 3]) 
    src_to_det_init  = [1,5.5] #[0,1] is the default
    det_axis_init =   [-1,1.3] #[1,0] is the default
    det_shift_func(angle) = [-1.5, 0.4]   
    rotations=  range(pi/3,stop=2*pi/3,length=Nproj)
    translation = [0.5,0.9]

    dete_radius = 5.7
    source_radius = 5.0
    dete_plate = range(-6, stop = 10, length = Nrays)

    (qt,pixelvector)=setuppixels(Nx,Ny,Os;lowx=odl_space.min_pt[2],lowy=-odl_space.max_pt[1],widthx=abs(odl_space.max_pt[2]-odl_space.min_pt[2]),widthy=abs(odl_space.max_pt[1]-odl_space.min_pt[1]))  
    rays = constructfanrays(Nrays,Nproj;translation=translation, src_to_det_init = src_to_det_init, det_axis_init=det_axis_init, det_shift_func=det_shift_func,apart=rotations,det_radius=dete_radius,src_radius=source_radius, dpart=dete_plate)

    #imshow(reshape(zn[:,1],256,256),extent=[-1,1,-1,1])
    #xlim([-5,5])
    #ylim([-5,5])

    M = constructmatrix(qt, pixelvector, rays)
    
    yp = M * zn

    # For simulating a rotating object, but stationary device
    # Nlog = size(zn)[2]
    # rad = zeros(Nrays,Nlog)
    # for i = 1:Nlog
    #     p = zn[:,i]
    #     yp = M * p
    #     rad[:,i] = yp
    # end

    figure()
    rad = reshape(yp,  Nrays,Nproj)
    println(size(rad))
    imshow(rad, aspect = "auto")
    title("Julia")
    # figure()
    # imshow(reshape(zn,256,256))
    return rad,M
end

sinogram,M=test2()

# Extremely simple reconstruction.
# using RandomizedLinAlg 
# Q = rsvd(M, 500)
# y = sinogram[:]
# x = (sparse(Q.Vt')*(spdiagm(0=>1.0./Q.S))*(sparse(Q.U')*y));
# figure()
# imshow(reshape(x,256,256))



#################################################################################
## ODL code for comparison

# import numpy as np
# import odl
# import scipy.io
# from PIL import Image
# import matplotlib.pyplot as plt

# plt.close("all")

# reco_space = odl.uniform_discr(min_pt=[-1.4, -2], max_pt=[3.6, 3], shape=[256, 256], dtype='float32')

# angle_partition = odl.uniform_partition(np.pi/3,2*np.pi/3, 360)
# detector_partition = odl.uniform_partition(-6, 10, 400) 
# # If the first param of the det_shift_func is constant, the effect is the same as increasing or decreasing the detector radius.
# # Positive values mean decreasing the radius, negative values decrease it. 

# ds = lambda angle: np.array([[-1.5, 0.4]])
# geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, det_shift_func = ds, translation = [0.5,0.9], src_to_det_init=[1,5.5] , det_axis_init=[-1,1.3], src_radius=5, det_radius=5.7)

# # Ray transform (= forward projection).
# ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')

# mat = Image.open("simplephantom.png").convert('L')
# phantom = np.array(mat)/255.0
# proj_data = ray_trafo(phantom)
# p = proj_data.data.T

# print(p.shape)
# plt.figure()
# plt.imshow(p,aspect = "auto")
# plt.title("ODL")
# #plt.plot(p)
# plt.show()