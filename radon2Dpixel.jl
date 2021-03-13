using RegionTrees
using StaticArrays
using PyPlot
using Statistics
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using ProgressBars
#using RandomizedLinAlg
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
        detcenter = -[cos(apart[i]), sin(apart[i])]*det_radius
        rays[i] = Vector{ray2d}(undef,Nrays)
        # ps = zeros(2,2)
        det_tangent = [-sin(apart[i]), cos(apart[i])]
        det_orth = [cos(apart[i]), sin(apart[i])]
        detshift = det_shift_func(apart[i])
        totcenter = detcenter - detshift[1]*det_orth - detshift[2]*det_tangent
        for j = 1:Nrays                    
            aux = [-sin(apart[i]+dextra_rot), cos(apart[i]+dextra_rot)]*span[j] + totcenter
            #aux[1] = (aux[1] - totcenter[1])*cos(dextra_rot) - (aux[2] - totcenter[2])*sin(dextra_rot) + totcenter[1]
            #aux[2] = (aux[2] - totcenter[2])*cos(dextra_rot) + (aux[1] - totcenter[1])*sin(dextra_rot) + totcenter[2]
            p1 = aux + translation
            p2 = source + translation
            ray = ray2d(p1, p2)
            rays[i][j] = ray

            # if(j==1)
            #     ps[1,:] = aux
            # elseif(j==Nrays)
            #     ps[2,:] = aux
            # end

            #plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )
            #scatter(aux[1],aux[2])
            
        end
        # plot([rays[i][1].origin[1], rays[i][1].origin[1] +  rays[i][1].dir[1]],[rays[i][1].origin[2], rays[i][1].origin[2] +  rays[i][1].dir[2]] )
        # plot([rays[i][end].origin[1], rays[i][end].origin[1] +  rays[i][end].dir[1]],[rays[i][end].origin[2], rays[i][end].origin[2] +  rays[i][end].dir[2]] )
        
        ###plot(-sin(apart[i])*[span[1],span[end]]*(dsource_origo+ddetect_origo)/dsource_origo .+ totcenter[1] , cos(apart[i])*[span[1],span[end]]*(dsource_origo+ddetect_origo)/dsource_origo .+ totcenter[2]  )
        # plot(ps[:,1],ps[:,2])
        # scatter(source[1],source[2])
        # scatter(totcenter[1],totcenter[2])
        # println(norm((ps[1,:] - ps[2,:])) ,ps)
        # #xlim([-9,9])
        # #ylim([-9,9])
        # axis(:equal)
     end
     #error("")
    return rays
end


function test()

    close("all")
    
    #(qt,pixelvector)=setuppixels(Nx,Ny,Os)   # Default grid span is a rectangle with extents of [-1,-1] and [1,1].
    #fn = "r.mat"
    #file = matopen(fn)
    # zn = read(file, "S")
    # zn = zn[:,1]

    fn = "kuvio.png"
    zn = load(fn)
    zn = channelview(zn)
    zn = zn[:]
    

    ## When compared to ODL, the logic of the detector span is the same. The span vector should be increasing, so
    ## detector_partition = odl.uniform_partition(-8, 4, 100) equals dete_plate = range(-8, stop = 4, length = 100)

    ## However, with the default angle settings in ODL,  π/2 radians plus a shift must be added  to obtain the same 
    ## rotational span due to the inverted coordiante system logic. 
    ## This is very important if the code is modified, but the current fanbeam function takes care of that.
    ## The vector src_to_det_init defines the shift to the initial angle:
    ## angle_partition = odl.uniform_partition(0,2*np.pi, 360) 
    ## equals  rotations = range(0,stop=2*pi,length=360) .+ 2*pi*1/4+atan(src_to_det_init[2],src_to_det_init[1]) 
    ## This is likely due to the flipped coordinate axis between ODL and this code.

    ## ODL's parameter det_axis_init=[1,-0.5] refers to extra rotation of the detector plate. That is,
    ## det_axis_init=[3,-0.7] equals dextra_rot = atan(-0.7,3)

    ## Source and detector radii have the same logic.

    ## TODO: translations/shifts.

    Nx = 256; Ny = 256;  # Number of X and Y pixels,
    Os = 6 # Splitting factor of the pixel quadtree.
    Nproj = 360 # Number of projections i.e. no. angles
    Nrays = 300 # Number of rays in a projection

    odl_space = (min_pt=[-1.4, -2.0], max_pt=[3.6, 3.0]) 
    src_to_det_init  =   ([1,5.5])
    det_axis_init =  ([-1,1.3])
    det_shift_func(angle) = [-1.5,0.7]      
    rotations=  range(0,stop=2*pi,length=Nproj)
    translation = [0.9,-0.5]

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
    return rad,M,qt
end

y,M,qt=test()

# Simple reconstruction.
# figure()
#Q = rsvd(M, 500)
# x = (sparse(Q.Vt')*(spdiagm(0=>1.0./Q.S))*(sparse(Q.U')*y));
# imshow(reshape(x,256,256))



#################################################################################


# import numpy as np
# import odl
# import scipy.io
# from PIL import Image
# import matplotlib.pyplot as plt

# plt.close("all")

# # Reconstruction space: discretized functions on the rectangle
# # [-20, 20]^2 with 300 samples per dimension.
# reco_space = odl.uniform_discr(
#     min_pt=[-1.4, -2], max_pt=[3.6, 3], shape=[256, 256], dtype='float32')


# # Make a fan beam geometry with flat detector
# # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
# angle_partition = odl.uniform_partition(0,2*np.pi, 360)
# # Detector: uniformly sampled, n = 512, min = -30, max = 30
# detector_partition = odl.uniform_partition(-6, 10, 400) # What is the coordinate system? det_axis_init=[1,-1]
# #geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition, src_radius=50, det_radius=np.sqrt(2),translation=[-0.5,-0.5])det_axis_init=[3,-0.7],
# # If the first param of the det_shift_func is constant, the effect is the same as increasing or decreasing the detector radius.
# # Positive means decreasing the radius, negative decreasing. det_shift_func = lambda angle: [-1.5, 0.7],  det_axis_init=[1,1] det_axis_init=[-1,1.3],

# #det_axis_init=[3,-0.7]
# geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, det_shift_func = lambda angle: [-1.5, 0.7], translation = [0.5,0.9], src_to_det_init=[1,5.5] , det_axis_init=[-1,1.3], src_radius=5, det_radius=5.7)
# #geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition,  translation = [0.0,0.0],  src_radius=5, det_radius=5.7)


# # Ray transform (= forward projection).
# ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')

# # Create a discrete Shepp-Logan phantom (modified version)
# # mat = scipy.io.loadmat('r.mat')
# # M = mat['S']
# # mat = M[:,0]
# # phantom = np.reshape(mat,(256,256)).T
# # proj_data = ray_trafo(phantom)
# # p = proj_data.data.T

# mat = Image.open("kuvio.png").convert('L')
# # wpercent = (Nimage / float(mat.size[0]))
# # size = int((float(mat.size[1]) * float(wpercent)))
# # mat = mat.resize((size, size), Image.ANTIALIAS)
# phantom = np.array(mat)/255.0
# proj_data = ray_trafo(phantom)
# p = proj_data.data.T

# # A = np.zeros((512,360))
# # for i in range(360):
# #     mat = M[:,i]
# #     phantom = np.reshape(mat,(256,256)).T
# #     proj_data = ray_trafo(phantom)
# #     p = proj_data.data
# #     A[:,i] = p
# #phantom = odl.phantom.shepp_logan(reco_space, modified=True)
# #phantom=odl.phantom()

# # Create projection data by calling the ray transform on the phantom
# #proj_data = ray_trafo(phantom)

# # Back-projection can be done by simply calling the adjoint operator on the
# # projection data (or any element in the projection space).
# backproj = ray_trafo.adjoint(proj_data)

# # Shows a slice of the phantom, projections, and reconstruction

# #plt.imshow(backproj)
# print(p.shape)
# plt.figure()
# plt.imshow(p,aspect = "auto")
# plt.title("ODL")
# #plt.plot(p)
# plt.show()
# #proj_data.show(title='Projection Data (sinogram)',force_show=True)
# #backproj.show(title='Back-projection', force_show=True)

