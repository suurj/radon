# 2D Radon transform and 3D John transform matrices
The first Julia code constructs a 2D Radon transform matrix by a raycasting method, when a function is approximated with
linear triangular shape functions.  The supplied test function calculates a sinogram for Shepp–Logan using the file  _phantom.mat_. 

On the other hand, the second code uses raycasting to construct a 3D John transform matrix. The test function calculates projections for a simple ellipsoid.

## Requirements
- RegionTrees.jl
- PyPlot.jl
- VectorizedRoutines.jl 
- StaticArrays.jl
- MAT.jl 
