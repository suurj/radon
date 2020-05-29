# 2D Radon transform and 3D John transform matrices
The first Julia code constructs a discrete 2D Radon transform operator matrix by a raycasting method, when a function is approximated with
linear triangular shape functions, i.e, it works with an unstructured mesh.  The function _test_ calculates a sinogram for Shepp–Logan using the file  _phantom.mat_. The example mesh consists of 137155 shape functions and 273052 elements. 

On the other hand, the second code uses raycasting to construct a discrete 3D John transform  matrix in a structured mesh of voxels. The test function calculates projections for a simple ellipsoid. 

For instance, one can obtain discrete Radon transform for a mesh by using first the function _constructmatrix_, which returns a matrix **M**. Then **r=M\*x**, where **x** is a vector of trial function node values, and **sinogram=reshape(r,N,T)**. 
Naturally, 2D Radon transform matrix operator can be build to work with pixels by modifying the 3D routines. 

## Requirements
- RegionTrees.jl
- PyPlot.jl
- VectorizedRoutines.jl 
- StaticArrays.jl
- MAT.jl 
- Statistics.jl
- SparseArrays.jl

