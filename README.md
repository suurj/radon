# 2D/3D Radon transform and 3D John transform matrices
The first Julia code constructs a discrete 2D Radon transform operator matrix by a raycasting method, when a function is approximated with
linear triangular shape functions, i.e. it works with an unstructured mesh.  The function _test_ calculates a sinogram for Shepp–Logan using the file  _phantom.mat_. The example mesh consists of 137155 shape functions and 273052 elements. For instance, one can obtain discrete Radon transform for a mesh by using first the function _constructmatrix_, which returns a matrix **M**. Then **r=M\*x**, where **x** is a vector of trial function node values, and **sinogram=reshape(r,N,T)**. 

On the other hand, the second code uses raycasting to construct a discrete 3D John transform  matrix in a structured mesh of voxels. The test function calculates projections for a simple ellipsoid. 

The third code, _radon3Dvoxel.jl_, constructs a 3D Radon transform operator, which approximates the transform by applying a voxel mesh.

The fourth code,  _radon2Dpixel.jl_, uses pixels as finite-dimensional basis for a given object. The same code can be also used to construct a fan-beam transform theory matrix. This is demonstrated in a case, where a phantom is rotated off-the-origin around its center of mass (file _r.mat_).

## Requirements
- RegionTrees.jl
- PyPlot.jl
- VectorizedRoutines.jl 
- StaticArrays.jl
- MAT.jl 
- Statistics.jl
- SparseArrays.jl
- ProgressBars.jl
- (RandomizedLinAlg.jl, for a very simple reconstruction in 2D pixel case)


