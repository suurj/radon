# 2D/3D Radon transform and 3D John transform matrices
The first Julia code constructs a discrete 2D Radon transform operator matrix by a raycasting method, when a function is approximated with
linear triangular shape functions, i.e. it works with an unstructured mesh.  The function _test_ calculates a sinogram for Shepp–Logan using the file  _phantom.mat_. The example mesh consists of 137155 shape functions and 273052 elements. For instance, one can obtain discrete Radon transform for a mesh by using first the function _constructmatrix_, which returns a matrix _M_. Then _r=M\*x_, where _x_ is a vector of trial function node values, and _sinogram=reshape(r,N,T)_. 

On the other hand, _john3Dvoxel.jl_ uses raycasting to construct a discrete 3D John transform matrix in a structured mesh of voxels. An example function calculates projections of a simple ellipsoid at a few angles. 

The third code, _radon3Dvoxel.jl_, constructs a 3D Radon transform operator, which approximates the transform by applying a voxel mesh. The matrix is applied onto a vector representing a unit ball for demonstration.

The fourth code,  _radon2Dpixel.jl_, uses pixels as finite-dimensional basis for a given object and returns a parallel beam theory matrix. The same code can be also used to construct a fan-beam transform theory matrix. The function that constructs a fan-bem theory matrix follows more or less the conventions of the ODL package. However, it must be recalled that the **coordinate axis are seem to have flipped logic between ODL and this code**. When the function _test_ is called, the function _constructfanrays_ handles this automatically, but if one aims to modify the codes, this may be an issue. 


## Requirements
- RegionTrees.jl
- PyPlot.jl
- VectorizedRoutines.jl 
- StaticArrays.jl
- MAT.jl 
- Statistics.jl
- SparseArrays.jl
- ProgressBars.jl
- Images.jl (FileIO.jl, ImageMagick.jl and ImageIO.jl)


