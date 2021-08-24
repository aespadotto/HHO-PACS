This is an adaptation of the library HArD::Core (https://github.com/jdroniou/HArDCore) for educational purpose. 
The documentation of the original project can be found at https://jdroniou.github.io/HArDCore3D/

An extension is provided to apply the HHO method to the equations of magnetostatics. The directory 'magnetostatics-spadotto' contains
new or partially adapted code to implement the scheme. The material of the added branch resides mostly there.
The common library utilities are located in 'src'. Only minor changes have been introduced under 'src/common'.
Under 'Schemes/HHO-magnetostatics' the original version of the implementation can be found, which may be useful for a comparison.   
'Schemes' also contains implementations of hybrid methods on polytopal meshes. 
Meshfiles are found under 'meshes'.

Building requirements:

* Item CMake version 2.6 or above (https://cmake.org/)
* Item A C++ compiler that supports the C++14 standard, eg. GCC (https://gcc.gnu.org/) or Clang (https://clang.llvm.org/)
* Item Eigen C++ library, version 3.3 or above (http://eigen.tuxfamily.org/)
* Item The following Boost C++ libraries (http://www.boost.org/): filesystem, program options, timer, chrono
* Item GetPot(http://getpot.sourceforge.net/)

Building instructions: 

An automatic search of the external libraries is provided. 
If it fails,  setting the directories manually in CMakeLists.txt where indicated should fix the problem. 
GetPot's directory has to be set manually in every case.
For the minimal build it should be enough to execute the following steps:

```
mkdir build
cd build
cmake ..
make spadotto-magnetostatics/magnetostatics-spadotto
```
 
After this, the executable 'magnetostatics-spadotto' can be found under 'build/spadotto-magnetostatics' 

The executable 'magnetostatics-spadotto' can be launched specifying several options. You can have a look at 
'spadotto-magnetostatics/hho-magnetostatics-sum.cpp' to check the default setting.  

In the directory 'runs/magnetostatics-spadotto' the script 'runseries.sh' provides an utility to perform a convergence test. 
You will have to set for your system the content of 'runs/directories.sh'. To change the data of the simulation you can modify 'data.sh'.


