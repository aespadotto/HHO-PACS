This is an adaptation of the library HArD::Core (https://github.com/jdroniou/HArDCore) for educational purpose. 
The documentation of the original project can be found at https://jdroniou.github.io/HArDCore3D/

An extension is provided to apply the HHO method to the equations of magnetostatics. The folder 'magnetostatics-spadotto' contains
new or partially adapted code to implement the scheme. The material of the added branch resides mostly there.
Minor changes have been done on the common library under 'src/common'. 

Building requirements:

* Item CMake version 2.6 or above (https://cmake.org/)
* Item A C++ compiler that supports the C++14 standard, eg. GCC (https://gcc.gnu.org/) or Clang (https://clang.llvm.org/)
* Item Eigen C++ library, version 3.3 or above (http://eigen.tuxfamily.org/)
* Item The following Boost C++ libraries (http://www.boost.org/): filesystem, program options, timer, chrono
* Item GetPot(http://getpot.sourceforge.net/)

Building instructions: 

For the minimal build it should be enough to follow the usual steps for a CMake project:

```
mkdir build
cd build
cmake ..
make spadotto-magnetostatics/magnetostatics-spadotto
```



If the automatic search of the external libraries fails it is recommended to set the directories manually in the CMakeLists.txt. 

The executable 'magnetostatics-spadotto' can be launched specifying several otions. For a description you can type:
```
magnetostatics-spadotto --help
```

In the directory 'runs/magnetostatics-spadotto' the script 'runseries.sh' provides an utility to perform a convergence test. 
You will have to set for your system the content of 'runs/directories.sh'. To change the data of the simulation you can modify 'data.sh'.


