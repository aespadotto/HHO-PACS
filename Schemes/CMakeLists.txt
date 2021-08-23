# Libraries
ADD_LIBRARY(TestCase OBJECT TestCase/TestCase.cpp)
ADD_LIBRARY(BoundaryConditions OBJECT BoundaryConditions/BoundaryConditions.cpp)
ADD_LIBRARY(hho-general OBJECT HHO-general/HHO3D.cpp)
ADD_LIBRARY(hho-magneto OBJECT HHO-magnetostatics/hhospace.cpp HHO-magnetostatics/yspace.cpp HHO-magnetostatics/xspace.cpp)
ADD_LIBRARY(hho-magneto-alt OBJECT HHO-magnetostatics/hhospace.cpp HHO-magnetostatics/yspace.cpp HHO-magnetostatics/xspace-alternative-faceBasis.cpp)
#ADD_LIBRARY(ddr-stokes-library OBJECT DDR-stokes/ddr-stokes.cpp)

# HHO Diffusion
ADD_EXECUTABLE(hho-diffusion $<TARGET_OBJECTS:TestCase> $<TARGET_OBJECTS:BoundaryConditions> HHO-diffusion/HHO_Diffusion.cpp)
TARGET_LINK_LIBRARIES(hho-diffusion mesh quadrature hybridcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(hho-diffusion PUBLIC ${EIGEN3_INCLUDE_DIR})

# HHO Locally variable diffusion
ADD_EXECUTABLE(hho-locvardiff $<TARGET_OBJECTS:TestCase> $<TARGET_OBJECTS:BoundaryConditions> HHO-locvardiff/HHO_LocVarDiff.cpp)
TARGET_LINK_LIBRARIES(hho-locvardiff mesh quadrature hybridcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(hho-locvardiff PUBLIC ${EIGEN3_INCLUDE_DIR})

# HHO Diffusion Advection Reaction
ADD_EXECUTABLE(hho-diff-advec-reac $<TARGET_OBJECTS:hho-general> $<TARGET_OBJECTS:TestCase>  $<TARGET_OBJECTS:BoundaryConditions> HHO-diff-advec-reac/HHO_DiffAdvecReac.cpp)
TARGET_LINK_LIBRARIES(hho-diff-advec-reac mesh quadrature hybridcore plot ${Boost_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(hho-diff-advec-reac PUBLIC ${EIGEN3_INCLUDE_DIR})

# HHO magnetostatics
ADD_EXECUTABLE(hho-magnetostatics $<TARGET_OBJECTS:hho-magneto>  $<TARGET_OBJECTS:BoundaryConditions> HHO-magnetostatics/hho-magnetostatics.cpp)
TARGET_LINK_LIBRARIES(hho-magnetostatics mesh quadrature ddrcore plot ${Boost_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(hho-magnetostatics PUBLIC ${EIGEN3_INCLUDE_DIR})

#HHO magnetostatics alternative
ADD_EXECUTABLE(hho-magnetostatics-alt $<TARGET_OBJECTS:hho-magneto-alt>  $<TARGET_OBJECTS:BoundaryConditions> HHO-magnetostatics/hho-magnetostatics-alt.cpp)
TARGET_LINK_LIBRARIES(hho-magnetostatics-alt mesh quadrature ddrcore plot ${Boost_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(hho-magnetostatics-alt PUBLIC ${EIGEN3_INCLUDE_DIR})

# DDR magnetostatics
ADD_EXECUTABLE(ddr-magnetostatics DDR-magnetostatics/ddr-magnetostatics.cpp)
TARGET_LINK_LIBRARIES(ddr-magnetostatics mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(ddr-magnetostatics PUBLIC ${EXTERNAL_INCLUDES})

# DDR Stokes
ADD_EXECUTABLE(ddr-stokes DDR-stokes/ddr-stokes.cpp)
TARGET_LINK_LIBRARIES(ddr-stokes mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(ddr-stokes PUBLIC ${EXTERNAL_INCLUDES})

#### Tests

# Basis test
ADD_EXECUTABLE(basis-test DDRCore/basis-test.cpp)
TARGET_LINK_LIBRARIES(basis-test mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(basis-test PUBLIC ${EIGEN3_INCLUDE_DIR})

# DDRCore test
ADD_EXECUTABLE(ddrcore-test DDRCore/ddrcore-test.cpp)
TARGET_LINK_LIBRARIES(ddrcore-test mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(ddrcore-test PUBLIC ${EIGEN3_INCLUDE_DIR})

# XGrad test
ADD_EXECUTABLE(xgrad-test DDRCore/xgrad-test.cpp)
TARGET_LINK_LIBRARIES(xgrad-test mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(xgrad-test PUBLIC ${EIGEN3_INCLUDE_DIR})

# XCurl test
ADD_EXECUTABLE(xcurl-test DDRCore/xcurl-test.cpp)
TARGET_LINK_LIBRARIES(xcurl-test mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(xcurl-test PUBLIC ${EIGEN3_INCLUDE_DIR})

# XDiv test
ADD_EXECUTABLE(xdiv-test DDRCore/xdiv-test.cpp)
TARGET_LINK_LIBRARIES(xdiv-test mesh quadrature ddrcore plot ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(xdiv-test PUBLIC ${EIGEN3_INCLUDE_DIR})

# Cartesian product HHO space test
ADD_EXECUTABLE(cartesianproducthhospace-test $<TARGET_OBJECTS:hho-magneto>  $<TARGET_OBJECTS:BoundaryConditions> HHO-magnetostatics/cartesianproducthhospace-test.cpp)
TARGET_LINK_LIBRARIES(cartesianproducthhospace-test mesh quadrature ddrcore plot ${Boost_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(cartesianproducthhospace-test PUBLIC ${EIGEN3_INCLUDE_DIR})
