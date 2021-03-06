project(HArD::Core3D)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

#---------------------------------------------------------------------------------------------------------------------------------------#
#  Directories to be set manually: ENABLE THIS SECTION IF AUTOMATIC LIBRARY SEARCH FAILS (GetPot directory has to be set manually anyway)
#---------------------------------------------------------------------------------------------------------------------------------------#

#SET(GETPOT_INCLUDEDIR ...)
#SET(BOOST_ROOT ...)
SET(EIGEN_INCLUDEDIR ...) # it has to be set manually 

#------------------------------------------------------------------------------#

message (STATUS "***packages loaded***")
message (STATUS ${GETPOT_INCLUDEDIR})
#message (STATUS ${BOOST_INCLUDEDIR})
#message (STATUS ${BOOST_LIBRARYDIR})
message (STATUS ${EIGEN_INCLUDEDIR})

#------------------------------------------------------------------------------#
#                             Compiler setup
#------------------------------------------------------------------------------#

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  SET(ADDITIONAL_FLAGS "-DWIN32")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

SET(CMAKE_CXX_COMPILER "c++")
SET(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")
# IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
#   SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wno-maybe-uninitialized")
# ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
SET(CMAKE_CXX_FLAGS_DEBUGALL "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -pthread" )
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O3 -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )

#SET(CMAKE_BUILD_TYPE Debug)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

#------------------------------------------------------------------------------#
#                             Required libraries
#------------------------------------------------------------------------------#

SET(Boost_DEBUG "ON")
SET(Boost_USE_STATIC_LIBS "OFF")
SET(HOME_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
SET(CMAKE_MODULE_PATH ${HOME_DIR}/CMake)
SET(Boost_NO_SYSTEM_PATHS "ON")

FIND_PACKAGE(Boost COMPONENTS filesystem program_options timer chrono REQUIRED)
SET(EXTERNAL_INCLUDES ${Boost_INCLUDE_DIRS})
SET(EXTERNAL_LIBRARIES ${Boost_LIBRARIES})

FIND_PACKAGE(Eigen3 3.3.4)
SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${EIGEN_INCLUDEDIR})

FIND_PACKAGE(SuperLU)
IF(SUPERLU_FOUND)
  SET(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${SUPERLU_LIBRARIES})
  SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${SUPERLU_INCLUDES})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_SUPERLU")
ENDIF()

FIND_PACKAGE(Umfpack)
IF(UMFPACK_FOUND)
  SET(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${UMFPACK_LIBRARIES})
  SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${UMFPACK_INCLUDES})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_UMFPACK")
ENDIF()

FIND_PACKAGE(MKL)
IF(MKL_FOUND)
  SET(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${MKL_LIBRARIES})
  SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${MKL_INCLUDES})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MKL")
ENDIF()

SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${GETPOT_INCLUDEDIR})

INCLUDE_DIRECTORIES(${EXTERNAL_INCLUDES})
message(STATUS ${EXTERNAL_INCLUDES})
message(STATUS ${Boost_LIBRARIES})
#------------------------------------------------------------------------------#
#                             Directories for compiled libraries
#------------------------------------------------------------------------------#

INCLUDE_DIRECTORIES(src/Mesh)
INCLUDE_DIRECTORIES(src/Quadrature)
INCLUDE_DIRECTORIES(src/Common)
INCLUDE_DIRECTORIES(src/HybridCore)
INCLUDE_DIRECTORIES(src/Plot)
INCLUDE_DIRECTORIES(src/DDRCore)
#INCLUDE_DIRECTORIES(src/DDRCore-orth)

ADD_SUBDIRECTORY(src/Mesh)
ADD_SUBDIRECTORY(src/Quadrature)
ADD_SUBDIRECTORY(src/Common)
ADD_SUBDIRECTORY(src/HybridCore)
ADD_SUBDIRECTORY(src/Plot)
ADD_SUBDIRECTORY(src/DDRCore)
#ADD_SUBDIRECTORY(src/DDRCore-orth)
#------------------------------------------------------------------------------#
#                             Directories for methods
#------------------------------------------------------------------------------#

INCLUDE_DIRECTORIES(Schemes)
ADD_SUBDIRECTORY(Schemes)

#------------------------------------------------------------------------------#
#                             Directories for  tests                           #
#------------------------------------------------------------------------------#

INCLUDE_DIRECTORIES(spadotto-magnetostatics)
ADD_SUBDIRECTORY(spadotto-magnetostatics)

INCLUDE_DIRECTORIES(Mesh-Visualization)
ADD_SUBDIRECTORY(Mesh-Visualization)
