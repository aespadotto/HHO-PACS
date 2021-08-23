SET (PATH_TO_HHO_MAGN "../Schemes/HHO-magnetostatics")
SET (PATH_TO_SCHEMES "../Schemes")

# variant with sum for face spaces
ADD_LIBRARY (hho-magneto-evol OBJECT ${PATH_TO_HHO_MAGN}/hhospace.cpp
        ${PATH_TO_HHO_MAGN}/yspace.cpp
        xspace-sum-byparts-evol.cpp)


#######EXECUTABLES######



ADD_EXECUTABLE (hho-magnetostatics-spadotto $<TARGET_OBJECTS:hho-magneto-evol> $<TARGET_OBJECTS:BoundaryConditions> hho-magnetostatics-sum.cpp)
TARGET_LINK_LIBRARIES (hho-magnetostatics-spadotto basis mesh
                       quadrature ddrcore ${EXTERNAL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES (hho-magnetostatics-spadotto PUBLIC
                            ${PATH_TO_SCHEMES}/BoundaryConditions
                            ${PATH_TO_HHO_MAGN}
                            ${EXTERNAL_INCLUDES})
