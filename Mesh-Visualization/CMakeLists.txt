
ADD_EXECUTABLE (visualization visualization.cpp)
TARGET_LINK_LIBRARIES (visualization mesh plot)
TARGET_INCLUDE_DIRECTORIES (visualization PUBLIC ${EXTERNAL_INCLUDES})
