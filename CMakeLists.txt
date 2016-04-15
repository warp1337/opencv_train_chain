CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(opencv_train_chain)

FIND_PACKAGE(CUDA)

FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options)

SET(OpenCV_MIN_VERSION "3.1.0")

FIND_PATH(OpenCV_DIR NAMES opencv-config.cmake OpenCVConfig.cmake OpenCVModules.cmake PATH_SUFFIXES lib/cmake lib)

FIND_PACKAGE( OpenCV ${OpenCV_MIN_VERSION} REQUIRED COMPONENTS core features2d imgproc objdetect video highgui flann
              cudafeatures2d cudev cudaarithm cudabgsegm cudafilters cudaimgproc cudawarping cudacodec cudalegacy
              cudaobjdetect cudaoptflow cudastereo)
IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        MESSAGE("Found OpenCV ${OpenCV_VERSION} libraries ${OpenCV_LIBRARIES} and headers ${OpenCV_INCLUDE_DIRS}")
ELSE()
        MESSAGE(FATAL_ERROR "OpenCV not found.")
ENDIF()

SET(CMAKE_BUILD_TYPE RELEASE)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} "/usr/local/cuda/include")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -march=core-avx-i -O2")

FILE(GLOB ORB_HEADER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/*.h src/*.h)

ADD_EXECUTABLE(otc-opencv-orb opencv/src/opencv-orb.cpp ${ORB_HEADER_LIST})
ADD_EXECUTABLE(cuda-test opencv/src/cuda_test.cpp ${ORB_HEADER_LIST})

TARGET_LINK_LIBRARIES(otc-opencv-orb ${OpenCV_LIBRARIES} )
TARGET_LINK_LIBRARIES(cuda-test ${OpenCV_LIBRARIES})

SET_PROPERTY(TARGET otc-opencv-orb PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
SET_TARGET_PROPERTIES(otc-opencv-orb PROPERTIES COMPILE_FLAGS -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

INSTALL(TARGETS otc-opencv-orb RUNTIME DESTINATION bin)
INSTALL(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/training/otc-collect-data" DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
