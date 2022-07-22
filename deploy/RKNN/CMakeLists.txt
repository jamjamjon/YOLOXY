cmake_minimum_required(VERSION 3.1.1)
project(YOLO_RKNN)

# COMPILER
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# RKNN
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/rknn_api/include)

# SOURCE INCLUDE
include_directories(${CMAKE_SOURCE_DIR}/src)

# SOURCE CODE DIR
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)


# openmp
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
	message("==> find openmp")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS}")

endif()


# RKNN & OPENCV 
set(link_libs  	${CMAKE_SOURCE_DIR}/include/rknn_api/lib64/librknn_api.so
				# ${CMAKE_SOURCE_DIR}/rknn_api/lib64/librknn_runtime.so
				/usr/local/lib/libopencv_highgui.so
				/usr/local/lib/libopencv_core.so
				/usr/local/lib/libopencv_imgproc.so
				/usr/local/lib/libopencv_imgcodecs.so
				/usr/local/lib/libopencv_videoio.so
				pthread
)


add_executable(main 	${SRC_DIR}/yolo.cpp
						${SRC_DIR}/main.cpp 
						${SRC_DIR}/STrack.cpp
						${SRC_DIR}/utils.cpp
						${SRC_DIR}/lapjv.cpp
						${SRC_DIR}/BYTETracker.cpp
						${SRC_DIR}/kalmanFilter.cpp

			)
target_link_libraries(main ${link_libs} dl)

