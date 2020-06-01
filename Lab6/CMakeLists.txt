cmake_minimum_required(VERSION 3.8)

project(Lab6)

if (WIN32)
	set (OpenCV_DIR $ENV{OpenCV})
endif () 

find_package(OpenCV REQUIRED)

include_directories(
	include
	${OpenCV_INCLUDE_DIRS}
	"include/trackinImages.h"
)

add_executable(
	${PROJECT_NAME}
	"source/HW6.cpp"
	"source/trackingImages.cpp"
)

target_link_libraries(
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

#copy program data to the build directory
if (WIN32 AND MSVC)
	file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}" NATIVE_BIN_DIR)
	file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Data" NATIVE_DATA_DIR)
	add_custom_command(
		TARGET "${PROJECT_NAME}" POST_BUILD
		COMMAND xcopy "${NATIVE_DATA_DIR}" "${NATIVE_BIN_DIR}\\Data" /I /C /K /E /R /Y 
	)
endif (WIN32 AND MSVC)