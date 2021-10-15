cmake_minimum_required(VERSION 3.16.0)
project(CBLAS_STRASSEN LANGUAGES C)


# Build options
option(BUILD_SAMPLES "Build sample applications." OFF)

# Add CMake dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Compiler optimization
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message(STATUS "Compiler optimization enabled.")
    if(WIN32)
        add_compile_options(/O2)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(BUILD_SHARED_LIBS TRUE)
    else()
        add_compile_options(-O2)
    endif()
endif()

# Load MKL
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
        link_directories(${MKL_LIBRARIES})
include_directories(${CMAKE_BUILD_DIR}/include)

# Create shared and static libraries
add_library(cblas_strassen SHARED   src/cblas_sstrassen.c
                                    src/cblas_dstrassen.c
                                    src/cblas_cstrassen.c
                                    src/cblas_zstrassen.c)
add_library(cblas_strassen_static STATIC    src/cblas_sstrassen.c
                                            src/cblas_dstrassen.c
                                            src/cblas_cstrassen.c
                                            src/cblas_zstrassen.c)
target_link_libraries(cblas_strassen ${MKL_LIBRARIES})
target_link_libraries(cblas_strassen_static ${MKL_LIBRARIES})

# If required, build the samples
if (BUILD_SAMPLES)
    add_executable(TestSStrassen test/test_sstrassen.c)
    add_executable(TestDStrassen test/test_dstrassen.c)
    add_executable(TestCStrassen test/test_cstrassen.c)
    add_executable(TestZStrassen test/test_zstrassen.c)
    add_executable(TestTimes test/test_times.c)


    target_link_libraries(TestSStrassen cblas_strassen ${MKL_LIBRARIES})
    target_link_libraries(TestDStrassen cblas_strassen ${MKL_LIBRARIES})
    target_link_libraries(TestCStrassen cblas_strassen ${MKL_LIBRARIES})
    target_link_libraries(TestZStrassen cblas_strassen ${MKL_LIBRARIES})
    target_link_libraries(TestTimes cblas_strassen ${MKL_LIBRARIES})
endif()


# Install
install(FILES "${PROJECT_SOURCE_DIR}/include/cblas_strassen.h"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
install(TARGETS cblas_strassen cblas_strassen_static
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")