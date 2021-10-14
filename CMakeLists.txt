cmake_minimum_required(VERSION 3.19.0)
project(CBLAS_STRASSEN LANGUAGES C)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(/O2)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})

include_directories(include)


add_library(cblas_strassen STATIC   src/cblas_sstrassen.c
                                    src/cblas_dstrassen.c
                                    src/cblas_cstrassen.c
                                    src/cblas_zstrassen.c)


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