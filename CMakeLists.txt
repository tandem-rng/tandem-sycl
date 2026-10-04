cmake_minimum_required(VERSION 3.25)
project(tandem_sycl VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

add_library(tandem_sycl INTERFACE)
add_library(tandem::sycl ALIAS tandem_sycl)
target_include_directories(tandem_sycl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/tandem-cuda/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(tandem_sycl INTERFACE cxx_std_20)
set_target_properties(tandem_sycl PROPERTIES EXPORT_NAME sycl)

include(CMakePackageConfigHelpers)
install(DIRECTORY include/tandem external/tandem-cuda/include/tandem
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS tandem_sycl EXPORT tandem-sycl-targets)
install(EXPORT tandem-sycl-targets NAMESPACE tandem::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tandem-sycl)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tandem-syclConfig.cmake
     "include(\${CMAKE_CURRENT_LIST_DIR}/tandem-sycl-targets.cmake)\n")
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tandem-syclConfigVersion.cmake
                                 COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tandem-syclConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/tandem-syclConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tandem-sycl)

option(TANDEM_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(TANDEM_BENCH "Build the benchmark" OFF)

# SYCL is a compiler mode, so the library target carries no SYCL flags. The tests build with
# DPC++ when it is the compiler and with AdaptiveCpp otherwise.
function(tandem_add_sycl target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # icpx defaults to fast math, whose float log and cos miss the normals' tolerance.
        target_compile_options(${target} PRIVATE -fsycl -ffp-model=precise)
        target_link_options(${target} PRIVATE -fsycl)
        # The SYCL runtime sits next to the compiler, which a pixi or conda environment does not
        # put on the loader path.
        get_filename_component(sycl_lib "${CMAKE_CXX_COMPILER}/../../lib" ABSOLUTE)
        target_link_options(${target} PRIVATE "-Wl,-rpath,${sycl_lib}")
    else()
        find_package(AdaptiveCpp CONFIG REQUIRED)
        add_sycl_to_target(TARGET ${target})
    endif()
endfunction()

if(TANDEM_TESTS)
    enable_testing()
    add_executable(test_tandem tests/test_tandem.cpp)
    target_link_libraries(test_tandem PRIVATE tandem::sycl)
    tandem_add_sycl(test_tandem)
    add_test(NAME tandem COMMAND test_tandem ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
endif()

if(TANDEM_BENCH)
    add_executable(bench tools/bench.cpp)
    target_link_libraries(bench PRIVATE tandem::sycl)
    tandem_add_sycl(bench)
endif()
