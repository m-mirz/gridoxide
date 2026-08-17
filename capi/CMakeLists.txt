# CMake integration for gridoxide's C ABI.
#
# Two ways in, because a C++ project embedding gridoxide is in one of two
# situations and they want opposite things:
#
#   -DGRIDOXIDE_FROM_SOURCE=ON   Build the library with cargo as part of this
#                                build. Always in step with the source; needs a
#                                Rust toolchain on every developer's machine.
#
#   -DGRIDOXIDE_LIBRARY=<path>   Use a prebuilt .a/.so, with
#   -DGRIDOXIDE_INCLUDE_DIR=<d>  headers from <d>. No Rust toolchain anywhere.
#
# Either way the result is one imported target, `gridoxide::gridoxide`, so the
# consuming CMakeLists does not care which was used:
#
#   target_link_libraries(my_app PRIVATE gridoxide::gridoxide)

cmake_minimum_required(VERSION 3.16)
project(gridoxide_capi LANGUAGES C CXX)

option(GRIDOXIDE_FROM_SOURCE "Build libgridoxide with cargo as part of this build" OFF)
option(GRIDOXIDE_BUILD_EXAMPLES "Build the C and C++ example programs" ON)
set(GRIDOXIDE_LIBRARY "" CACHE FILEPATH "Path to a prebuilt libgridoxide.a or .so")
set(GRIDOXIDE_INCLUDE_DIR "" CACHE PATH "Directory holding gridoxide.h and gridoxide.hpp")
set(GRIDOXIDE_CARGO_PROFILE "release" CACHE STRING "cargo profile when building from source")

get_filename_component(GRIDOXIDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(GRIDOXIDE_FROM_SOURCE)
    find_program(CARGO_EXECUTABLE cargo REQUIRED)

    if(GRIDOXIDE_CARGO_PROFILE STREQUAL "release")
        set(_cargo_flags --release)
        set(_cargo_outdir "${GRIDOXIDE_ROOT}/target/release")
    else()
        set(_cargo_flags "")
        set(_cargo_outdir "${GRIDOXIDE_ROOT}/target/debug")
    endif()

    set(_staticlib "${_cargo_outdir}/libgridoxide.a")

    # A custom command rather than Corrosion: this needs one crate, one
    # feature and one artifact, and a dependency that must itself be fetched is
    # a poor trade for that. Corrosion is the better answer for a workspace of
    # several crates with per-target configuration.
    #
    # `--features capi` also regenerates include/gridoxide.h as a side effect,
    # which is why the header is listed as a byproduct.
    add_custom_command(
        OUTPUT "${_staticlib}"
        COMMAND ${CARGO_EXECUTABLE} build ${_cargo_flags} --features capi
        WORKING_DIRECTORY "${GRIDOXIDE_ROOT}"
        BYPRODUCTS "${GRIDOXIDE_ROOT}/include/gridoxide.h"
        COMMENT "Building libgridoxide with cargo (--features capi)"
        VERBATIM
        USES_TERMINAL
    )
    add_custom_target(gridoxide_cargo DEPENDS "${_staticlib}")

    add_library(gridoxide_imported STATIC IMPORTED GLOBAL)
    set_target_properties(gridoxide_imported PROPERTIES IMPORTED_LOCATION "${_staticlib}")
    add_dependencies(gridoxide_imported gridoxide_cargo)
    set(_gridoxide_include "${GRIDOXIDE_ROOT}/include")

else()
    if(NOT GRIDOXIDE_LIBRARY)
        message(FATAL_ERROR
            "Set -DGRIDOXIDE_LIBRARY=<path to libgridoxide.a|.so> and "
            "-DGRIDOXIDE_INCLUDE_DIR=<dir with gridoxide.h>, or configure with "
            "-DGRIDOXIDE_FROM_SOURCE=ON to build it here with cargo.")
    endif()
    if(NOT EXISTS "${GRIDOXIDE_LIBRARY}")
        message(FATAL_ERROR "GRIDOXIDE_LIBRARY does not exist: ${GRIDOXIDE_LIBRARY}")
    endif()

    if(GRIDOXIDE_INCLUDE_DIR)
        set(_gridoxide_include "${GRIDOXIDE_INCLUDE_DIR}")
    else()
        set(_gridoxide_include "${GRIDOXIDE_ROOT}/include")
    endif()
    if(NOT EXISTS "${_gridoxide_include}/gridoxide.h")
        message(FATAL_ERROR "gridoxide.h not found under ${_gridoxide_include}")
    endif()

    get_filename_component(_ext "${GRIDOXIDE_LIBRARY}" EXT)
    if(_ext STREQUAL ".a")
        add_library(gridoxide_imported STATIC IMPORTED GLOBAL)
    else()
        add_library(gridoxide_imported SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(gridoxide_imported PROPERTIES IMPORTED_LOCATION "${GRIDOXIDE_LIBRARY}")
endif()

# The public target. An INTERFACE library wrapping the imported one, so the
# include directory and the platform link dependencies travel with it.
add_library(gridoxide INTERFACE)
add_library(gridoxide::gridoxide ALIAS gridoxide)
target_include_directories(gridoxide INTERFACE "${_gridoxide_include}")
target_link_libraries(gridoxide INTERFACE gridoxide_imported)
target_compile_features(gridoxide INTERFACE cxx_std_17)

# A Rust staticlib does not carry its own runtime dependencies, so a consumer
# linking one must name them. Getting this wrong shows up as pages of undefined
# symbols from libstd, which is a confusing first experience.
if(UNIX AND NOT APPLE)
    target_link_libraries(gridoxide INTERFACE pthread dl m)
elseif(APPLE)
    target_link_libraries(gridoxide INTERFACE m)
    set_property(TARGET gridoxide APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "-framework;CoreFoundation;-framework;Security")
elseif(WIN32)
    target_link_libraries(gridoxide INTERFACE ws2_32 userenv bcrypt ntdll)
endif()

if(GRIDOXIDE_BUILD_EXAMPLES)
    add_executable(powerflow_c examples/powerflow.c)
    target_link_libraries(powerflow_c PRIVATE gridoxide::gridoxide)
    set_target_properties(powerflow_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_compile_options(powerflow_c PRIVATE
        $<$<C_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror>)

    add_executable(powerflow_cpp examples/powerflow.cpp)
    target_link_libraries(powerflow_cpp PRIVATE gridoxide::gridoxide)
    set_target_properties(powerflow_cpp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_compile_options(powerflow_cpp PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror>)
endif()

# Installation, so a consumer can `find_package(gridoxide)`.
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(FILES "${_gridoxide_include}/gridoxide.h" "${_gridoxide_include}/gridoxide.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# The library itself. Easy to leave out — the headers and the config file are
# what one thinks of as "the package" — and then `find_package` configures
# happily and fails at `find_library`, which is a confusing way to learn that
# an install was incomplete.
get_target_property(_gridoxide_artifact gridoxide_imported IMPORTED_LOCATION)
install(FILES "${_gridoxide_artifact}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# `PATH_VARS` is not optional decoration: without it the `@PACKAGE_*@`
# placeholders in the template expand to **empty strings**, and the installed
# config then fails at `set_and_check` with "File or directory referenced by
# variable GRIDOXIDE_INCLUDE_DIR does not exist". Naming the variables here is
# what lets `configure_package_config_file` rewrite them as paths relative to
# wherever the package ends up installed.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/gridoxideConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gridoxideConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gridoxide"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gridoxideConfigVersion.cmake"
    VERSION 0.0.2
    COMPATIBILITY SameMinorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gridoxideConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gridoxideConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gridoxide")
