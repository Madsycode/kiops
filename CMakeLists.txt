cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# -----------------------------------------
# WORKSPACE CMAKELIST.TXT FILE
# -----------------------------------------

project(Empy VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# only Debug and Release are supported configurations
set(EMPY_CONFIGURATIONS Debug Release)
if (CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES ${EMPY_CONFIGURATIONS} CACHE STRING "Build configurations" FORCE)
elseif (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build configuration" FORCE)
elseif (NOT CMAKE_BUILD_TYPE IN_LIST EMPY_CONFIGURATIONS)
    message(FATAL_ERROR "Unsupported CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}' (expected Debug or Release)")
endif()

# build options
option(EMPY_BUILD_TESTS "Build the EmpyTests executable and register CTest suites" ON)
option(EMPY_WITH_QUIC "Enable the MsQuic transport binding when MsQuic is available" ON)
option(EMPY_BUILD_MLOPS "Build the Agentic-MLOps library and executables" ON)
set(EMPY_RUNTIME_DIRECTORIES "" CACHE STRING "Additional runtime dependency directories")
set(EMPY_RUNTIME_LIBRARIES "" CACHE STRING "Additional dynamically loaded providers to stage")
set(EMPY_MSQUIC_ROOT "$ENV{MSQUIC_ROOT}" CACHE PATH "MsQuic installation prefix (include/, lib/, bin/)")
if (NOT DEFINED EMPY_MSQUIC_RUNTIME AND NOT "$ENV{EMPY_MSQUIC_RUNTIME}" STREQUAL "")
    set(EMPY_MSQUIC_RUNTIME "$ENV{EMPY_MSQUIC_RUNTIME}" CACHE FILEPATH "Selected MsQuic runtime provider")
endif()

# helper scripts used by the target functions below
set(EMPY_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR}/cmake)

# define libs and execs names
set(EMPY_EXE_BROKER EmpyBroker)
set(EMPY_EXE_CLIENT EmpyClient)
set(EMPY_EXE_TESTS EmpyTests)
set(EMPY_LIB_CORE EmpyCore)
set(EMPY_LIB_MQ EmpyMQ)
set(EMPY_LIB_OPS EmpyOps)
set(EMPY_EXE_HARNESS EmpyHarness)
set(EMPY_EXE_WORKLOAD EmpyWorkload)

# every target writes binaries to one bin/ folder per configuration
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if (UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# third-party dependencies (provided by Conan)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(asio REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)

# optional MsQuic (not packaged by Conan for MinGW): import the prebuilt C API
set(EMPY_QUIC_ENABLED OFF)
if (EMPY_WITH_QUIC)
    find_path(EMPY_MSQUIC_INCLUDE_DIR msquic.h
        HINTS ${EMPY_MSQUIC_ROOT} PATH_SUFFIXES include)
    find_library(EMPY_MSQUIC_LIBRARY NAMES msquic
        HINTS ${EMPY_MSQUIC_ROOT} PATH_SUFFIXES lib)
    find_file(EMPY_MSQUIC_RUNTIME NAMES msquic.dll libmsquic.so libmsquic.so.2
        HINTS ${EMPY_MSQUIC_ROOT} PATH_SUFFIXES bin lib)
    if (EMPY_MSQUIC_INCLUDE_DIR AND EMPY_MSQUIC_LIBRARY AND (NOT WIN32 OR EMPY_MSQUIC_RUNTIME))
        add_library(empy::msquic SHARED IMPORTED)
        set_target_properties(empy::msquic PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EMPY_MSQUIC_INCLUDE_DIR}")
        if (WIN32)
            set_target_properties(empy::msquic PROPERTIES
                IMPORTED_IMPLIB "${EMPY_MSQUIC_LIBRARY}"
                IMPORTED_LOCATION "${EMPY_MSQUIC_RUNTIME}")
        else()
            set_target_properties(empy::msquic PROPERTIES
                IMPORTED_LOCATION "${EMPY_MSQUIC_LIBRARY}")
        endif()
        set(EMPY_QUIC_ENABLED ON)
        message(STATUS "Empy: MsQuic transport enabled (${EMPY_MSQUIC_LIBRARY})")
    else()
        message(STATUS "Empy: MsQuic not found; QUIC transport disabled (set EMPY_MSQUIC_ROOT)")
    endif()
endif()

# -----------------------------------------
# shared target configuration
# -----------------------------------------

# apply the common compiler options, definitions and conventions to a target
function(configure_project_target target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    # libraries keep their documented names (EmpyCore.dll, not libEmpyCore.dll)
    get_target_property(target_type ${target} TYPE)
    if (target_type STREQUAL "SHARED_LIBRARY")
        set_target_properties(${target} PROPERTIES PREFIX "" IMPORT_PREFIX "")
    endif()
    target_compile_definitions(${target} PRIVATE
        EMPY_ENABLE_LOGGING
        $<$<CONFIG:Debug>:EMPY_DEBUG_BUILD>
        $<$<CONFIG:Release>:NDEBUG>
    )
    if (WIN32)
        # Windows 10 API level: required by asio overlapped I/O and MsQuic
        target_compile_definitions(${target} PRIVATE
            _WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if (MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # Header-implemented broker templates exceed COFF's normal section limit in Debug.
            target_compile_options(${target} PRIVATE -Wa,-mbig-obj)
        endif()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wno-unused-parameter
            $<$<CONFIG:Debug>:-O0 -g3 -fno-omit-frame-pointer>
            $<$<CONFIG:Release>:-O2>
        )
        # cheap standard-library precondition checks for development builds
        target_compile_definitions(${target} PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
    elseif (MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /bigobj
            $<$<CONFIG:Release>:/O2>)
    endif()
endfunction()

# stage non-system runtime dependencies beside an executable and record them in a manifest
function(empy_stage_runtime target)
    if (NOT WIN32)
        return()
    endif()
    get_filename_component(compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(search "${compiler_bin}" "${OPENSSL_ROOT_DIR}/bin" ${EMPY_RUNTIME_DIRECTORIES})
    foreach(library IN ITEMS "${OPENSSL_CRYPTO_LIBRARY}" "${OPENSSL_SSL_LIBRARY}" "${EMPY_MSQUIC_LIBRARY}" "${EMPY_MSQUIC_RUNTIME}")
        if (library)
            get_filename_component(directory "${library}" DIRECTORY)
            list(APPEND search "${directory}" "${directory}/../bin" "${directory}/../../bin")
        endif()
    endforeach()
    get_cmake_property(variables VARIABLES)
    foreach(variable IN LISTS variables)
        if (variable MATCHES "(_BIN_DIRS|_BINDIRS)")
            list(APPEND search ${${variable}})
        endif()
    endforeach()
    list(APPEND search "${EMPY_MSQUIC_ROOT}/bin")
    list(REMOVE_DUPLICATES search)
    # OpenSSL loads compression providers through DSO_load, outside the PE import graph.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${OPENSSL_INCLUDE_DIR}")
    set(dynamic ${EMPY_RUNTIME_LIBRARIES})
    if (EMPY_QUIC_ENABLED)
        list(APPEND dynamic "${EMPY_MSQUIC_RUNTIME}")
    endif()
    foreach(algorithm IN ITEMS ZLIB BROTLI ZSTD)
        check_cxx_source_compiles("#include <openssl/configuration.h>
#if defined(OPENSSL_NO_COMP) || defined(OPENSSL_NO_${algorithm}) || defined(OPENSSL_NO_${algorithm}_DYNAMIC)
#error Compression provider is not loaded dynamically
#endif
int main() { return 0; }" EMPY_OPENSSL_DYNAMIC_${algorithm})
        if (EMPY_OPENSSL_DYNAMIC_${algorithm})
            if (algorithm STREQUAL "ZLIB")
                set(components ZLIB)
                set(ZLIB_names zlib1.dll ZLIB1.dll)
            elseif(algorithm STREQUAL "BROTLI")
                set(components BROTLI_DEC BROTLI_ENC)
                set(BROTLI_DEC_names brotlidec.dll libbrotlidec.dll)
                set(BROTLI_ENC_names brotlienc.dll libbrotlienc.dll)
            else()
                set(components ZSTD)
                set(ZSTD_names libzstd.dll zstd.dll)
            endif()
            foreach(component IN LISTS components)
                find_file(EMPY_${component}_RUNTIME NAMES ${${component}_names} HINTS ${search} REQUIRED)
                list(APPEND dynamic "${EMPY_${component}_RUNTIME}")
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES dynamic)
    set(configuration "${CMAKE_BINARY_DIR}/Runtime/${target}-$<CONFIG>.cmake")
    file(GENERATE OUTPUT "${configuration}" CONTENT
        "set(EMPY_EXECUTABLE [==[$<TARGET_FILE:${target}>]==])
        set(EMPY_DLL_SEARCH [==[${search}]==])
        set(EMPY_DYNAMIC_DLLS [==[${dynamic}]==])
        set(EMPY_IMPORTED_DLLS [==[$<TARGET_RUNTIME_DLLS:${target}>]==])
        set(EMPY_OBJDUMP [==[${CMAKE_OBJDUMP}]==])
        include([==[${EMPY_CMAKE_DIR}/EmpyStage.cmake]==])")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -P "${configuration}" VERBATIM)
    add_custom_target(${target}Runtime ALL
        COMMAND "${CMAKE_COMMAND}" -P "${configuration}"
        DEPENDS ${target} VERBATIM)
endfunction()

# add project subdirectories (libraries first: executables depend on them)
add_subdirectory(EMPY_LIB_CORE)
add_subdirectory(EMPY_LIB_MQ)
add_subdirectory(EMPY_EXE_BROKER)
add_subdirectory(EMPY_EXE_CLIENT)

# optional Agentic-MLOps extension (library first: its executables depend on it)
if (EMPY_BUILD_MLOPS)
    add_subdirectory(EMPY_LIB_OPS)
    add_subdirectory(EMPY_EXE_HARNESS)
    add_subdirectory(EMPY_EXE_WORKLOAD)
endif()

# enable tests
if (EMPY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(EMPY_EXE_TESTS)
endif()
