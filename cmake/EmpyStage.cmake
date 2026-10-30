# Stage the actual Windows import graph and explicitly enabled dynamic providers.
if (POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()
get_filename_component(output "${EMPY_EXECUTABLE}" DIRECTORY)
file(LOCK "${output}/.runtime.lock" TIMEOUT 30)
# Explicitly selected runtimes win over a stale DLL already present in bin/.
foreach(dependency IN LISTS EMPY_IMPORTED_DLLS EMPY_DYNAMIC_DLLS)
    get_filename_component(filename "${dependency}" NAME)
    get_filename_component(source "${dependency}" ABSOLUTE)
    get_filename_component(destination "${output}/${filename}" ABSOLUTE)
    if (NOT source STREQUAL destination)
        file(COPY_FILE "${source}" "${destination}" ONLY_IF_DIFFERENT)
    endif()
endforeach()
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "windows+pe")
if (EMPY_OBJDUMP)
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "objdump")
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "${EMPY_OBJDUMP}")
endif()
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${EMPY_EXECUTABLE}"
    LIBRARIES ${EMPY_DYNAMIC_DLLS}
    DIRECTORIES "${output}" ${EMPY_DLL_SEARCH}
    RESOLVED_DEPENDENCIES_VAR resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
    POST_INCLUDE_REGEXES ".*/[Vv][Cc][Rr][Uu][Nn][Tt][Ii][Mm][Ee].*\\.dll$" ".*/[Mm][Ss][Vv][Cc][Pp].*\\.dll$"
    POST_EXCLUDE_REGEXES ".*[/\\\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\\\].*" ".*[/\\\\][Ss][Yy][Ss][Ww][Oo][Ww]64[/\\\\].*")
if (unresolved)
    message(FATAL_ERROR "Missing runtime DLLs for ${EMPY_EXECUTABLE}: ${unresolved}. Set EMPY_RUNTIME_DIRECTORIES to their directories.")
endif()
get_filename_component(name "${EMPY_EXECUTABLE}" NAME)
set(manifest "Runtime imports and dynamic providers for ${name}; Windows OS DLLs are supplied by Windows.\n")
list(APPEND resolved ${EMPY_DYNAMIC_DLLS})
list(REMOVE_DUPLICATES resolved)
foreach(dependency IN LISTS resolved)
    get_filename_component(filename "${dependency}" NAME)
    get_filename_component(source "${dependency}" ABSOLUTE)
    get_filename_component(destination "${output}/${filename}" ABSOLUTE)
    if (NOT source STREQUAL destination)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${source}" "${destination}"
            RESULT_VARIABLE copied)
        if (NOT copied EQUAL 0)
            message(FATAL_ERROR "Could not stage ${source}")
        endif()
    endif()
    file(SHA256 "${destination}" hash)
    string(APPEND manifest "${hash}  ${filename}\n")
endforeach()
file(WRITE "${output}/${name}.runtime.txt" "${manifest}")
