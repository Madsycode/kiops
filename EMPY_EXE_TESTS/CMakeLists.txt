project(${EMPY_EXE_TESTS})

# gather source files
file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# Agentic-MLOps suites (src/Ops) are built only with the extension
if (NOT EMPY_BUILD_MLOPS)
    list(FILTER sources EXCLUDE REGEX "/src/Ops/")
    list(FILTER headers EXCLUDE REGEX "/src/Ops/")
endif()
add_executable(${PROJECT_NAME} ${sources} ${headers})
configure_project_target(${PROJECT_NAME})

# the tests exercise header-implemented broker components in-process
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/EMPY_EXE_BROKER/include
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    EMPY_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

# OpenSSL is linked directly only to generate throw-away test certificates
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${EMPY_LIB_CORE}
    ${EMPY_LIB_MQ}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    asio::asio
)
add_dependencies(${PROJECT_NAME} ${EMPY_EXE_BROKER} ${EMPY_EXE_CLIENT})
if (WIN32)
    add_dependencies(${PROJECT_NAME} ${EMPY_EXE_BROKER}Runtime ${EMPY_EXE_CLIENT}Runtime)
endif()

# the MLOps suites link the extension and launch its executables beside EmpyTests
if (EMPY_BUILD_MLOPS)
    target_link_libraries(${PROJECT_NAME} PUBLIC
        ${EMPY_LIB_OPS}
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        EMPY_WITH_MLOPS=1
    )
    add_dependencies(${PROJECT_NAME} ${EMPY_EXE_HARNESS} ${EMPY_EXE_WORKLOAD})
    if (WIN32)
        add_dependencies(${PROJECT_NAME} ${EMPY_EXE_HARNESS}Runtime ${EMPY_EXE_WORKLOAD}Runtime)
    endif()
endif()
empy_stage_runtime(${PROJECT_NAME})

# register one CTest entry per suite declared with EMPY_TEST(Suite, Name)
set(suites "")
foreach (source ${sources})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${source})
    file(STRINGS ${source} declarations REGEX "^[ \t]*EMPY_TEST\\([A-Za-z0-9_]+,")
    foreach (declaration ${declarations})
        string(REGEX REPLACE "^[ \t]*EMPY_TEST\\(([A-Za-z0-9_]+),.*" "\\1" suite "${declaration}")
        list(APPEND suites ${suite})
    endforeach()
endforeach()
list(REMOVE_DUPLICATES suites)
foreach (suite ${suites})
    if (NOT EMPY_QUIC_ENABLED AND (suite STREQUAL "Quic" OR suite STREQUAL "FabricQuic"))
        continue()
    endif()
    add_test(NAME ${suite} COMMAND ${PROJECT_NAME} --suite ${suite}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    set_tests_properties(${suite} PROPERTIES TIMEOUT 300)
endforeach()
