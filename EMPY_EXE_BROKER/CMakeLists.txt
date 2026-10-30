project(${EMPY_EXE_BROKER})

# gather all source files: broker components are header-implemented under include/
# so that EmpyTests can exercise them in-process; src/ holds the entry point
file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
add_executable(${PROJECT_NAME} ${sources} ${headers})
configure_project_target(${PROJECT_NAME})

# includes directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# libraries directories
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${EMPY_LIB_CORE}
    asio::asio
)
empy_stage_runtime(${PROJECT_NAME})
