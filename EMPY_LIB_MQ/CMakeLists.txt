project(${EMPY_LIB_MQ})

# gather source files (vendored C sources are compiled as C)
file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE vendors CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Vendors/*.c)
add_library(${PROJECT_NAME} SHARED ${sources} ${vendors} ${headers})
configure_project_target(${PROJECT_NAME})

# define api macros
target_compile_definitions(${PROJECT_NAME} PRIVATE
    EMPY_MQ_SHARED
)

# include directories: implementation-only vendors stay private
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Vendors/Ascon
)

# libraries directories
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${EMPY_LIB_CORE}
)
