project(${EMPY_LIB_OPS})

# gather source files: one directory per concern under include/eOps and src
file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE privates CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
add_library(${PROJECT_NAME} SHARED ${sources} ${headers} ${privates})
configure_project_target(${PROJECT_NAME})

# define api macros
target_compile_definitions(${PROJECT_NAME} PRIVATE
    EMPY_OPS_SHARED
)

# include directories: implementation helpers stay private
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# libraries: asio and OpenSSL (HTTP/TLS clients) stay private, public headers never expose them
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${EMPY_LIB_MQ}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    asio::asio OpenSSL::SSL OpenSSL::Crypto
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ASIO_STANDALONE ASIO_NO_DEPRECATED
)

# windows certificate store access for HTTPS trust
if (WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        crypt32)
endif()
