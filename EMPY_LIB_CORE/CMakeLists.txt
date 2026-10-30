project(${EMPY_LIB_CORE})

# gather source files
file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${sources} ${headers})
configure_project_target(${PROJECT_NAME})

# define api macros
target_compile_definitions(${PROJECT_NAME} PRIVATE
    EMPY_CORE_SHARED
)

# include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# link window libraries (sockets, certificate stores and CNG key storage)
if (WIN32)
    target_link_libraries(${PROJECT_NAME} PUBLIC
        ws2_32 mswsock)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        crypt32 ncrypt)
endif()

# link others libraries: asio and OpenSSL stay private, public headers never expose them
target_link_libraries(${PROJECT_NAME} PUBLIC
    Threads::Threads
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    asio::asio OpenSSL::SSL OpenSSL::Crypto
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ASIO_STANDALONE ASIO_NO_DEPRECATED
)

# optional QUIC transport
if (EMPY_QUIC_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE empy::msquic)
    target_compile_definitions(${PROJECT_NAME} PUBLIC EMPY_WITH_QUIC=1)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC EMPY_WITH_QUIC=0)
endif()
