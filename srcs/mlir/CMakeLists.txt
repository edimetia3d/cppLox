
set_mlir_build_options()

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(include)

add_subdirectory(include)

add_subdirectory(lib)

target_include_directories(MLIRLox PUBLIC include)
target_include_directories(MLIRLox PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
