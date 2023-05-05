include(third_party_llvm)
setup_llvm()

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(include)

add_subdirectory(include)
add_subdirectory(lib)

# MLIRLoxAll carries all the header/libs.
add_library(MLIRLoxAll INTERFACE)
target_include_directories(MLIRLoxAll INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(MLIRLoxAll INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(MLIRLoxAll INTERFACE MLIRLoxDialect LoxToMixedLox MixedLoxToLLVM)
