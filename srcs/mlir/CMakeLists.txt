include(third_party_llvm)
setup_llvm()

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# MLIRLoxAll carries all the header/libs.
add_library(MLIRLoxAll INTERFACE)
target_include_directories(MLIRLoxAll INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(MLIRLoxAll INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(MLIRLoxAll INTERFACE MLIRLoxDialect LoxToMixedLox MixedLoxToLLVM)


if (LLVM_EXTERNAL_LIT)
  configure_file(lit.site.cfg.in lit.site.cfg @ONLY)
  add_custom_target(lit_all
      COMMAND ${LLVM_EXTERNAL_LIT} ${CMAKE_CURRENT_BINARY_DIR} -v
      DEPENDS lox-mlir-opt)
  add_test(run_lit_all "${CMAKE_COMMAND}" --build ${PROJECT_BINARY_DIR} --target lit_all)
endif ()
