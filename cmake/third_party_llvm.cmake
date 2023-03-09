# these option and definitions should only be used when building things related to MLIR
# we do not populate these options to global in the main CMakeLists.txt
macro(setup_llvm)
  cmake_policy(SET CMP0116 OLD)
  find_package(MLIR REQUIRED CONFIG)
  message(DEBUG "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(DEBUG "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})

  include(HandleLLVMOptions) # This will populate LLVM compilation/link flags
  add_definitions(${LLVM_DEFINITIONS})

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ambiguous-reversed-operator") # ignore errors from llvm headers
  endif ()
endmacro()