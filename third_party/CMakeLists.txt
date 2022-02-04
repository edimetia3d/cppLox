if (NOT DEFINED CLI11_DIR)
    set(CLI11_DIR "${CMAKE_CURRENT_LIST_DIR}/CLI11" CACHE STRING "Bundled CLI11 repo")
endif ()
if (NOT DEFINED SPDLOG_DIR)
    set(SPDLOG_DIR "${CMAKE_CURRENT_LIST_DIR}/spdlog" CACHE STRING "Bundled spdlog repo")
endif ()

if (ENABLE_MLIR_JIT_BACKEND)
    message(STATUS "JIT Backend is enabled")
    add_custom_target(llvm_mlir
            COMMAND bash prepare_mlir.sh "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
else ()
    message(STATUS "JIT Backend is disabled")
endif ()
