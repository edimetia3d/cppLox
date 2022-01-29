IF (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    message(STATUS "Using win32 compat")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/win32_compat/include)
endif ()