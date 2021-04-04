# the commit's SHA1, and whether the building workspace was dirty or not
find_package(Git)
if (Git_FOUND)
    message("Git found: ${GIT_EXECUTABLE}")
    execute_process(COMMAND
            "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_SHA1
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    # the date of the commit
    execute_process(COMMAND
            "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_DATE
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
else ()
    set(GIT_SHA1 GIT_NOTFOUND)
    set(GIT_DATE GIT_NOTFOUND)
endif ()

# generate version.cc
configure_file("${CMAKE_CURRENT_LIST_DIR}/version.cc.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cc" @ONLY)

file(GLOB VERSION_INFO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/version.cc")
add_library(version_info OBJECT ${VERSION_INFO_SRCS})