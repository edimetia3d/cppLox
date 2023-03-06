# Provide bazel like cmake function that mangling target name to avoid name conflict
# 1. target in the `src` whose path is `srcs/a/b/c` will be named as `a::b::c` and `a.b.c`
# 2. target out of `srcs` whose path is `a/b/c` will be named as `a::b::c` and `a.b.c`
# 3. target whose name is same as the folder name will be named as shorter name, e.g. `srcs/a/b/c/c` will be named as `a::b::c` and `a.b.c`
# 4. `::` separated name is used for alias target, `.` separated name is used for real target
# 5. target's output name will leave unchanged

function(_create_valid_raw_name_and_alias name)
  # get relative path
  string(REPLACE "${PROJECT_SOURCE_DIR}/srcs/" "" relative_path "${CMAKE_CURRENT_LIST_DIR}")
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" relative_path "${relative_path}")

  # generate alias name
  string(REPLACE "/" "::" alias_base "${relative_path}")
  string(REGEX MATCH ".*${name}$" tail_same "${alias_base}")

  if (tail_same)
    set(alias_name "${alias_base}")
  else ()
    set(alias_name "${alias_base}::${name}")
  endif ()

  # replace :: in alias_name to `.` to make it a valid target name
  string(REPLACE "::" "." raw_name "${alias_name}")
  set(ret_raw_name ${raw_name} PARENT_SCOPE)
  set(ret_alias_name ${alias_name} PARENT_SCOPE)
endfunction()

function(_cc_impl cmake_fn_name name)
  _create_valid_raw_name_and_alias(${name})
  message(STATUS "${cmake_fn_name}: ${ret_raw_name} ${ret_alias_name}")
  cmake_language(CALL ${cmake_fn_name} ${ret_raw_name} ${ARGN})

  if (ret_raw_name STREQUAL ret_alias_name)
    message(DEBUG "The mangled name have the same content.")
  else ()
    cmake_language(CALL ${cmake_fn_name} ${ret_alias_name} ALIAS ${ret_raw_name})
  endif ()
  set_target_properties(${ret_raw_name} PROPERTIES OUTPUT_NAME ${name})
endfunction()

function(cc_binary name)
  _cc_impl(add_executable ${name} ${ARGN})
endfunction()

function(cc_library name)
  _cc_impl(add_library ${name} ${ARGN})
endfunction()