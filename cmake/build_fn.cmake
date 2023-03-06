function(_create_valid_raw_name_and_alias)
  # get relative path
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" relative_path "${CMAKE_CURRENT_LIST_DIR}")

  # generate alias name
  string(REPLACE "/" "::" alias_base "${relative_path}")
  set(alias_name "${alias_base}::${name}")

  # replace :: in alias_name to _ to make it a valid target name
  string(REPLACE "::" "_" raw_name "${alias_name}")
  set(raw_name "${raw_name}")

  set(ret_raw_name ${raw_name} PARENT_SCOPE)
  set(ret_alias_name ${alias_name} PARENT_SCOPE)
endfunction()

function(cc_binary name)
  _create_valid_raw_name_and_alias(name)
  message(STATUS "cc_binary: ${ret_raw_name} ${ret_alias_name}")
  add_executable(${ret_raw_name} ${ARGN})
  add_executable(${ret_alias_name} ALIAS ${ret_raw_name})
  set_target_properties(${ret_raw_name} PROPERTIES OUTPUT_NAME ${name})
endfunction()

function(cc_library name)
  _create_valid_raw_name_and_alias(name)
  message(STATUS "cc_library: ${ret_raw_name} ${ret_alias_name}")
  add_library(${ret_raw_name} ${ARGN})
  add_library(${ret_alias_name} ALIAS ${ret_raw_name})
  set_target_properties(${ret_raw_name} PROPERTIES OUTPUT_NAME ${name})
endfunction()