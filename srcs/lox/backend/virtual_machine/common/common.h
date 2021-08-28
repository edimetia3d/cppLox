//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_COMMON_H_
#define CLOX_SRCS_CLOX_COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifndef NDEBUG
#define DEBUG_TRACE_EXECUTION
#define DPRINTF(...) printf("[Dbg] " __VA_ARGS__)
#else
#define DPRINTF(...) ()
#endif
#endif  // CLOX_SRCS_CLOX_COMMON_H_
