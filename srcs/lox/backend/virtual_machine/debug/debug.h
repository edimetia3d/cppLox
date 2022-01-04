//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_DEBUG_H_
#define CLOX_SRCS_CLOX_DEBUG_H_

#include <cstdio>

#include "lox/backend/virtual_machine/core/chunk.h"
#include "lox/backend/virtual_machine/core/vm.h"

namespace lox::vm {
#ifndef NDEBUG
int DumpInstruction(const Chunk* chunk, int offset);

void DumpChunkCode(const Chunk* chunk);

void DumpChunkConstant(const Chunk* chunk);

void DumpStack(const VM* vm);
void DumpGlobal(const VM* vm);
#else
#define DumpInstruction(__VA_ARGS__) \
  do {                               \
  } while (0)
#define DumpChunkCode(__VA_ARGS__) \
  do {                             \
  } while (0)
#define DumpChunkConstant(__VA_ARGS__) \
  do {                                 \
  } while (0)
#define DumpStack(__VA_ARGS__) \
  do {                         \
  } while (0)
#define DumpGlobal(__VA_ARGS__) \
  do {                          \
  } while (0)
#endif
}  // namespace lox::vm
#endif  // CLOX_SRCS_CLOX_DEBUG_H_
