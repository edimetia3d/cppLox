//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_DEBUG_H_
#define CLOX_SRCS_CLOX_DEBUG_H_

#include <cstdio>

#include "lox/backend/virtual_machine/bytecode/chunk.h"

namespace lox {
namespace vm {

int disassembleInstruction(Chunk *chunk, int offset);

void disassembleChunk(Chunk *chunk, const char *name);
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_DEBUG_H_
