//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CHUNK_H_
#define CLOX_SRCS_CLOX_CHUNK_H_

#include <vector>

#include "lox/backend/virtual_machine/core/opcode.h"
#include "lox/object/value.h"

namespace lox::vm {
/**
 * Chunk is an abstraction of `data segment` and `code segment` of the stack machine, the code could run directly.
 *
 * Every chunk is logically belong to a function. That is, a valid chunk usually begin with some stack access, to
 * get function arguments, and ended with a return instruction, to leave some value on stack.
 *
 * Chunk will only get updated at compile time, and will not be updated at runtime. That is, only a compiler should
 * update some chunk.
 *
 * When the chunk is executed, the instruction in it could access three kinds of data:
 * 1. Access stack variables (only at runtime in VM) with instruction like OP_POP/OP_PUSH
 * 2. Access constants (in itself) with instruction like OP_CONSTANT
 * 3. Access globals variables (only at runtime in VM) with instruction like OP_GET_GLOBAL/OP_SET_GLOBAL
 *
 * Note that some data like immediate value, or jump offset, will be stored in the chunk.code directly.
 */
struct Chunk {
  using ConstatntIndex = uint8_t;
  std::vector<uint8_t> code;
  std::vector<Value> constants;
  std::vector<int> lines;

  void WriteUInt8(uint8_t data, int line_number);
  int ChunkSize() const;
  ConstatntIndex AddConstant(Value value);
};
}  // namespace lox::vm
#endif  // CLOX_SRCS_CLOX_CHUNK_H_
