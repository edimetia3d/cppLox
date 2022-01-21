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
 * Chunk is an abstraction of `data segment` and `code segment`.
 *
 * Every chunk is logically belong to a function. That is, a valid chunk's bytecode usually begin with some stack
 * access, to process function arguments, and ended with a return instruction, to leave some value on stack.
 *
 * Chunk will only get updated at compile time, and will not be updated at runtime. That is, only a compiler should
 * update some chunk.
 *
 * Note that some data like immediate value, or jump offset, will be stored in the chunk's code directly.
 */
struct Chunk {
  std::vector<uint8_t> code;
  std::vector<Value> constants;
  std::vector<int> lines;

  void WriteUInt8(uint8_t data, int line_number);
  [[nodiscard]] std::size_t ChunkSize() const;
  std::size_t AddConstant(Value value);
};
}  // namespace lox::vm
#endif  // CLOX_SRCS_CLOX_CHUNK_H_
