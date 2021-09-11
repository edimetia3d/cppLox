//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CHUNK_H_
#define CLOX_SRCS_CLOX_CHUNK_H_
#include "lox/backend/virtual_machine/common/buffer.h"
#include "lox/backend/virtual_machine/common/clox_value.h"
#include "lox/backend/virtual_machine/common/common.h"

namespace lox {
namespace vm {

enum class OpCode : uint8_t {
  OP_CONSTANT,
  OP_NIL,
  OP_TRUE,
  OP_FALSE,
  OP_POP,
  OP_EQUAL,
  OP_GREATER,
  OP_LESS,
  OP_NOT,
  OP_NEGATE,
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_DIVIDE,
  OP_PRINT,
  OP_RETURN,
};

struct Chunk {
  Buffer<uint8_t, 0> code;
  Buffer<Value, 0> constants;
  Buffer<int, 0> lines;

  void WriteOpCode(OpCode opcode, int line_number);
  void WriteUInt8(uint8_t data, int line_number);
  int ChunkSize();
  int addConstant(Value value);
  void DumpCode();
  void DumpCode(int offset);
  void DumpConstant();
};
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CHUNK_H_
