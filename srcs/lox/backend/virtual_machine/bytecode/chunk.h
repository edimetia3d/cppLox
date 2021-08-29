//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CHUNK_H_
#define CLOX_SRCS_CLOX_CHUNK_H_
#include "lox/backend/virtual_machine/common/clox_value.h"
#include "lox/backend/virtual_machine/common/common.h"
#include "lox/backend/virtual_machine/common/memory.h"

namespace lox {
namespace vm {

enum class OpCode : uint8_t {
  OP_CONSTANT,
  OP_NIL,
  OP_TRUE,
  OP_FALSE,
  OP_EQUAL,
  OP_GREATER,
  OP_LESS,
  OP_NOT,
  OP_NEGATE,
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_DIVIDE,
  OP_RETURN,
};

struct Chunk {
  CustomVec<uint8_t, 0> code;
  CustomVec<Value, 0> constants;
  CustomVec<int, 0> lines;

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
