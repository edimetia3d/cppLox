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
  OP_CONSTANT,  // read constant
  OP_NIL,
  OP_TRUE,
  OP_FALSE,
  OP_POP,
  OP_GET_LOCAL,
  OP_SET_LOCAL,
  OP_GET_GLOBAL,
  OP_DEFINE_GLOBAL,
  OP_SET_GLOBAL,
  OP_GET_UPVALUE,
  OP_SET_UPVALUE,
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
  OP_JUMP,
  OP_JUMP_IF_FALSE,
  OP_JUMP_BACK,
  OP_CALL,
  OP_CLOSURE,        // create closure
  OP_CLOSE_UPVALUE,  // move a stack value to heap
  OP_RETURN,
  OP_CLASS,
  OP_SET_ATTR,
  OP_GET_ATTR,
  OP_METHOD,
  OP_INVOKE,
  OP_INHERIT,
};

struct Chunk {
  Buffer<uint8_t, 0> code;
  Buffer<Object, 0> constants;
  Buffer<int, 0> lines;

  void WriteOpCode(OpCode opcode, int line_number);
  void WriteUInt8(uint8_t data, int line_number);
  int ChunkSize();
  int addConstant(Object value);
  void DumpCode(const char* name = "Dump");
  void DumpByOffset(int offset);
  void DumpConstant();
};
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CHUNK_H_
