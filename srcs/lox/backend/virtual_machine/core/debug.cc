//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/debug.h"

#include "lox/backend/virtual_machine/common/clox_value.h"
namespace lox {
namespace vm {
static int simpleInstruction(const char *name, int offset) {
  printf("%s\n", name);
  return offset + 1;
}

static int constantInstruction(const char *name, Chunk *chunk, int offset) {
  uint8_t const_index = chunk->code[offset + 1];
  printf("%-16s %4d '", name, const_index);
  printValue(chunk->constants[const_index]);
  printf("'\n");
  return offset + 2;
}

int disassembleInstruction(Chunk *chunk, int offset) {
  printf("%04d ", offset);
  if (offset > 0 && chunk->lines[offset] == chunk->lines[offset - 1]) {
    printf("   | ");
  } else {
    printf("%4d ", chunk->lines[offset]);
  }
  auto instruction = static_cast<OpCode>(chunk->code[offset]);
  switch (instruction) {
    case OpCode::OP_CONSTANT:
      return constantInstruction("OP_CONSTANT", chunk, offset);
    case OpCode::OP_ADD:
      return simpleInstruction("OP_ADD", offset);
    case OpCode::OP_SUBTRACT:
      return simpleInstruction("OP_SUBTRACT", offset);
    case OpCode::OP_MULTIPLY:
      return simpleInstruction("OP_MULTIPLY", offset);
    case OpCode::OP_DIVIDE:
      return simpleInstruction("OP_DIVIDE", offset);
    case OpCode::OP_NEGATE:
      return simpleInstruction("OP_NEGATE", offset);
    case OpCode::OP_RETURN:
      return simpleInstruction("OP_RETURN", offset);
    default:
      printf("Unknown opcode %d\n", instruction);
      return offset + 1;
  }
}
void disassembleChunk(Chunk *chunk, const char *name) {
  printf("== %s ==\n", name);

  for (int offset = 0; offset < chunk->ChunkSize();) {
    offset = disassembleInstruction(chunk, offset);
  }
}
}  // namespace vm
}  // namespace lox
