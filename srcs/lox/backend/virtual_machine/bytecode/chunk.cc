//
// LICENSE: MIT
//
#include "lox/backend/virtual_machine/bytecode/chunk.h"

#include "lox/backend/virtual_machine/bytecode/debug.h"

namespace lox {
namespace vm {

void Chunk::WriteUInt8(uint8_t data, int line_number) {
  code.push_back(data);
  lines.push_back(line_number);
}
int Chunk::addConstant(Object value) {
  int index = constants.size();
  constants.push_back(value);
  return index;
}
int Chunk::ChunkSize() { return code.size(); }
void Chunk::WriteOpCode(OpCode opcode, int line_number) { WriteUInt8(static_cast<uint8_t>(opcode), 0); }
void Chunk::DumpCode(const char *name) { disassembleChunk(this, name); }
void Chunk::DumpByOffset(int offset) { disassembleInstruction(this, offset); }
void Chunk::DumpConstant() {
  printf("== Constant ==\n");
  for (Object *p = constants.data(); p != constants.data() + constants.size(); ++p) {
    printf("[ ");
    lox::printValue(*p);
    printf(" ]");
  }
  printf("\n");
}
}  // namespace vm
}  // namespace lox
