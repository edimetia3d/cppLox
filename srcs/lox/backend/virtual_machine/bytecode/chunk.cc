//
// LICENSE: MIT
//
#include "lox/backend/virtual_machine/bytecode/chunk.h"
namespace lox {
namespace vm {

void Chunk::WriteUInt8(uint8_t data, int line_number) {
  code.push_back(data);
  lines.push_back(line_number);
}
int Chunk::addConstant(Value value) {
  int index = constants.size();
  constants.push_back(value);
  return index;
}
int Chunk::ChunkSize() { return code.size(); }
void Chunk::WriteOpCode(OpCode opcode, int line_number) { WriteUInt8(static_cast<uint8_t>(opcode), 0); }
}  // namespace vm
}  // namespace lox
