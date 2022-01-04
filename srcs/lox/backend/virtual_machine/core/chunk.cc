//
// LICENSE: MIT
//
#include "chunk.h"

namespace lox::vm {

void Chunk::WriteUInt8(uint8_t data, int line_number) {
  code.push_back(data);
  lines.push_back(line_number);
}
Chunk::ConstatntIndex Chunk::AddConstant(Value value) {
  int index = constants.size();
  constants.push_back(value);
  assert(index < (1 << sizeof(Chunk::ConstatntIndex)));
  return index;
}
int Chunk::ChunkSize() const { return code.size(); }
}  // namespace lox::vm
