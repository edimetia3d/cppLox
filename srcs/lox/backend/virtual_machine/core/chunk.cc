//
// LICENSE: MIT
//
#include "chunk.h"

namespace lox::vm {

void Chunk::WriteUInt8(uint8_t data, int line_number) {
  code.push_back(data);
  lines.push_back(line_number);
}
std::size_t Chunk::AddConstant(Value value) {
  int index = constants.size();
  constants.push_back(value);
  return index;
}
std::size_t Chunk::ChunkSize() const { return code.size(); }
} // namespace lox::vm
