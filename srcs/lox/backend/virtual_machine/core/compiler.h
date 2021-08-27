//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/common/err_code.h"
#include "lox/frontend/scanner.h"
namespace lox {

namespace vm {
struct Parser {
  Token previous;
  Token current;
  bool hadError = false;
  bool panicMode = false;
};
class Compiler {
 public:
  Compiler(Scanner& scanner) : scanner_(scanner) {}
  ErrCode Compile(Chunk* target);

 private:
  void Advance();
  void errorAtCurrent(const char* message);
  void error(const char* message);
  void errorAt(Token token, const char* message);
  void Consume(TokenType type, const char* message);
  void emitByte(uint8_t byte) { CurrentChunk()->WriteUInt8(byte, parser_.previous->line); }
  void emitBytes(uint8_t byte1, uint8_t byte2) {
    emitByte(byte1);
    emitByte(byte2);
  }
  void emitOpCode(OpCode opcode) { emitByte(static_cast<uint8_t>(opcode)); }
  Chunk* CurrentChunk();
  Parser parser_;
  Scanner scanner_;
  Chunk* current_trunk_;
  void endCompiler();
  void emitReturn();
};
}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
