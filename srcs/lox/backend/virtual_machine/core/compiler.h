//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

#include <functional>

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

enum class Precedence {
  PREC_NONE,
  PREC_ASSIGNMENT,  // =
  PREC_OR,          // or
  PREC_AND,         // and
  PREC_EQUALITY,    // == !=
  PREC_COMPARISON,  // < > <= >=
  PREC_TERM,        // + -
  PREC_FACTOR,      // * /
  PREC_UNARY,       // ! -
  PREC_CALL,        // . ()
  PREC_PRIMARY
};

class Compiler;
struct ParseRule {
  std::function<void(Compiler*)> prefix;
  std::function<void(Compiler*)> infix;
  Precedence precedence;
};

/**
 * Compiler is basically a Pratt Parser, which could generate an IR that can work directly with stack machine.
 */
class Compiler {
 public:
  Compiler() = default;
  ErrCode Compile(Scanner* scanner, Chunk* target);

 private:
  void Advance();
  void errorAtCurrent(const char* message);
  void error(const char* message);
  void errorAt(Token token, const char* message);
  void Consume(TokenType type, const char* message);
  void emitByte(uint8_t byte) { CurrentChunk()->WriteUInt8(byte, parser_.previous->line); }
  void emitBytes(OpCode byte1, uint8_t byte2);
  void emitOpCode(OpCode opcode) { emitByte(static_cast<uint8_t>(opcode)); }
  Chunk* CurrentChunk();
  Parser parser_;
  Scanner* scanner_;
  Chunk* current_trunk_;
  void endCompiler();
  void emitReturn();
  uint8_t makeConstant(Value value);
  void emitConstant(Value value) { emitBytes(OpCode::OP_CONSTANT, makeConstant(value)); }
  void number() {
    double value = std::stod(parser_.previous->lexeme);
    emitConstant(value);
  }
  void Expression(Precedence precedence);
  void grouping() {
    Expression(Precedence::PREC_ASSIGNMENT);
    Consume(TokenType::LEFT_PAREN, "Expect ')' after expression.");
  }
  void unary();
  ParseRule* getRule(TokenType type);
  ParseRule* getRule(Token token);
  void binary();
};

}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
