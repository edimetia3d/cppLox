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
  Token current;
  Token next;
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
 * Compiler is mainly a Pratt Parser, which could generate an bytecode IR that can work directly with stack machine.
 *
 * There are some thing to note for myself.
 * 1. In summary. Pratt Parser makes no essentially difference from RD parse, it is, just another parser,
 * and all `emitXXX()` here could be replaced with a `return CreateXXXAstNode()` to build an AST.
 * >> Read the http://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/ will give you a smooth(maybe) transition from RD parser to
 * Pratt Parser.
 * 2. It's easy to see that, when we replace `CreateXXXAstNode()` to `emitXXX()` in parser,
 * Obviously, the order of call to `emitXXX()` is same as `CreateXXXAstNode()`.
 * 3. We can prove that, The order of call `EvaluateXXXAstNode()` could be implemented to be basically same as call `CreateXXXAstNode()` when evaluator walking in same AST.
 * It's just that some control flow makes evaluation history changed a little. (If without control flow, the order would be exactly same.) So tree-walker evaluator
 * is basically evaluate **all the node in their creation order**.
 * This tells us: we could just follow the order of call `EvaluateXXXAstNode()` when running the emitted code to get correct answer.
 *  a. To most emitted code, **first emitted code get run first too**
 *  b. The `throw` we used in tree-walker could be simply simulated by a instruction pointer jump.
 *  c. The return could be simulated by a stack, like we did in Visitor Pattern of AST evaluation. That's why our compiler work with
 * stack machine naturally.
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
  void emitByte(uint8_t byte) { CurrentChunk()->WriteUInt8(byte, parser_.current->line); }
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
    double value = std::stod(parser_.current->lexeme);
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
