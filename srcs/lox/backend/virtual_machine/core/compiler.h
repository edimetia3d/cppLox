//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

#include <functional>

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/common/clox_object.h"
#include "lox/backend/virtual_machine/common/err_code.h"
#include "lox/backend/virtual_machine/common/hash_map.h"
#include "lox/frontend/scanner.h"

#define STACK_LOOKUP_OFFSET_MAX (UINT8_MAX + 1)  // we only use one byte to store the local lookup offset

namespace lox {

namespace vm {
struct Parser {
  Token previous;  // previous is the last consumed token
  Token current;   // current is the next might to be consumed token.
  bool hadError = false;
  bool panicMode = false;
};

enum class OperatorType {
  NONE,
  ASSIGNMENT,  // =
  OR,          // or
  AND,         // and
  EQUALITY,    // == !=
  COMPARISON,  // < > <= >=
  TERM,        // + -
  FACTOR,      // * /
  UNARY,       // ! -
  CALL,        // . ()
  PRIMARY
};
using Precedence = OperatorType;  // OperatorType is intended to sorted by precedence

class Compiler;
struct ParseRule {
  std::function<void(Compiler*)> EmitPrefixFn;
  std::function<void(Compiler*)> EmitInfixFn;
  OperatorType operator_type;
};

struct CompileUnit {
  enum class CUType { UNKOWN, FUNCTION, SCRIPT };
  CompileUnit(CUType type, std::string name) : type(type) {
    func = new ObjFunction();  // obj function will get gc cleaned, so we only new , not delete
    func->name = name;
  }
  struct Local {
    Token name;
    int depth;
  };
  ObjFunction* func;
  CUType type = CUType::UNKOWN;
  Local locals[STACK_LOOKUP_OFFSET_MAX];
  int localCount = 0;
  int scopeDepth = 0;
};

/**
 * Compiler is mainly a Pratt Parser, which could generate an bytecode IR that can work directly with stack machine.
 *
 * There are some thing to note for myself.
 * 1. In summary. Pratt Parser makes no essentially difference from RD parse, it is, just another parser,
 * and all `emitOpCodeXXX()` here could be replaced with a `return CreateXXXAstNode()` to build an AST.
 * >> Read the http://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/ will give you a
 * smooth(maybe) transition from RD parser to Pratt Parser.
 * 2. It's easy to see that, when we replace `CreateXXXAstNode()` to `emitOpCodeXXX()` in parser,
 * Obviously, the order of call to `emitOpCodeXXX()` is same as `CreateXXXAstNode()`.
 * 3. We can prove that, The order of call `EvaluateXXXAstNode()` could be implemented to be basically same as call
 * `CreateXXXAstNode()` when evaluator walking in same AST. It's just that some control flow makes evaluation history
 * changed a little. (If without control flow, the order would be exactly same.) So tree-walker evaluator is basically
 * evaluate **all the node in their creation order**. This tells us: we could just follow the order of call
 * `EvaluateXXXAstNode()` when running the emitted code to get correct answer. a. To most emitted code, **first emitted
 * code get run first too** b. The `throw` we used in tree-walker could be simply simulated by a instruction pointer
 * jump. c. The return could be simulated by a stack, like we did in Visitor Pattern of AST evaluation. That's why our
 * compiler work with stack machine naturally.
 */
class Compiler {
 public:
  Compiler() = default;
  ObjFunction* Compile(Scanner* scanner);

 private:
  void Advance();
  void errorAtCurrent(const char* message);
  void error(const char* message);
  void errorAt(Token token, const char* message);
  void Consume(TokenType type, const char* message);
  void emitByte(uint8_t byte) { CurrentChunk()->WriteUInt8(byte, parser_.previous->line); }
  void emitBytes(OpCode byte1, uint8_t byte2);
  void emitByte(OpCode opcode) { emitByte(static_cast<uint8_t>(opcode)); }
  void emitBytes(OpCode opcode0, OpCode opcode1) { emitBytes(opcode0, static_cast<uint8_t>(opcode1)); }
  Chunk* CurrentChunk();
  void endCompiler();
  void emitReturn();
  uint8_t makeConstant(Value value);
  void emitConstant(Value value) { emitBytes(OpCode::OP_CONSTANT, makeConstant(value)); }
  void number() {
    double value = std::stod(parser_.previous->lexeme);
    emitConstant(Value(value));
  }
  /**
   * The input operator_type is a mark to say that: we are parsing a expression that will be part of operand of
   * `operator_type` e.g. if `operator_type` is `+`, it means the expression we are parsing will be used in a binary
   * plus operation.
   *
   * Because we know what the expression will be used for, we known when to stop the parsing, that is , when we meet
   * some operator that has lower (or equal) precedence
   */
  void Expression(OperatorType operator_type = OperatorType::ASSIGNMENT);
  void grouping() {
    Expression();
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
  }
  void unary();
  void literal();
  friend std::vector<ParseRule> BuildRuleMap();
  ParseRule* getRule(TokenType type);
  ParseRule* getRule(Token token);
  void binary();
  void string();
  void and_();
  void or_();
  bool canAssign();
  void variable();
  bool MatchAndAdvance(TokenType type);
  void declaration();
  void statement();
  void printStatement();
  bool Check(TokenType type);
  void expressionStatement();
  void synchronize();
  void varDeclaration();
  uint8_t parseVariable(const char* string);
  uint8_t identifierConstant(Token token);
  void defineVariable(uint8_t global);
  void namedVariable(Token varaible_token);

  CompileUnit* current_cu_;
  Parser parser_;
  Scanner* scanner_;
  Precedence last_expression_precedence = Precedence::NONE;
  struct LoopBreak {
    int offset = -1;
    LoopBreak* next = nullptr;
    int level = -1;   // nested loop level value
  } loop_break_info;  // a dummy head
  int loop_nest_level = -1;
  void openBreak();
  void breakStatement();
  void closeBreak();
  void block();
  void beginScope();
  void endScope();
  void declareVariable();
  void addLocal(Token shared_ptr);
  bool identifiersEqual(Token shared_ptr, Token shared_ptr_1);
  int resolveLocal(Token shared_ptr);
  void markInitialized();
  void ifStatement();
  int emitJumpDown(OpCode jump_cmd);
  void patchJumpDown(int jump);
  void whileStatement();
  void emitJumpBack(int start);
  void forStatement();
};

}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
