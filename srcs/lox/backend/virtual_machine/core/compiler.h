//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

#include <functional>

#include "lox/backend/virtual_machine/bytecode/chunk.h"
#include "lox/backend/virtual_machine/common/clox_value.h"
#include "lox/backend/virtual_machine/common/err_code.h"
#include "lox/backend/virtual_machine/common/hash_map.h"
#include "lox/frontend/scanner.h"

#define STACK_LOOKUP_OFFSET_MAX (UINT8_MAX + 1)  // we only use one byte to store the local lookup offset
#define UPVALUE_LIMIT 256
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
  ASSIGNMENT,   // =
  OR,           // or
  AND,          // and
  EQUALITY,     // == !=
  COMPARISON,   // < > <= >=
  TERM,         // + -
  FACTOR,       // * /
  UNARY,        // ! -
  CALL_OR_DOT,  // . ()
  PRIMARY
};
using Precedence = OperatorType;  // OperatorType is intended to sorted by precedence

enum class ScopeType { UNKOWN, BLOCK, IF_ELSE, WHILE, FOR, FUNCTION };

enum class FunctionType { UNKNOWN, FUNCTION, METHOD, INITIALIZER, SCRIPT };
class Compiler;
struct ParseRule {
  std::function<void(Compiler*)> EmitPrefixFn;
  std::function<void(Compiler*)> EmitInfixFn;
  OperatorType operator_type;
};
/**
 * Function compilation unit is the representation of function during compilation
 */
struct FunctionCU {
  FunctionCU(FunctionCU* enclosing, FunctionType type, const std::string& name) : type(type) {
    enclosing_ = enclosing;
    func = new ObjFunction();  // obj function will get gc cleaned, so we only new , not delete
    func->name = name;
    // the function object will be pushed to stack at runtime, so locals[0] is occupied here
    auto& local = locals[localCount++];
    if (type == FunctionType::METHOD || type == FunctionType::INITIALIZER) {
      local.name = "this";
    } else {
      local.name = name;
    }
    local.depth = 0;
  }
  struct Local {
    std::string name;
    int depth;
    bool isCaptured = false;
  };
  struct UpValue {
    bool isLocal = false;
    int index;
  };
  FunctionCU* enclosing_ = nullptr;
  ObjFunction* func;
  FunctionType type = FunctionType::UNKNOWN;
  Local locals[STACK_LOOKUP_OFFSET_MAX];
  UpValue upvalues[UPVALUE_LIMIT];
  int localCount = 0;
  int scopeDepth = 0;
};

struct ClassScope {
  ClassScope(ClassScope* encolsing) : enclosing(encolsing) {}
  ClassScope* enclosing = nullptr;
  bool hasSuperclass = false;
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
  Compiler();
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
  void endFunctionCompilation();
  void emitDefaultReturn();
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
  void call();
  void dot();
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
  void namedVariable(Token varaible_token, bool can_assign);

  FunctionCU* current_cu_;
  ClassScope* currentClass = nullptr;
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
  void beginScope(ScopeType type);
  void endScope(ScopeType type);
  void declareVariable();
  void addLocal(Token shared_ptr);
  bool identifiersEqual(const std::string& t0, const std::string& t1);
  int resolveLocal(FunctionCU* cu, Token token);
  void markInitialized();
  void ifStatement();
  int emitJumpDown(OpCode jump_cmd);
  void patchJumpDown(int jump);
  void whileStatement();
  void emitJumpBack(int start);
  void forStatement();
  void patchBreaks(int level);
  void createBreakJump(int level);
  int updateScopeCount();
  void funDeclaration();
  void func(FunctionType type);
  uint8_t argumentList();
  void returnStatement();
  int resolveUpvalue(FunctionCU* cu, Token sharedPtr);
  int addUpvalue(FunctionCU* cu, uint8_t index, bool isOnStack);
  void cleanUpLocals(int scope_var_num);
  static void markRoots(void* compiler);
  GC::RegisterMarkerGuard marker_register_guard;
  void classDeclaration();
  void method();
  void this_();
};

}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
