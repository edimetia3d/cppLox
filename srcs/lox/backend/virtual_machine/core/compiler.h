//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

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
 *
 * Most code-gen are simple 1 pass style, we can gen bytecode as we see the source code, but some cases are more
 * complicated, for we had to do parsing and code-gen interleaved. For example, `lvalue = rvalue_expression`, we had to
 * generate bytecode for the rvalue expression, before we generate the bytecode for the lvalue assignment.
 * And for this reason, most code about byte-code gen will be located in FunctionUnit, and some will be located in
 * Compiler.
 */

#include <functional>

#include "lox/backend/virtual_machine/core/function_unit.h"
#include "lox/err_code.h"
#include "lox/frontend/scanner.h"
#include "lox/object/gc.h"
#include "lox/object/value.h"

namespace lox::vm {

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

class Compiler {
 public:
  Compiler();
  ObjFunction* Compile(Scanner* scanner);

 private:
  void Advance();
  void ErrorAt(Token token, const char* message);
  void Consume(TokenType type, const char* message);
  bool MatchAndAdvance(TokenType type);
  bool Check(TokenType type);
  void Synchronize();

  void AnyStatement();

  void BlockStmt();
  void BreakOrContinueStmt();
  void ClassDefStmt();
  void ExpressionStmt();
  void ForStmt();
  void FunStmt();
  void IfStmt();
  void PrintStmt();
  void ReturnStmt();
  void VarDefStmt();
  void WhileStmt();

  /**
   * The input operator_type is a mark to say that: we are parsing a expression that will be part of operand of
   * `operator_type` e.g. if `operator_type` is `+`, it means the expression we are parsing will be used in a binary
   * plus operation.
   *
   * Because we know what the expression will be used for, we known when to stop the parsing, that is , when we meet
   * some operator that has lower (or equal) precedence
   */
  void AnyExpression(OperatorType operator_type = OperatorType::ASSIGNMENT);
  void GroupingExpr();
  void NumberExpr();
  void CallExpr();
  void DotExpr();
  void UnaryExpr();
  void LiteralExpr();

  void BinaryExpr();
  void StringExpr();
  void AndExpr();
  void OrExpr();
  void VariableExpr();
  void ThisExpr();

  /**
   * Get named value from the current scope, and leaves the value on stack.
   *
   * When can_assign is false, this function could only be used to get named value.
   * When doing a assignment, the assigned value will be leaves on stack
   */

  void GetNamedValue(Token name);
  void GetOrSetNamedValue(Token varaible_token, bool can_assign);
  bool CanAssign();

  /**
   * CreateFunc will create a ObjFunction stored in chunk's constant,
   * and emit a OP_CLOUSRE to leave a ObjClosure on stack at runtime.
   */
  void CreateFunc(FunctionType type);
  uint8_t ArgumentList();
  static void MarkRoots(void* compiler);

  GC::RegisterMarkerGuard marker_register_guard;
  FunctionUnit* cu_ = nullptr;
  /**
   * Push a new Function Unit as the main compilation unit
   */
  void PushCU(FunctionType type, const std::string& name);
  /**
   * Pop and return the current Function Unit, and switch to the enclosing Function Unit as main compilation unit.
   *
   * A Pop means the current Function Unit is finished.
   *
   * @return
   */
  std::unique_ptr<FunctionUnit> PopCU();
  struct ClassLevel {
    explicit ClassLevel(ClassLevel* encolsing) : enclosing(encolsing) {}
    ClassLevel* enclosing = nullptr;
    bool hasSuperclass = false;
  }* currentClass = nullptr;
  Token previous;  // previous is the last consumed token
  Token current;   // current is the next might to be consumed token.
  bool had_error = false;
  bool panic_mode = false;
  Scanner* scanner_;
  Precedence last_expression_precedence = Precedence::NONE;
  friend struct ScopeGuard;
  friend int BuildRuleMap();
};

}  // namespace lox::vm
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
