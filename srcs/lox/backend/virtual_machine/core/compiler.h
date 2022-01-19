//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_

/**
 * Compiler is a parser that generate bytecodes instead of AST.
 *
 * There are some thing to note for myself.
 * 1. In summary. Pratt Parser makes no essentially difference from RD parse, the core difference is the RD parser parse
 * expressions in a different way.
 * >> Read the http://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/ will give you a
 * smooth(maybe) transition from RD parser to Pratt Parser.
 *
 * 2. Most code-gen are simple 1 pass style, we can gen bytecode as we see the source code, but some cases are more
 * complicated, for we had to do parsing and code-gen interleaved. For example, `lvalue = rvalue_expression`, we had to
 * generate bytecode for the rvalue expression, before we generate the bytecode for the lvalue assignment.
 * And for this reason, most code about byte-code gen will be located in FunctionUnit, and some will be located in
 * Compiler.
 *
 * 3. The code-gen process is recursive, and it works with stack machine naturally. e.g., for a sub-expression, if it
 * get parsed first, it should be executed first too, that is, it's byte code should be emitted first.
 */

#include <functional>

#include "lox/backend/virtual_machine/core/function_unit.h"
#include "lox/err_code.h"
#include "lox/frontend/scanner.h"
#include "lox/object/gc.h"
#include "lox/object/value.h"

namespace lox::vm {

/**
 * The int value of InfixPrecedence will be used in comparison, the order or the enum item is very important.
 */
enum class InfixPrecedence {
  ASSIGNMENT,   // =
  OR,           // or
  AND,          // and
  EQUALITY,     // == !=
  COMPARISON,   // < > <= >=
  TERM,         // + -
  FACTOR,       // * /
  UNARY,        // ! -
  CALL_OR_DOT,  // . ()
};
enum class InfixAssociativity {
  LEFT_TO_RIGHT,
  RIGHT_TO_LEFT,
};

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
   * Parse a new expression and emit code for it, at runtime, a new value will be left on stack after the code for
   * expression is executed.
   *
   * The param `lower_bound` and `associativity` is used when the compiled expression contains infix expression:
   * When a new infix operator comes, the parsing will continue if one of following conditions is met:
   *   1. The infix operators that has a higher `lower_bound`.
   *   2. The infix operators that has the same `lower_bound`, and `associativity` is RIGHT_TO_LEFT.
   *
   */
  void AnyExpression(InfixPrecedence lower_bound = InfixPrecedence::ASSIGNMENT);
  void EmitPrefix();
  void EmitInfix();

  void GetNamedValue(Token name);

  /**
   * When can_assign is false, this function could only be used to get named value.
   * When doing a assignment, the assigned value will be leaves on stack
   */
  void GetOrSetNamedValue(Token varaible_token, bool can_assign);
  bool CanAssign();

  /**
   * CreateFunc will generate a valid FunctionObject at compile time, and emit extra instruction to make it
   * a valid ObjClosure at runtime. the valid ObjClosure will be leaved on stack at runtime.
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
  InfixPrecedence last_expr_lower_bound = InfixPrecedence::ASSIGNMENT;
  friend struct ScopeGuard;
};

}  // namespace lox::vm
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_COMPILER_H_
