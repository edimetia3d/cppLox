//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/compiler.h"

#include <spdlog/spdlog.h>

#include <map>
#include <memory>

#include "lox/backend/virtual_machine/debug/debug.h"
#include "lox/backend/virtual_machine/errors.h"
#include "lox/common/finally.h"
#include "lox/common/global_setting.h"

// there is always a stack used by function pointer
namespace lox::vm {

struct InfixOpInfoMap {
  struct InfixOpInfo {
    InfixPrecedence precedence;
    InfixAssociativity associativity;
    ;
  };

  static InfixOpInfoMap &Instance() {
    static InfixOpInfoMap instance;
    return instance;
  }

  static InfixOpInfo *Get(TokenType type) {
    if (InfixOpInfoMap::Instance().data.contains(type)) {
      return &InfixOpInfoMap::Instance().data[type];
    }
    return nullptr;
  }

  static InfixOpInfo *Get(Token token) { return Get(token->type); }

  std::map<TokenType, InfixOpInfo> data;

 private:
  InfixOpInfoMap() {
    /**
     * There is no rule for the `operator=`, or the `TokenType::EQUAL`, because our implement of compiler will need more
     * tracking utils to support assignment, which will introduce unnecessary complexity.
     *
     * Assignment have only three cases
     * 1. `var a = expression`, initialization.
     * 2. `a = expression`, assignment.
     * 3. `a.b = expression`, set attribute.
     *
     * With only one token ahead is known. We can hardly know what the operator='s real meaning.
     * e.g. `a.b.c.d.e = expression`, when the compiler saw the `=`, it will only see a `e` as previous token,
     * it could be a global, a local, or a attribute, for these three cases would
     * generate different code, we would never go without knowing what the `e` is.
     *
     */
#define RULE_ITEM(TOKEN_T, PRECEDENCE_V, ASSOCIATIVITY_V)                                      \
  {                                                                                            \
    TokenType::TOKEN_T, { InfixPrecedence::PRECEDENCE_V, InfixAssociativity::ASSOCIATIVITY_V } \
  }
    // clang-format off
    auto map_tmp = std::map<TokenType,InfixOpInfo> {
      RULE_ITEM(LEFT_PAREN    , CALL_OR_DOT , LEFT_TO_RIGHT) ,
      RULE_ITEM(DOT           , CALL_OR_DOT , LEFT_TO_RIGHT) ,
      RULE_ITEM(MINUS         , TERM        , LEFT_TO_RIGHT) ,
      RULE_ITEM(PLUS          , TERM        , LEFT_TO_RIGHT) ,
      RULE_ITEM(SLASH         , FACTOR      , LEFT_TO_RIGHT) ,
      RULE_ITEM(STAR          , FACTOR      , LEFT_TO_RIGHT) ,
      RULE_ITEM(BANG_EQUAL    , EQUALITY    , LEFT_TO_RIGHT) ,
      RULE_ITEM(EQUAL         , ASSIGNMENT  , RIGHT_TO_LEFT) ,
      RULE_ITEM(EQUAL_EQUAL   , EQUALITY    , LEFT_TO_RIGHT) ,
      RULE_ITEM(GREATER       , COMPARISON  , LEFT_TO_RIGHT) ,
      RULE_ITEM(GREATER_EQUAL , COMPARISON  , LEFT_TO_RIGHT) ,
      RULE_ITEM(LESS          , COMPARISON  , LEFT_TO_RIGHT) ,
      RULE_ITEM(LESS_EQUAL    , COMPARISON  , LEFT_TO_RIGHT) ,
      RULE_ITEM(AND           , AND         , LEFT_TO_RIGHT) ,
      RULE_ITEM(OR            , OR          , LEFT_TO_RIGHT) ,
    };
    // clang-format on
#undef RULE_ITEM

    data = std::move(map_tmp);
  };
};

enum class ScopeType { UNKOWN, BLOCK, IF_ELSE, WHILE, FOR, FUNCTION, CLASS };

/**
 * ScopeGuard is a helper to emit some scope related code.
 * 1. local clean up code
 * 2. break path code.
 */
struct ScopeGuard {
  ScopeGuard(FunctionUnit *cu, ScopeType type) : cu(cu), type(type) {
    ++cu->current_semantic_scope_level;
    if (type == ScopeType::FOR || type == ScopeType::WHILE) {
      cu->loop_infos.push_back(FunctionUnit::LoopInfo{.initial_stack_size = static_cast<int>(cu->locals.size())});
    }
  }
  int OutOfScopeVar() {
    int scope_var_count = 0;
    uint64_t N = cu->locals.size();
    while (cu->locals[N - scope_var_count - 1].semantic_scope_depth >= cu->current_semantic_scope_level) {
      ++scope_var_count;
    }
    return scope_var_count;
  }
  ~ScopeGuard() {
    if (type == ScopeType::FUNCTION) {
      return;  // function scope is cleared by a frame switch at runtime, all stack variables in the frame will be
               // discarded.
    }

    int var_num_of_just_out_of_scope = OutOfScopeVar();
    cu->CleanUpNLocalFromTail(var_num_of_just_out_of_scope);

    if (type == ScopeType::FOR || type == ScopeType::WHILE) {
      for (auto &jmp : cu->loop_infos.back().breaks) {
        cu->JumpHerePatch(jmp);
      }
      cu->loop_infos.pop_back();
    }
    --cu->current_semantic_scope_level;
  }
  FunctionUnit *cu;
  ScopeType type;
};

ObjFunction *Compiler::Compile(Scanner *scanner, std::string *err_msg) {
  err_msgs.clear();
  PushCU(FunctionType::SCRIPT, "<script>");

  scanner_ = scanner;
  Advance();
  while (!MatchAndAdvance(TokenType::EOF_TOKEN)) {
    AnyStatement();
  }
  auto top_level_cu = PopCU();
  if (err_msgs.size() > 0) {
    for (auto &msg : err_msgs) {
      *err_msg += (msg + "\n");
    }
    return nullptr;
  }
  return top_level_cu->func;
}
void Compiler::Advance() {
  if (previous && previous->type == TokenType::EOF_TOKEN) {
    throw CompilationError("Unexpected EOF");
  }
  previous = current;
  current = scanner_->ScanOne();
}
void Compiler::ErrorAt(Token token, const char *message) {
#ifdef UPSTREAM_STYLE_ERROR_MSG
  if (!panic_mode) {
    auto err_msg = CreateErrMsg(token, message);
    err_msgs.emplace_back(err_msg);
    panic_mode = true;
  }
#else
  auto err_msg = CreateErrMsg(token, message);
  err_msgs.emplace_back(err_msg);
  throw CompilationError(err_msg);
#endif
}
std::string Compiler::CreateErrMsg(const Token &token, const char *message) const {
  std::vector<char> buf(256);
  int offset = 0;
  if (token->type == TokenType::EOF_TOKEN) {
    offset += snprintf(buf.data() + offset, 256, "[line %d] Error at end: %s", token->line + 1, message);
  } else {
    offset += snprintf(buf.data() + offset, 256, "[line %d] Error at '%s': %s", token->line + 1, token->lexeme.c_str(),
                       message);
  }
  return std::string(buf.data());
}
void Compiler::Consume(TokenType type, const char *message) {
  if (current->type == type) {
    Advance();
    return;
  }
  ErrorAt(current, message);
}

void Compiler::AnyExpression(InfixPrecedence lower_bound) {
  /**
   *
   * It is easy to understand the expression category from it's runtime behavior.
   *  1. An expression that will consume zero or one value from stack is a prefix expression.
   *  2. An expression that will consume more than one values from stack is a infix expression.
   *
   * Every expression starts with a prefix expression, because `AnyExpression` will start a new expression,
   * and nothing would be on stack at that time logically, there is no way to emit a infix expression.
   *
   * prefix expression could be treated as the minimal expression, and infix expression is composed of prefix expression
   * fundamentally.
   *
   */
  auto bak = last_expr_lower_bound;
  last_expr_lower_bound = lower_bound;
  Advance();
  EmitPrefix();
  while (auto op_info = InfixOpInfoMap::Get(current)) {
    if ((op_info->precedence > lower_bound ||
         (op_info->precedence == lower_bound && op_info->associativity == InfixAssociativity::RIGHT_TO_LEFT))) {
      Advance();
      EmitInfix();
    } else {
      break;
    }
  }
  last_expr_lower_bound = bak;
}

bool Compiler::MatchAndAdvance(TokenType type) {
  if (!CheckCurrentTokenType(type)) return false;
  Advance();
  return true;
}

void Compiler::AnyStatement(const std::vector<TokenType> &not_allowed_stmt, const char *not_allowed_msg) {
#ifdef UPSTREAM_STYLE_ERROR_MSG
  DoAnyStatement(not_allowed_stmt, not_allowed_msg);
  if (panic_mode) {
    Synchronize();
    panic_mode = false;
  }
#else
  try {
    DoAnyStatement(not_allowed_stmt, not_allowed_msg);
  } catch (const CompilationError &e) {
    Synchronize();
  }
#endif
}

void Compiler::DoAnyStatement(const std::vector<TokenType> &not_allowed_stmt, const char *not_allowed_msg) {
  for (auto t : not_allowed_stmt) {
    if (CheckCurrentTokenType(t)) {
      ErrorAt(current, not_allowed_msg);
    }
  }
  if (MatchAndAdvance(TokenType::CLASS)) {
    ClassDefStmt();
  } else if (MatchAndAdvance(TokenType::FUN)) {
    FunStmt();
  } else if (MatchAndAdvance(TokenType::VAR)) {
    VarDefStmt();
  } else if (MatchAndAdvance(TokenType::IF)) {
    IfStmt();
  } else if (MatchAndAdvance(TokenType::WHILE)) {
    WhileStmt();
  } else if (MatchAndAdvance(TokenType::BREAK) || MatchAndAdvance(TokenType::CONTINUE)) {
    BreakOrContinueStmt();
  } else if (MatchAndAdvance(TokenType::FOR)) {
    ForStmt();
  } else if (MatchAndAdvance(TokenType::PRINT)) {
    PrintStmt();
  } else if (MatchAndAdvance(TokenType::RETURN)) {
    ReturnStmt();
  } else if (MatchAndAdvance(TokenType::LEFT_BRACE)) {
    ScopeGuard guard(cu_, ScopeType::BLOCK);
    BlockStmt();
  } else {
    ExpressionStmt();
  }
}
void Compiler::PrintStmt() {
  AnyExpression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  cu_->EmitByte(OpCode::OP_PRINT);
}
bool Compiler::CheckCurrentTokenType(TokenType type) { return current->type == type; }
void Compiler::ExpressionStmt() {
  AnyExpression();
  if (GlobalSetting().interactive_mode && !CheckCurrentTokenType(TokenType::SEMICOLON)) {
    return cu_->EmitByte(OpCode::OP_PRINT);
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after expression.");
  cu_->EmitByte(OpCode::OP_POP);
}
void Compiler::Synchronize() {
  while (current->type != TokenType::EOF_TOKEN) {
    switch (previous->type) {
      case TokenType::SEMICOLON:
        return;
      default:
        break;  // do nothing
    };
    switch (current->type) {
      case TokenType::CLASS:
      case TokenType::FUN:
      case TokenType::VAR:
      case TokenType::FOR:
      case TokenType::IF:
      case TokenType::WHILE:
      case TokenType::PRINT:
      case TokenType::RETURN:
        return;
      default:
        break;  // Do nothing.
    }

    Advance();
  }
}
void Compiler::VarDefStmt() {
  Consume(TokenType::IDENTIFIER, "Expect variable name.");
  Token var_name = previous;
  auto new_var = cu_->DeclNamedValue(var_name);

  if (MatchAndAdvance(TokenType::EQUAL)) {
    AnyExpression();
  } else {
    cu_->EmitByte(OpCode::OP_NIL);
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after VariableExpr declaration.");

  cu_->DefineNamedValue(new_var);
}

bool Compiler::CanAssign() { return last_expr_lower_bound <= InfixPrecedence::ASSIGNMENT; }
void Compiler::BlockStmt() {
  // Note that block stmt do not create a new scope, the scope for block stmt is created by caller.
  while (!CheckCurrentTokenType(TokenType::RIGHT_BRACE) && !CheckCurrentTokenType(TokenType::EOF_TOKEN)) {
    AnyStatement();
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after block.");
}

void Compiler::IfStmt() {
  ScopeGuard guard(cu_, ScopeType::IF_ELSE);
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'if'.");
  AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");

  auto jump_to_else = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
  // these expressions are allowed technically, but lox disable them on purpose
  AnyStatement({TokenType::CLASS, TokenType::FUN, TokenType::VAR}, "Expect expression.");
  auto jump_to_exit = cu_->CreateJumpDownHole(OpCode::OP_JUMP);
  cu_->JumpHerePatch(jump_to_else);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
  if (MatchAndAdvance(TokenType::ELSE)) {
    // these expressions are allowed technically, but lox disable them on purpose
    AnyStatement({TokenType::CLASS, TokenType::FUN, TokenType::VAR}, "Expect expression.");
  }
  cu_->JumpHerePatch(jump_to_exit);
}

void Compiler::WhileStmt() {
  ScopeGuard guard(cu_, ScopeType::WHILE);
  int loop_begin_offset = cu_->Chunk()->ChunkSize();
  cu_->loop_infos.back().beg_offset =
      loop_begin_offset;  // save the loop begin offset, so we can `continue` to here later
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'while'.");
  AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");

  auto exitJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
  // these expressions are allowed technically, but lox disable them on purpose
  AnyStatement({TokenType::CLASS, TokenType::FUN, TokenType::VAR}, "Expect expression.");
  cu_->EmitJumpBack(loop_begin_offset);
  cu_->JumpHerePatch(exitJump);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
}

void Compiler::ForStmt() {
  ScopeGuard guard(cu_, ScopeType::FOR);
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'for'.");
  if (MatchAndAdvance(TokenType::SEMICOLON)) {
    // No initializer.
  } else if (MatchAndAdvance(TokenType::VAR)) {
    VarDefStmt();
    cu_->loop_infos.back().contains_init_value = true;
  } else {
    ExpressionStmt();
  }

  int loop_begin_offset = cu_->Chunk()->ChunkSize();
  cu_->loop_infos.back().beg_offset =
      loop_begin_offset;  // save the loop begin offset, so we can `continue` to here later
  FunctionUnit::JumpDownHole exitJump;
  if (!MatchAndAdvance(TokenType::SEMICOLON)) {
    AnyExpression();
    Consume(TokenType::SEMICOLON, "Expect ';' after loop condition.");

    // Jump out of the loop if the condition is false.
    exitJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
    cu_->EmitByte(OpCode::OP_POP);  // Condition.
  }

  if (!MatchAndAdvance(TokenType::RIGHT_PAREN)) {
    auto bodyJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP);
    int incrementStart = cu_->Chunk()->ChunkSize();
    AnyExpression();                // this will leave a value on top of stack
    cu_->EmitByte(OpCode::OP_POP);  // discard stack top
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after for clauses.");

    cu_->EmitJumpBack(loop_begin_offset);
    loop_begin_offset = incrementStart;
    cu_->JumpHerePatch(bodyJump);
  }
  // these expressions are allowed technically, but lox disable them on purpose
  AnyStatement({TokenType::CLASS, TokenType::FUN, TokenType::VAR}, "Expect expression.");

  cu_->EmitJumpBack(loop_begin_offset);
  if (exitJump.beg_offset != -1) {
    cu_->JumpHerePatch(exitJump);
    cu_->EmitByte(OpCode::OP_POP);  // Condition.
  }
}
void Compiler::BreakOrContinueStmt() {
  /**
   * To support `break`, there are two things to consider:
   * 1. break a nested loop correctly, that is, we only break the innermost loop.
   * 2. break will clean up the local variables correctly, before we jump to the loop-end, stack should be cleared.
   * up.
   *
   * To support these: We will:
   *  a. log stack size when a loop scope started, so we can clear the stack correctly before jump
   *  b. patch the jump command at every end of loop scope
   */
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  if (cu_->loop_infos.empty()) {
    ErrorAt(previous, "Can not break/continue here");
  }
  int clear_size = cu_->locals.size() - cu_->loop_infos.back().initial_stack_size;
  if (previous->type == TokenType::BREAK) {
    cu_->CleanUpNLocalFromTail(clear_size);
    auto jump = cu_->CreateJumpDownHole(OpCode::OP_JUMP);
    cu_->loop_infos.back().breaks.push_back(jump);
  } else if (previous->type == TokenType::CONTINUE) {
    cu_->CleanUpNLocalFromTail(
        clear_size - cu_->loop_infos.back().contains_init_value);  // a continue may need to keep the init value
    assert(cu_->loop_infos.back().beg_offset > 0);
    cu_->EmitJumpBack(cu_->loop_infos.back().beg_offset);
  } else {
    ErrorAt(previous, "Fatality: Unknown break/continue statement");
  }
}

void Compiler::FunStmt() {
  Consume(TokenType::IDENTIFIER, "Expect function name.");
  Token function_name = previous;
  auto named_value = cu_->DeclNamedValue(function_name);
  CreateFunc(FunctionType::FUNCTION);
  cu_->DefineNamedValue(named_value);
}
void Compiler::CreateFunc(FunctionType type) {
  PushCU(type, previous->lexeme);
  ScopeGuard guard(cu_, ScopeType::FUNCTION);  // function and method share the same scope type

  Consume(TokenType::LEFT_PAREN, "Expect '(' after function name.");
  if (!CheckCurrentTokenType(TokenType::RIGHT_PAREN)) {
    do {
      // parameter count is limited to STACK_COUNT_LIMIT-1
      if (cu_->func->arity == (STACK_COUNT_LIMIT - 1)) {
        std::vector<char> buf(100);
        snprintf(buf.data(), buf.size(), "Can't have more than %d parameters.", (STACK_COUNT_LIMIT - 1));
        ErrorAt(current, buf.data());
      }
      cu_->func->arity++;
      Consume(TokenType::IDENTIFIER, "Expect parameter name.");
      auto arg = cu_->DeclNamedValue(previous);
      cu_->DefineNamedValue(arg);
    } while (MatchAndAdvance(TokenType::COMMA));
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after parameters.");
  Consume(TokenType::LEFT_BRACE, "Expect '{' before function body.");
  BlockStmt();

  auto new_cu = PopCU();
  // We will create a closure at runtime for the function, OP_CLOSURE will update the newly created closure's upvalue,
  // to make them in a valid state.
  cu_->EmitOpClosure(new_cu.get());
}

uint8_t Compiler::ArgumentList() {
  int argCount = 0;
  if (!CheckCurrentTokenType(TokenType::RIGHT_PAREN)) {
    do {
      AnyExpression();
      // argument/parameter count is limited to STACK_COUNT_LIMIT-1
      if (argCount == (STACK_COUNT_LIMIT - 1)) {
        std::vector<char> buf(100);
        snprintf(buf.data(), buf.size(), "Can't have more than %d arguments.", (STACK_COUNT_LIMIT - 1));
        ErrorAt(previous, buf.data());
      }
      argCount++;
    } while (MatchAndAdvance(TokenType::COMMA));
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after arguments.");
  return argCount;
}
void Compiler::ReturnStmt() {
  if (cu_->type == FunctionType::SCRIPT) {
    ErrorAt(previous, "Can't return from top-level code.");
  }
  if (MatchAndAdvance(TokenType::SEMICOLON)) {
    cu_->EmitDefaultReturn();
  } else {
    if (cu_->type == FunctionType::INITIALIZER) {
      ErrorAt(previous, "Can't return a value from an initializer.");
    }
    AnyExpression();
    Consume(TokenType::SEMICOLON, "Expect ';' after return value.");
    cu_->EmitByte(OpCode::OP_RETURN);
  }
}
void Compiler::MarkRoots() {
  Compiler *compiler = this;
  // RecursiveMark functions
  auto cu = compiler->cu_;
  while (cu) {
    GC::Instance().RecursiveMark(cu->func);
    cu = cu->enclosing;
  }
}

Compiler::Compiler() {
  GC::Instance().markers[this] = [this]() { MarkRoots(); };
}

void Compiler::ClassDefStmt() {
  Consume(TokenType::IDENTIFIER, "Expect class name.");
  Token class_name = previous;
  auto handle = cu_->DeclNamedValue(class_name);

  // use OP_CLASS to leave a value on the stack, to init the named value we just created.
  uint8_t name_const = cu_->GetSymbolConstant(class_name->lexeme);
  cu_->EmitBytes(OpCode::OP_CLASS, name_const);

  cu_->DefineNamedValue(handle);

  all_classes[class_name->lexeme] =
      ClassInfo{.name_token = class_name, .superclass = nullptr, .enclosing = current_class};
  current_class = &all_classes[class_name->lexeme];
  Finally reset_current_class_guard([this]() { this->current_class = this->current_class->enclosing; });

  if (MatchAndAdvance(TokenType::LESS)) {
    Consume(TokenType::IDENTIFIER, "Expect superclass name.");
    Token superclass_name = previous;
    GetNamedValue(superclass_name);
    if (class_name->lexeme == superclass_name->lexeme) {
      ErrorAt(superclass_name, "A class can't inherit from itself.");
    }
    current_class->superclass = &all_classes[superclass_name->lexeme];
    GetNamedValue(class_name);
    cu_->EmitByte(OpCode::OP_INHERIT);  // inherit will consume the two class object on stack, and leaves nothing on
                                        // stack, so stmt rule will be followed
  }
  // we give a new semantic scope to class, so it could be treated as a local scope, to enable the ability to close
  // values
  ScopeGuard guard(cu_, ScopeType::CLASS);
  GetNamedValue(class_name);  // put the class object on stack, so later method define could use it
  Consume(TokenType::LEFT_BRACE, "Expect '{' before class body.");
  while (!CheckCurrentTokenType(TokenType::RIGHT_BRACE) && !CheckCurrentTokenType(TokenType::EOF_TOKEN)) {
    Consume(TokenType::IDENTIFIER, "Expect Method name.");
    uint8_t fn_name_cst = cu_->GetSymbolConstant(previous->lexeme);
    FunctionType Type = FunctionType::METHOD;
    if (previous->lexeme == "init") {
      Type = FunctionType::INITIALIZER;
    }
    CreateFunc(Type);
    cu_->EmitBytes(OpCode::OP_METHOD, fn_name_cst);  // just move the objClosure on stack to the class object's dict
  }
  Consume(TokenType::RIGHT_BRACE, "Expect '}' after class body.");
  cu_->EmitByte(OpCode::OP_POP);  // pop the class object, for class def is always a stmt.
}

void Compiler::PushCU(FunctionType type, const std::string &name) {
  cu_ = new FunctionUnit(
      cu_, type, name, [this]() { return this->previous->line; },
      [this](const char *msg) { this->ErrorAt(this->previous, msg); });
}
std::unique_ptr<FunctionUnit> Compiler::PopCU() {
  cu_->EmitDefaultReturn();  // always inject a default return to make sure the function ends
#ifndef NDEBUG
  SPDLOG_DEBUG("=========== {:^20} CODE  ===========", cu_->func->name);
  DumpChunkCode(cu_->Chunk());
  SPDLOG_DEBUG("=========== {:^20} CONST ===========", cu_->func->name);
  DumpChunkConstant(cu_->Chunk());
  SPDLOG_DEBUG("=========== {:^20} END   ===========", cu_->func->name);
#endif
  auto latest_cu = cu_;
  cu_ = cu_->enclosing;
  return std::unique_ptr<FunctionUnit>(latest_cu);
}

void Compiler::GetNamedValue(Token name) {
  auto handle = cu_->ResolveNamedValue(name);
  cu_->EmitBytes(handle.get_op, handle.reslove.position);
}

void Compiler::EmitPrefix() {
  switch (previous->type) {
    case TokenType::LEFT_PAREN: {
      AnyExpression();
      Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
      break;
    }
    case TokenType::MINUS:
      [[fallthrough]];
    case TokenType::BANG: {
      auto op_type = previous->type;
      AnyExpression(InfixPrecedence::UNARY);
      cu_->EmitUnary(op_type);
      break;
    }
    case TokenType::IDENTIFIER: {
      if (IsAccessingClassAttr(previous, current)) {
        EmitClassAttrAccess(previous);
      } else {
        auto handle = cu_->ResolveNamedValue(previous);
        if (MatchAndAdvance(TokenType::EQUAL)) {
          if (CanAssign()) {
            AnyExpression();
            cu_->EmitBytes(handle.set_op, handle.reslove.position);
          } else {
            ErrorAt(previous, "Invalid assignment target.");
          }
        } else {
          cu_->EmitBytes(handle.get_op, handle.reslove.position);
        }
      }
      break;
    }
    case TokenType::STRING: {
      std::string tmp = previous->lexeme;
      *tmp.rbegin() = '\0';
      cu_->EmitConstant(Value(Symbol::Intern(tmp.c_str() + 1)));
      break;
    }
    case TokenType::NUMBER: {
      cu_->EmitConstant(lox::Value(std::stod(previous->lexeme)));
      break;
    }
    case TokenType::TRUE_TOKEN:
      [[fallthrough]];
    case TokenType::FALSE_TOKEN:
      [[fallthrough]];
    case TokenType::NIL: {
      cu_->EmitLiteral(previous->type);
      break;
    }
    case TokenType::THIS: {
      if (current_class == nullptr) {
        ErrorAt(previous, "Can't use 'this' outside of a class.");
      }
      GetNamedValue(previous);
      break;
    }
    case TokenType::SUPER: {
      if (!current_class) {
        ErrorAt(previous, "Can't use 'super' outside of a class.");
      } else if (!current_class->superclass) {
        ErrorAt(previous, "Can't use 'super' in a class with no superclass.");
      } else {
        EmitClassAttrAccess(current_class->superclass->name_token);
#ifdef UPSTREAM_STYLE_ERROR_MSG
        if (!CheckCurrentTokenType(TokenType::DOT)) {
          ErrorAt(current, "Expect '.' after 'super'.");
        }
        if (auto op_info = InfixOpInfoMap::Get(current)) {
          if ((op_info->precedence > last_expr_lower_bound ||
               (op_info->precedence == last_expr_lower_bound &&
                op_info->associativity == InfixAssociativity::RIGHT_TO_LEFT))) {
            Advance();
            if (!CheckCurrentTokenType(TokenType::IDENTIFIER)) {
              ErrorAt(current, "Expect superclass method name.");
            }
            EmitInfix();
          } else {
            ErrorAt(current, "Unexpected precedence");
          }
        }
#endif
      }

      break;
    }
    default:
      ErrorAt(previous, "Expect expression.");
  }
}
void Compiler::EmitClassAttrAccess(Token class_token) {
  cu_->ForceCloseValue(class_token);
  GetNamedValue(Token(TokenType::THIS, "this", previous->line));
  cu_->EmitByte(OpCode::OP_INSTANCE_TYPE_CAST);
}
bool Compiler::IsAccessingClassAttr(Token class_name, Token next_token) {
  if (all_classes.contains(class_name->lexeme) && next_token->type == TokenType::DOT) {
    if (!current_class) {
#ifdef UPSTREAM_STYLE_ERROR_MSG
      return false;
#else
      ErrorAt(class_name, "Can't access class attr outside of a method.");
#endif
    }

    // check if class_name is a base class of current class
    auto p = current_class;
    while (p && p->name_token->lexeme != class_name->lexeme) {
      p = p->superclass;
    }
    if (!p) {
      ErrorAt(class_name, "Can't access attr of class that is not the base class.");
    }
    return true;
  }
  return false;
}

void Compiler::EmitInfix() {
  switch (previous->type) {
    case TokenType::LEFT_PAREN: {
      cu_->EmitBytes(OpCode::OP_CALL, ArgumentList());
      break;
    }
    case TokenType::DOT: {
      Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
      uint8_t attr_name = cu_->GetSymbolConstant(previous->lexeme);

      if (CanAssign() && MatchAndAdvance(TokenType::EQUAL)) {
        AnyExpression();
        cu_->EmitBytes(OpCode::OP_SET_ATTR, attr_name);
      } else if (MatchAndAdvance(TokenType::LEFT_PAREN)) {
        uint8_t ArgCount = ArgumentList();
        cu_->EmitBytes(OpCode::OP_INVOKE, attr_name);
        cu_->EmitByte(ArgCount);
      } else {
        cu_->EmitBytes(OpCode::OP_GET_ATTR, attr_name);
      }
      break;
    }
    case TokenType::EQUAL: {
      ErrorAt(previous, "Invalid assignment target.");
    }
    case TokenType::MINUS:
      [[fallthrough]];
    case TokenType::PLUS:
      [[fallthrough]];
    case TokenType::SLASH:
      [[fallthrough]];
    case TokenType::STAR:
      [[fallthrough]];
    case TokenType::BANG_EQUAL:
      [[fallthrough]];
    case TokenType::EQUAL_EQUAL:
      [[fallthrough]];
    case TokenType::GREATER:
      [[fallthrough]];
    case TokenType::GREATER_EQUAL:
      [[fallthrough]];
    case TokenType::LESS:
      [[fallthrough]];
    case TokenType::LESS_EQUAL: {
      TokenType op_token = previous->type;
      AnyExpression(InfixOpInfoMap::Get(op_token)->precedence);
      cu_->EmitBinary(op_token);
      break;
    }
    case TokenType::AND: {
      auto end_jump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);

      cu_->EmitByte(OpCode::OP_POP);
      AnyExpression(InfixPrecedence::AND);

      cu_->JumpHerePatch(end_jump);
      break;
    }
    case TokenType::OR: {
      auto else_jump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
      auto end_jump = cu_->CreateJumpDownHole(OpCode::OP_JUMP);

      cu_->JumpHerePatch(else_jump);
      cu_->EmitByte(OpCode::OP_POP);

      AnyExpression(InfixPrecedence::OR);
      cu_->JumpHerePatch(end_jump);
      break;
    }
    default:
      ErrorAt(previous, "Expect expression.");
  }
}
Compiler::~Compiler() { GC::Instance().markers.erase(this); }
}  // namespace lox::vm
