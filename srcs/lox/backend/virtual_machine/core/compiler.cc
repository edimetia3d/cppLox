//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/compiler.h"

#include <map>

#include "lox/backend/virtual_machine/debug/debug.h"

namespace lox::vm {

struct ParseRule {
  std::function<void(Compiler *)> EmitPrefixFn;
  std::function<void(Compiler *)> EmitInfixFn;
  OperatorType operator_type;
};

static std::vector<ParseRule> &RuleMap();
int BuildRuleMap() {
#define RULE_ITEM(TOKEN_T, MEMBER_FN0, MEMBER_FN1, PRECEDENCE_V)                         \
  {                                                                                      \
    TokenType::TOKEN_T, ParseRule { MEMBER_FN0, MEMBER_FN1, OperatorType::PRECEDENCE_V } \
  }
#define M(MEMBER_FN) &Compiler::MEMBER_FN
  // clang-format off
    auto map_tmp = std::map<TokenType,ParseRule> {
/*     TokenType ,         PrefixEmitFn , InfixEmitFn, OperatorType */
      RULE_ITEM(LEFT_PAREN   , M(GroupingExpr), M(CallExpr)  , CALL_OR_DOT),
      RULE_ITEM(RIGHT_PAREN  , nullptr        , nullptr      , NONE),
      RULE_ITEM(LEFT_BRACE   , nullptr        , nullptr      , NONE),
      RULE_ITEM(RIGHT_BRACE  , nullptr        , nullptr      , NONE),
      RULE_ITEM(COMMA        , nullptr        , nullptr      , NONE),
      RULE_ITEM(DOT          , nullptr        , M(DotExpr)   , CALL_OR_DOT),
      RULE_ITEM(MINUS        , M(UnaryExpr)   , M(BinaryExpr), TERM),
      RULE_ITEM(PLUS         , nullptr        , M(BinaryExpr), TERM),
      RULE_ITEM(SEMICOLON    , nullptr        , nullptr      , NONE),
      RULE_ITEM(SLASH        , nullptr        , M(BinaryExpr), FACTOR),
      RULE_ITEM(STAR         , nullptr        , M(BinaryExpr), FACTOR),
      RULE_ITEM(BANG         , M(UnaryExpr)   , nullptr      , NONE),
      RULE_ITEM(BANG_EQUAL   , nullptr        , M(BinaryExpr), EQUALITY),
      RULE_ITEM(EQUAL        , nullptr        , nullptr      , NONE),
      RULE_ITEM(EQUAL_EQUAL  , nullptr        , M(BinaryExpr), EQUALITY),
      RULE_ITEM(GREATER      , nullptr        , M(BinaryExpr), COMPARISON),
      RULE_ITEM(GREATER_EQUAL, nullptr        , M(BinaryExpr), COMPARISON),
      RULE_ITEM(LESS         , nullptr        , M(BinaryExpr), COMPARISON),
      RULE_ITEM(LESS_EQUAL   , nullptr        , M(BinaryExpr), COMPARISON),
      RULE_ITEM(IDENTIFIER   , M(VariableExpr), nullptr      , NONE),
      RULE_ITEM(STRING       , M(StringExpr)  , nullptr      , NONE),
      RULE_ITEM(NUMBER       , M(NumberExpr)  , nullptr      , NONE),
      RULE_ITEM(AND          , nullptr        , M(AndExpr)   , AND),
      RULE_ITEM(CLASS        , nullptr        , nullptr      , NONE),
      RULE_ITEM(ELSE         , nullptr        , nullptr      , NONE),
      RULE_ITEM(FALSE        , M(LiteralExpr) , nullptr      , NONE),
      RULE_ITEM(FOR          , nullptr        , nullptr      , NONE),
      RULE_ITEM(FUN          , nullptr        , nullptr      , NONE),
      RULE_ITEM(IF           , nullptr        , nullptr      , NONE),
      RULE_ITEM(NIL          , M(LiteralExpr) , nullptr      , NONE),
      RULE_ITEM(OR           , nullptr        , M(OrExpr)    , OR),
      RULE_ITEM(PRINT        , nullptr        , nullptr      , NONE),
      RULE_ITEM(RETURN       , nullptr        , nullptr      , NONE),
      RULE_ITEM(THIS         , M(ThisExpr)    , nullptr      , NONE),
      RULE_ITEM(TRUE         , M(LiteralExpr) , nullptr      , NONE),
      RULE_ITEM(VAR          , nullptr        , nullptr      , NONE),
      RULE_ITEM(WHILE        , nullptr        , nullptr      , NONE),
      RULE_ITEM(EOF_TOKEN    , nullptr        , nullptr      , NONE),
    };
  // clang-format on
#undef RULE_ITEM
#undef M

  std::vector<ParseRule> ret((int)TokenType::_TOKEN_COUNT_NUMBER);
  for (auto &pair : map_tmp) {
    ret[(int)pair.first] = pair.second;
  }
  RuleMap() = std::move(ret);
  return 1;
};

static int call_once = BuildRuleMap();

static std::vector<ParseRule> &RuleMap() {
  static std::vector<ParseRule> ret;
  static int unused = call_once;
  return ret;
}
static ParseRule *GetRule(TokenType type) { return &RuleMap()[(int)type]; }
static ParseRule *GetRule(Token token) { return GetRule(token->type); }

enum class ScopeType { UNKOWN, BLOCK, IF_ELSE, WHILE, FOR, FUNCTION };

/**
 * ScopeGuard is a helper to emit code that will be executed when scope ends
 */
struct ScopeGuard {
  ScopeGuard(Compiler *compiler, ScopeType type) : compiler(compiler), type(type) {
    auto cu = compiler->cu_;
    if (type == ScopeType::FOR || type == ScopeType::WHILE) {
      cu->loop_infos.push_back(
          FunctionUnit::LoopInfo{.initial_stack_size = static_cast<int>(compiler->cu_->locals.size())});
    }
    ++cu->semantic_scope_depth;
  }
  int OutOfScopeVar() {
    auto cu = compiler->cu_;
    int scope_var_count = 0;
    uint64_t N = cu->locals.size();
    while (cu->locals[N - scope_var_count - 1].semantic_scope_depth >= cu->semantic_scope_depth) {
      ++scope_var_count;
    }
    return scope_var_count;
  }
  ~ScopeGuard() {
    auto cu = compiler->cu_;
    if (type == ScopeType::FUNCTION) {
      return;  // function scope is cleared by a frame switch at runtime, all stack variables in the frame will be
               // discarded.
    }

    int var_num_of_just_out_of_scope = OutOfScopeVar();
    cu->CleanUpLocals(var_num_of_just_out_of_scope);

    if (type == ScopeType::FOR || type == ScopeType::WHILE) {
      for (auto &jmp : cu->loop_infos.back().breaks) {
        cu->JumpHerePatch(jmp);
      }  // before jumping here ,the locals had been cleared.
      cu->loop_infos.pop_back();
    }
    --cu->semantic_scope_depth;
  }
  Compiler *compiler;
  ScopeType type;
};

ObjFunction *Compiler::Compile(Scanner *scanner) {
  PushCU(FunctionType::SCRIPT, "<script>");

  scanner_ = scanner;
  Advance();
  while (!MatchAndAdvance(TokenType::EOF_TOKEN)) {
    AnyStatement();
  }
  auto top_level_cu = PopCU();
  if (had_error) {
    return nullptr;
  }
  return top_level_cu->func;
}
void Compiler::Advance() {
  previous = current;

  for (;;) {
    auto err = scanner_->ScanOne(&current);
    if (err.NoError()) break;

    ErrorAt(current, current->lexeme.c_str());
  }
}
void Compiler::ErrorAt(Token token, const char *message) {
  if (panic_mode) return;
  panic_mode = true;
  fprintf(stderr, "[line %d] Error", token->line);
  fprintf(stderr, " at '%.*s'", (int)token->lexeme.size(), token->lexeme.c_str());
  fprintf(stderr, ": %s\n", message);
  had_error = true;
}
void Compiler::Consume(TokenType type, const char *message) {
  if (current->type == type) {
    Advance();
    return;
  }
  ErrorAt(current, message);
}

void Compiler::AnyExpression(OperatorType operator_type) {
  auto bak_last_expression_precedence = last_expression_precedence;
  last_expression_precedence = operator_type;
  Precedence precedence = operator_type;
  Advance();
  auto EmitPrefixFn = GetRule(previous)->EmitPrefixFn;
  if (EmitPrefixFn == nullptr) {
    ErrorAt(previous, "Expect expression.");
    return;
  }

  EmitPrefixFn(this);

  while (precedence <= GetRule(current)->operator_type) {
    Advance();
    auto EmitInfixFn = GetRule(previous)->EmitInfixFn;
    EmitInfixFn(this);
  }
  last_expression_precedence = bak_last_expression_precedence;
}

void Compiler::UnaryExpr() {
  TokenType token_type = previous->type;
  // Compile the operand.
  AnyExpression(OperatorType::UNARY);
  // Emit the operator instruction.
  cu_->EmitUnary(token_type);
}

void Compiler::BinaryExpr() {
  TokenType token_type = previous->type;
  ParseRule *rule = GetRule(token_type);
  AnyExpression((OperatorType)((int)(rule->operator_type) + 1));

  cu_->EmitBinary(token_type);
}

void Compiler::LiteralExpr() { cu_->EmitLiteral(previous->type); }

void Compiler::StringExpr() {
  std::string tmp = previous->lexeme;
  *tmp.rbegin() = '\0';
  cu_->EmitConstant(Value(Symbol::Intern(tmp.c_str() + 1)));
}
bool Compiler::MatchAndAdvance(TokenType type) {
  if (!Check(type)) return false;
  Advance();
  return true;
}
void Compiler::AnyStatement() {
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
    BlockStmt();
  } else {
    ExpressionStmt();
  }
  if (panic_mode) Synchronize();
}
void Compiler::PrintStmt() {
  AnyExpression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  cu_->EmitByte(OpCode::OP_PRINT);
}
bool Compiler::Check(TokenType type) { return current->type == type; }
void Compiler::ExpressionStmt() {
  AnyExpression();
  Consume(TokenType::SEMICOLON, "Expect ';' after expression.");
  cu_->EmitByte(OpCode::OP_POP);
}
void Compiler::Synchronize() {
  panic_mode = false;

  while (current->type != TokenType::EOF_TOKEN) {
    if (previous->type == TokenType::SEMICOLON) return;
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

      default:;  // Do nothing.
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

void Compiler::VariableExpr() { GetOrSetNamedValue(previous, CanAssign()); }
void Compiler::GetOrSetNamedValue(Token varaible_token, bool can_assign) {
  /**
   * About UpValue:
   *
   * At runtime, If the closed value we want to access is still on the stack, we should access the value on stack. If
   * the closed value is not on stack, we should access the copied one. e.g. we define a closure, and then call it
   * immediately, we should access the raw value that is still on stack.
   *
   * To support this feature:
   *    1. compiler and virtual machine will have to work together.
   *    2. The closed value at runtime will be a pointer point to stack initially, and it will point to a new location
   *       when the stack position it points to is popped.
   *
   * Note that the global level function will not have any upvalue, all unresolved symbol in global level will go to
   * global directly. That is , the global level function is a closure that closes nothing.
   *
   * To a normal closure:
   *    1. The ObjFunction of closure is already created by compiler at compilation time, and we can do nothing to it.
   *       (It may contains some GET_UPVALUE, SET_UPVALUE op, but not very important.)
   *    2. There will always be a "outer function", act as the runtime creator of the closure.
   *
   * The ClosureCreator will do all the magic things at runtime to make the inner defined ObjFunction a real closure.
   *    1. At compile time, just after the inner function is defined (a `PopCU()` is executed), a `OP_CLOUSRE` will be
   * emitted. At runtime, `OP_CLOUSRE` will create a ObjClosure, and update the upvalues of it to make the upvalues
   * point to correct position, and for the upvalues that point to stack, we will track them as `opened-upvalues`, they
   * will get updated later.
   *    2. At compile time, normally, when a var out of scope, a OP_POP will be emit, but if it is needed by some inner
   * function, a `OP_CLOSE_UPVALUE` will be emitted, this OP will not only do the pop, but also do some upvalue updating
   *       job at runtime. At runtime, `OP_CLOSE_UPVALUE` will try to close all opened-upvalues that point to a invalid
   *       stack position. Normally, a `out of scope` will move the sp to correct place, and all value on stack after
   *       the correct place will be treated as `poped`, thus, it is very easy to determine whether a opened-upvalue
   * point to a invalid place. And also, OP_CLOSE_UPVALUE just do the `close`, it doesnt care which function the upvalue
   *       belongs to.
   *
   * Note that:
   *    1. the truth that closed values are get "copied" (or captured, closed) when they went out of scope, it may cause
   * some logical misleading, e.g. creating closure in loops will make every closure closed to the same value.
   *    2. we will also support the chain style upvalue, if a nested closure is created, and they need to access same
   *    variable, then same value will always be used at runtime.
   *
   *
   * To support the code gen of `OP_CLOUSRE` and `OP_CLOSE_UPVALUE`, we need to track the upvalue at compile time. And
   * all these tracking will be done by the TryResolveUpValue, obviously, upvalue is only known by compiler, user code
   * will not decl/define upvalue.
   */
  OpCode getOp, setOp;
  FunctionUnit::NamedValue *reslove = nullptr;
  if ((reslove = cu_->TryResolveLocal(varaible_token))) {
    getOp = OpCode::OP_GET_LOCAL;
    setOp = OpCode::OP_SET_LOCAL;
    assert(reslove->position < STACK_LOOKUP_MAX);
  } else if ((reslove = cu_->TryResolveUpValue(varaible_token))) {
    getOp = OpCode::OP_GET_UPVALUE;
    setOp = OpCode::OP_SET_UPVALUE;
    assert(reslove->position < UPVALUE_LOOKUP_MAX);
  } else {
    reslove = cu_->TryResolveGlobal(varaible_token);
    getOp = OpCode::OP_GET_GLOBAL;
    setOp = OpCode::OP_SET_GLOBAL;
    assert(reslove->position < GLOBAL_LOOKUP_MAX);
  }
  if (!reslove) {
    ErrorAt(varaible_token, "Undefined variable.");
  }
  if (!reslove->is_inited) {
    ErrorAt(previous, "Can not use uninitialized variable here.");
  }
  if (MatchAndAdvance(TokenType::EQUAL)) {
    if (can_assign) {
      AnyExpression();
      cu_->EmitBytes(setOp, reslove->position);
    } else {
      ErrorAt(previous, "Invalid assignment target.");
    }
  } else {
    cu_->EmitBytes(getOp, reslove->position);
  }
}
bool Compiler::CanAssign() { return last_expression_precedence <= Precedence::ASSIGNMENT; }
void Compiler::BlockStmt() {
  ScopeGuard guard(this, ScopeType::BLOCK);
  while (!Check(TokenType::RIGHT_BRACE) && !Check(TokenType::EOF_TOKEN)) {
    AnyStatement();
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after BlockStmt.");
}

void Compiler::IfStmt() {
  ScopeGuard guard(this, ScopeType::IF_ELSE);
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'if'.");
  AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");

  auto jump_to_else = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
  AnyStatement();
  auto jump_to_exit = cu_->CreateJumpDownHole(OpCode::OP_JUMP);
  cu_->JumpHerePatch(jump_to_else);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
  if (MatchAndAdvance(TokenType::ELSE)) {
    AnyStatement();
  }
  cu_->JumpHerePatch(jump_to_exit);
}

void Compiler::AndExpr() {
  auto endJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);

  cu_->EmitByte(OpCode::OP_POP);
  AnyExpression(Precedence::AND);

  cu_->JumpHerePatch(endJump);
}
void Compiler::OrExpr() {
  auto elseJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
  auto endJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP);

  cu_->JumpHerePatch(elseJump);
  cu_->EmitByte(OpCode::OP_POP);

  AnyExpression(Precedence::OR);
  cu_->JumpHerePatch(endJump);
}
void Compiler::WhileStmt() {
  ScopeGuard guard(this, ScopeType::WHILE);
  int loop_begin_offset = cu_->Chunk()->ChunkSize();
  cu_->loop_infos.back().beg_offset =
      loop_begin_offset;  // save the loop begin offset, so we can `continue` to here later
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'while'.");
  AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");

  auto exitJump = cu_->CreateJumpDownHole(OpCode::OP_JUMP_IF_FALSE);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
  AnyStatement();
  cu_->EmitJumpBack(loop_begin_offset);
  cu_->JumpHerePatch(exitJump);
  cu_->EmitByte(OpCode::OP_POP);  // discard the condition
}

void Compiler::ForStmt() {
  ScopeGuard guard(this, ScopeType::FOR);
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
  AnyStatement();
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
    return;
  }
  int clear_size = cu_->locals.size() - cu_->loop_infos.back().initial_stack_size;
  if (previous->type == TokenType::BREAK) {
    cu_->CleanUpLocals(clear_size);
    auto jump = cu_->CreateJumpDownHole(OpCode::OP_JUMP);
    cu_->loop_infos.back().breaks.push_back(jump);
  } else if (previous->type == TokenType::CONTINUE) {
    cu_->CleanUpLocals(clear_size -
                       cu_->loop_infos.back().contains_init_value);  // a continue may need to keep the init value
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
  named_value->is_inited = true;
  // a hack to make the function name could be resolved later in function definition, to support recursion
  // this impl is limited for we only have a 1 token ahead compiler
  CreateFunc(FunctionType::FUNCTION);
  cu_->DefineNamedValue(named_value);
}
void Compiler::CreateFunc(FunctionType type) {
  PushCU(type, previous->lexeme);
  ScopeGuard guard(this, ScopeType::FUNCTION);  // function and method share the same scope type

  Consume(TokenType::LEFT_PAREN, "Expect '(' after function name.");
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      cu_->func->arity++;
      if (cu_->func->arity > (ARG_COUNT_MAX - cu_->locals.size())) {
        ErrorAt(current, "Stack can not hold this much arguments");
      }
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
  cu_->EmitBytes(OpCode::OP_CLOSURE, cu_->MakeConstant(Value(new_cu->func)));
  assert(new_cu->upvalues.size() < UPVALUE_LOOKUP_MAX);
  cu_->EmitByte((uint8_t)new_cu->upvalues.size());
  for (auto &upvalue : new_cu->upvalues) {
    cu_->EmitByte(upvalue.is_on_stack_at_begin ? 1 : 0);
    cu_->EmitByte(upvalue.position);
  }
}
void Compiler::CallExpr() {
  uint8_t argCount = ArgumentList();
  cu_->EmitBytes(OpCode::OP_CALL, argCount);
}
uint8_t Compiler::ArgumentList() {
  uint8_t argCount = 0;
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      AnyExpression();
      if (argCount == 255) {
        ErrorAt(previous, "Can't have more than 255 arguments.");
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
void Compiler::MarkRoots(void *compiler_p) {
  Compiler *compiler = static_cast<Compiler *>(compiler);
  // RecursiveMark functions
  auto cu = compiler->cu_;
  while (cu) {
    GC::Instance().RecursiveMark(cu->func);
    cu = cu->enclosing_;
  }
}
Compiler::Compiler() : marker_register_guard(&MarkRoots, this) {}
void Compiler::ClassDefStmt() {
  Consume(TokenType::IDENTIFIER, "Expect class name.");
  Token className = previous;
  auto handle = cu_->DeclNamedValue(className);

  cu_->EmitBytes(OpCode::OP_CLASS, handle->position);  // create a objClass on stack
  cu_->DefineNamedValue(
      handle);  // most class def are in global scope, so here we will move the created class obj to global at runtime.

  ClassLevel nest_class(currentClass);
  currentClass = &nest_class;

  if (MatchAndAdvance(TokenType::LESS)) {
    Consume(TokenType::IDENTIFIER, "Expect superclass name.");
    Token superclass_name = previous;
    GetNamedValue(superclass_name);
    if (className->lexeme == superclass_name->lexeme) {
      ErrorAt(superclass_name, "A class can't inherit from itself.");
    }
    currentClass->hasSuperclass = true;
    GetNamedValue(className);
    cu_->EmitByte(OpCode::OP_INHERIT);  // inherit will consume the two class object on stack, and leaves nothing on
                                        // stack, so stmt rule will be followed
  }

  GetNamedValue(className);  // put the class object on stack, so later method define could use it
  Consume(TokenType::LEFT_BRACE, "Expect '{' before class body.");
  while (!Check(TokenType::RIGHT_BRACE) && !Check(TokenType::EOF_TOKEN)) {
    Consume(TokenType::IDENTIFIER, "Expect Method name.");
    uint8_t Constant = cu_->StoreTokenLexmeToConstant(previous);
    FunctionType Type = FunctionType::METHOD;
    if (previous->lexeme == "init") {
      Type = FunctionType::INITIALIZER;
    }
    CreateFunc(Type);
    cu_->EmitBytes(OpCode::OP_METHOD, Constant);  // just move the objClosure on stack to the class object's dict
  }
  Consume(TokenType::RIGHT_BRACE, "Expect '}' after class body.");
  cu_->EmitByte(OpCode::OP_POP);  // pop the class object, for class def is always a stmt.
  currentClass = currentClass->enclosing;
}
void Compiler::DotExpr() {
  // todo : check only method can use class method at compile time
  Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
  uint8_t name = cu_->StoreTokenLexmeToConstant(previous);

  if (CanAssign() && MatchAndAdvance(TokenType::EQUAL)) {
    AnyExpression();
    cu_->EmitBytes(OpCode::OP_SET_ATTR, name);
  } else if (MatchAndAdvance(TokenType::LEFT_PAREN)) {
    uint8_t argCount = ArgumentList();
    cu_->EmitBytes(OpCode::OP_INVOKE, name);
    cu_->EmitByte(argCount);
  } else {
    cu_->EmitBytes(OpCode::OP_GET_ATTR, name);
  }
}
void Compiler::ThisExpr() {
  if (currentClass == nullptr) {
    ErrorAt(previous, "Can't use 'this' outside of a class.");
    return;
  }
  GetNamedValue(previous);
}
void Compiler::GroupingExpr() {
  AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
}
void Compiler::NumberExpr() {
  double value = std::stod(previous->lexeme);
  cu_->EmitConstant(Value(value));
}

void Compiler::PushCU(FunctionType type, const std::string &name) {
  cu_ = new FunctionUnit(cu_, type, name, [this]() { return this->previous->line; });
}
std::unique_ptr<FunctionUnit> Compiler::PopCU() {
  auto old_cu = cu_;
  cu_->EmitDefaultReturn();  // always inject a default return to make sure the function ends
#ifndef NDEBUG
  DumpChunkCode(cu_->Chunk());
  DumpChunkConstant(cu_->Chunk());
#endif
  cu_ = cu_->enclosing_;
  return std::unique_ptr<FunctionUnit>(old_cu);
}
void Compiler::GetNamedValue(Token name) { return GetOrSetNamedValue(name, false); }

}  // namespace lox::vm
