//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/compiler.h"

#include <map>

namespace lox {

namespace vm {
ObjFunction *Compiler::Compile(Scanner *scanner) {
  CompileUnit top_level_cu(CompileUnit::CUType::SCRIPT, "<script>");
  // the function in top_level_cu will be pushed to stack at runtime, so locals[0] is occupied here
  auto &local = top_level_cu.locals[top_level_cu.localCount++];
  local.depth = 0;
  local.name = MakeToken(TokenType::EOF_TOKEN, "<script>", 0);

  current_cu_ = &top_level_cu;
  scanner_ = scanner;
  Advance();
  while (!MatchAndAdvance(TokenType::EOF_TOKEN)) {
    declaration();
  }
  endCompiler();
  if (parser_.hadError) {
    return nullptr;
  }
#ifndef NDEBUG
  if (!parser_.hadError) {
    CurrentChunk()->DumpCode(current_cu_->func->name.c_str());
    CurrentChunk()->DumpConstant();
  }
#endif
  return top_level_cu.func;
}
void Compiler::Advance() {
  parser_.previous = parser_.current;

  for (;;) {
    auto err = scanner_->ScanOne(&parser_.current);
    if (err.NoError()) break;

    errorAtCurrent(parser_.current->lexeme.c_str());
  }
}
void Compiler::errorAtCurrent(const char *message) { errorAt(parser_.current, message); }
void Compiler::errorAt(Token token, const char *message) {
  if (parser_.panicMode) return;
  parser_.panicMode = true;
  fprintf(stderr, "[line %d] Error", token->line);
  fprintf(stderr, " at '%.*s'", (int)token->lexeme.size(), token->lexeme.c_str());
  fprintf(stderr, ": %s\n", message);
  parser_.hadError = true;
}
void Compiler::Consume(TokenType type, const char *message) {
  if (parser_.current->type == type) {
    Advance();
    return;
  }

  errorAtCurrent(message);
}
Chunk *Compiler::CurrentChunk() { return current_cu_->func->chunk; }
void Compiler::endCompiler() { emitReturn(); }
void Compiler::emitReturn() { emitByte(OpCode::OP_RETURN); }
void Compiler::error(const char *message) { errorAt(parser_.previous, message); }
void Compiler::Expression(OperatorType operator_type) {
  auto bak_last_expression_precedence = last_expression_precedence;
  last_expression_precedence = operator_type;
  Precedence precedence = operator_type;
  Advance();
  auto EmitPrefixFn = getRule(parser_.previous)->EmitPrefixFn;
  if (EmitPrefixFn == nullptr) {
    error("Expect expression.");
    return;
  }

  EmitPrefixFn(this);

  while (precedence <= getRule(parser_.current)->operator_type) {
    Advance();
    auto EmitInfixFn = getRule(parser_.previous)->EmitInfixFn;
    EmitInfixFn(this);
  }
  last_expression_precedence = bak_last_expression_precedence;
}

std::vector<ParseRule> BuildRuleMap() {
#define RULE_ITEM(TOKEN_T, MEMBER_FN0, MEMBER_FN1, PRECEDENCE_V)                         \
  {                                                                                      \
    TokenType::TOKEN_T, ParseRule { MEMBER_FN0, MEMBER_FN1, OperatorType::PRECEDENCE_V } \
  }
#define M(MEMBER_FN) &Compiler::MEMBER_FN
  // clang-format off
    static std::map<TokenType,ParseRule> rules_map    = {
/*     TokenType ,         PrefixEmitFn , InfixEmitFn, OperatorType */
      RULE_ITEM(LEFT_PAREN   , M(grouping), nullptr  , NONE),
      RULE_ITEM(RIGHT_PAREN  , nullptr    , nullptr  , NONE),
      RULE_ITEM(LEFT_BRACE   , nullptr    , nullptr  , NONE),
      RULE_ITEM(RIGHT_BRACE  , nullptr    , nullptr  , NONE),
      RULE_ITEM(COMMA        , nullptr    , nullptr  , NONE),
      RULE_ITEM(DOT          , nullptr    , nullptr  , NONE),
      RULE_ITEM(MINUS        , M(unary)   , M(binary), TERM),
      RULE_ITEM(PLUS         , nullptr    , M(binary), TERM),
      RULE_ITEM(SEMICOLON    , nullptr    , nullptr  , NONE),
      RULE_ITEM(SLASH        , nullptr    , M(binary), FACTOR),
      RULE_ITEM(STAR         , nullptr    , M(binary), FACTOR),
      RULE_ITEM(BANG         , M(unary)    , nullptr  , NONE),
      RULE_ITEM(BANG_EQUAL   , nullptr    , M(binary)  , EQUALITY),
      RULE_ITEM(EQUAL        , nullptr    , nullptr  , NONE),
      RULE_ITEM(EQUAL_EQUAL  , nullptr    , M(binary)  , EQUALITY),
      RULE_ITEM(GREATER      , nullptr    , M(binary)  , COMPARISON),
      RULE_ITEM(GREATER_EQUAL, nullptr    , M(binary)  , COMPARISON),
      RULE_ITEM(LESS         , nullptr    , M(binary)  , COMPARISON),
      RULE_ITEM(LESS_EQUAL   , nullptr    , M(binary)  , COMPARISON),
      RULE_ITEM(IDENTIFIER   , M(variable)    , nullptr  , NONE),
      RULE_ITEM(STRING       , M(string)    , nullptr  , NONE),
      RULE_ITEM(NUMBER       , M(number)  , nullptr  , NONE),
      RULE_ITEM(AND          , nullptr    , M(and_)  , AND),
      RULE_ITEM(CLASS        , nullptr    , nullptr  , NONE),
      RULE_ITEM(ELSE         , nullptr    , nullptr  , NONE),
      RULE_ITEM(FALSE        , M(literal)    , nullptr  , NONE),
      RULE_ITEM(FOR          , nullptr    , nullptr  , NONE),
      RULE_ITEM(FUN          , nullptr    , nullptr  , NONE),
      RULE_ITEM(IF           , nullptr    , nullptr  , NONE),
      RULE_ITEM(NIL          , M(literal)    , nullptr  , NONE),
      RULE_ITEM(OR           , nullptr    , M(or_)  , OR),
      RULE_ITEM(PRINT        , nullptr    , nullptr  , NONE),
      RULE_ITEM(RETURN       , nullptr    , nullptr  , NONE),
      RULE_ITEM(THIS         , nullptr    , nullptr  , NONE),
      RULE_ITEM(TRUE         , M(literal)    , nullptr  , NONE),
      RULE_ITEM(VAR          , nullptr    , nullptr  , NONE),
      RULE_ITEM(WHILE        , nullptr    , nullptr  , NONE),
      RULE_ITEM(EOF_TOKEN    , nullptr    , nullptr  , NONE),
    };
  // clang-format on
#undef RULE_ITEM
#undef M
  std::vector<ParseRule> ret((int)TokenType::_TOKEN_COUNT_NUMBER);
  for (auto &pair : rules_map) {
    ret[(int)pair.first] = pair.second;
  }
  return ret;
};
ParseRule *Compiler::getRule(TokenType type) {
  static std::vector<ParseRule> rules = BuildRuleMap();
  return &rules[(int)type];
}
void Compiler::emitBytes(OpCode byte1, uint8_t byte2) {
  emitByte(byte1);
  emitByte(byte2);
}
uint8_t Compiler::makeConstant(Value value) {
  int constant = CurrentChunk()->addConstant(value);
  if (constant > UINT8_MAX) {
    error("Too many constants in one chunk.");
    return 0;
  }

  return (uint8_t)constant;
}
void Compiler::unary() {
  TokenType token_type = parser_.previous->type;

  // Compile the operand.
  Expression(OperatorType::UNARY);

  // Emit the operator instruction.
  switch (token_type) {
    case TokenType::MINUS:
      emitByte(OpCode::OP_NEGATE);
      break;
    case TokenType::BANG:
      emitByte(OpCode::OP_NOT);
      break;
    default:
      return;  // Unreachable.
  }
}
void Compiler::binary() {
  TokenType token_type = parser_.previous->type;
  ParseRule *rule = getRule(token_type);
  Expression((OperatorType)((int)(rule->operator_type) + 1));

  switch (token_type) {
    case TokenType::BANG_EQUAL:
      emitBytes(OpCode::OP_EQUAL, OpCode::OP_NOT);
      break;
    case TokenType::EQUAL_EQUAL:
      emitByte(OpCode::OP_EQUAL);
      break;
    case TokenType::GREATER:
      emitByte(OpCode::OP_GREATER);
      break;
    case TokenType::GREATER_EQUAL:
      emitBytes(OpCode::OP_LESS, OpCode::OP_NOT);
      break;
    case TokenType::LESS:
      emitByte(OpCode::OP_LESS);
      break;
    case TokenType::LESS_EQUAL:
      emitBytes(OpCode::OP_GREATER, OpCode::OP_NOT);
      break;
    case TokenType::PLUS:
      emitByte(OpCode::OP_ADD);
      break;
    case TokenType::MINUS:
      emitByte(OpCode::OP_SUBTRACT);
      break;
    case TokenType::STAR:
      emitByte(OpCode::OP_MULTIPLY);
      break;
    case TokenType::SLASH:
      emitByte(OpCode::OP_DIVIDE);
      break;
    default:
      return;  // Unreachable.
  }
}
ParseRule *Compiler::getRule(Token token) { return getRule(token->type); }
void Compiler::literal() {
  switch (parser_.previous->type) {
    case TokenType::FALSE:
      emitByte(OpCode::OP_FALSE);
      break;
    case TokenType::NIL:
      emitByte(OpCode::OP_NIL);
      break;
    case TokenType::TRUE:
      emitByte(OpCode::OP_TRUE);
      break;
    default:
      return;  // Unreachable.
  }
}
void Compiler::string() {
  emitConstant(
      Value(ObjInternedString::Make(parser_.previous->lexeme.c_str() + 1, parser_.previous->lexeme.size() - 2)));
}
bool Compiler::MatchAndAdvance(TokenType type) {
  if (!Check(type)) return false;
  Advance();
  return true;
}
void Compiler::declaration() {
  if (MatchAndAdvance(TokenType::VAR)) {
    varDeclaration();
  } else {
    statement();
  }
  if (parser_.panicMode) synchronize();
}
void Compiler::statement() {
  if (MatchAndAdvance(TokenType::PRINT)) {
    printStatement();
  } else if (MatchAndAdvance(TokenType::BREAK)) {
    breakStatement();
  } else if (MatchAndAdvance(TokenType::IF)) {
    beginScope(ScopeType::IF_ELSE);
    ifStatement();
    endScope(ScopeType::IF_ELSE);
  } else if (MatchAndAdvance(TokenType::WHILE)) {
    beginScope(ScopeType::WHILE);
    whileStatement();
    endScope(ScopeType::WHILE);
  } else if (MatchAndAdvance(TokenType::FOR)) {
    beginScope(ScopeType::FOR);
    forStatement();
    endScope(ScopeType::FOR);
  } else if (MatchAndAdvance(TokenType::LEFT_BRACE)) {
    beginScope(ScopeType::BLOCK);
    block();
    endScope(ScopeType::BLOCK);
  } else {
    expressionStatement();
  }
}
void Compiler::printStatement() {
  Expression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  emitByte(OpCode::OP_PRINT);
}
bool Compiler::Check(TokenType type) { return parser_.current->type == type; }
void Compiler::expressionStatement() {
  Expression();
  Consume(TokenType::SEMICOLON, "Expect ';' after expression.");
  emitByte(OpCode::OP_POP);
}
void Compiler::synchronize() {
  parser_.panicMode = false;

  while (parser_.current->type != TokenType::EOF_TOKEN) {
    if (parser_.previous->type == TokenType::SEMICOLON) return;
    switch (parser_.current->type) {
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
void Compiler::varDeclaration() {
  uint8_t global = parseVariable("Expect variable name.");

  if (MatchAndAdvance(TokenType::EQUAL)) {
    Expression();
  } else {
    emitByte(OpCode::OP_NIL);
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after variable declaration.");

  defineVariable(global);
}
uint8_t Compiler::parseVariable(const char *errorMessage) {
  Consume(TokenType::IDENTIFIER, errorMessage);
  declareVariable();
  if (current_cu_->scopeDepth > 0) return 0;
  return identifierConstant(parser_.previous);
}
uint8_t Compiler::identifierConstant(Token token) {
  return makeConstant(Value(ObjInternedString::Make(token->lexeme.c_str(), token->lexeme.size())));
}
void Compiler::defineVariable(uint8_t global) {
  if (current_cu_->scopeDepth > 0) {
    markInitialized();
    return;
  }
  emitBytes(OpCode::OP_DEFINE_GLOBAL, global);
}
void Compiler::variable() { namedVariable(parser_.previous); }
void Compiler::namedVariable(Token varaible_token) {
  OpCode getOp, setOp;
  int arg = resolveLocal(varaible_token);
  if (arg != -1) {
    getOp = OpCode::OP_GET_LOCAL;
    setOp = OpCode::OP_SET_LOCAL;
  } else {
    arg = identifierConstant(varaible_token);
    getOp = OpCode::OP_GET_GLOBAL;
    setOp = OpCode::OP_SET_GLOBAL;
  }
  if (MatchAndAdvance(TokenType::EQUAL)) {
    if (canAssign()) {
      Expression();
      emitBytes(setOp, arg);
    } else {
      error("Invalid assignment target.");
    }
  } else {
    emitBytes(getOp, arg);
  }
}
bool Compiler::canAssign() { return last_expression_precedence <= Precedence::ASSIGNMENT; }
void Compiler::block() {
  while (!Check(TokenType::RIGHT_BRACE) && !Check(TokenType::EOF_TOKEN)) {
    declaration();
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after block.");
}
void Compiler::beginScope(ScopeType type) {
  if (type == ScopeType::FOR || type == ScopeType::WHILE) {
    openBreak();
  }
  ++current_cu_->scopeDepth;
}
void Compiler::endScope(ScopeType type) {
  int scope_var_num = updateScopeCount();
  if (loop_nest_level != -1) {
    // we are in some loop, so there might be breaks, there will be a branch at runtime

    // if vm goes here directly, that means scope ended normally, we just clear scope and go on
    for (int i = 0; i < scope_var_num; ++i) {
      emitByte(OpCode::OP_POP);
    }
    auto offset = emitJumpDown(OpCode::OP_JUMP);
    // if vm goes here, it means vm jumped to here by some break, we clear scope, and tries to jump to next endScope
    patchBreaks(loop_nest_level);
    for (int i = 0; i < scope_var_num; ++i) {
      emitByte(OpCode::OP_POP);
    }
    if (type == ScopeType::FOR || type == ScopeType::WHILE) {
      // if the scope is an end of for loop, we just close it and go on
      closeBreak();
    } else {
      // else, we jump to next endScope
      createBreakJump(loop_nest_level);
    }
    patchJumpDown(offset);
  } else {
    for (int i = 0; i < scope_var_num; ++i) {
      emitByte(OpCode::OP_POP);
    }
  }
  --current_cu_->scopeDepth;
}
int Compiler::updateScopeCount() {
  int scope_var_count = 0;
  while (current_cu_->localCount > 0 &&
         current_cu_->locals[current_cu_->localCount - 1].depth > current_cu_->scopeDepth) {
    current_cu_->localCount--;
    ++scope_var_count;
  }
  return scope_var_count;
}
void Compiler::declareVariable() {
  if (current_cu_->scopeDepth == 0) return;

  Token name = parser_.previous;
  for (int i = current_cu_->localCount - 1; i >= 0; i--) {
    auto local = &current_cu_->locals[i];
    if (local->depth != -1 && local->depth < current_cu_->scopeDepth) {
      break;
    }

    if (identifiersEqual(name, local->name)) {
      error("Already a variable with this name in this scope.");
    }
  }
  addLocal(name);
}
void Compiler::addLocal(Token token) {
  if (current_cu_->localCount == STACK_LOOKUP_OFFSET_MAX) {
    error("Too many local variables in function.");
    return;
  }
  CompileUnit::Local *local = &current_cu_->locals[current_cu_->localCount++];
  local->name = token;
  local->depth = -1;
}
bool Compiler::identifiersEqual(Token t0, Token t1) { return t0->lexeme == t1->lexeme; }
int Compiler::resolveLocal(Token token) {
  for (int i = current_cu_->localCount - 1; i >= 0; i--) {
    CompileUnit::Local *local = &current_cu_->locals[i];
    if (identifiersEqual(token, local->name)) {
      if (local->depth == -1) {
        error("Can't read local variable in its own initializer.");
      }
      return i;
    }
  }

  return -1;
}
void Compiler::markInitialized() { current_cu_->locals[current_cu_->localCount - 1].depth = current_cu_->scopeDepth; }
void Compiler::ifStatement() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'if'.");
  Expression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");

  int thenJump = emitJumpDown(OpCode::OP_JUMP_IF_FALSE);
  emitByte(OpCode::OP_POP);
  statement();
  int elseJump = emitJumpDown(OpCode::OP_JUMP);
  patchJumpDown(thenJump);
  emitByte(OpCode::OP_POP);
  if (MatchAndAdvance(TokenType::ELSE)) {
    statement();
  }
  patchJumpDown(elseJump);
}
int Compiler::emitJumpDown(OpCode jump_cmd) {
  emitByte(jump_cmd);
  // reserve two bytes to store jmp diff
  emitByte(0xff);
  emitByte(0xff);
  return CurrentChunk()->ChunkSize() - 2;
}
void Compiler::patchJumpDown(int offset) {
  int ip_from = offset + 2;
  // at runtime, jump will load 1 byte of OPCode and 2 byte of position, so ip will pointer to `BASE + offset + 2`
  int ip_target = CurrentChunk()->ChunkSize();
  int jump_diff = ip_target - ip_from;

  if (jump_diff > UINT16_MAX) {
    error("Too much code to jump_diff over.");
  }

  CurrentChunk()->code[offset] = (jump_diff >> 8) & 0xff;
  CurrentChunk()->code[offset + 1] = jump_diff & 0xff;
}
void Compiler::and_() {
  int endJump = emitJumpDown(OpCode::OP_JUMP_IF_FALSE);

  emitByte(OpCode::OP_POP);
  Expression(Precedence::AND);

  patchJumpDown(endJump);
}
void Compiler::or_() {
  int elseJump = emitJumpDown(OpCode::OP_JUMP_IF_FALSE);
  int endJump = emitJumpDown(OpCode::OP_JUMP);

  patchJumpDown(elseJump);
  emitByte(OpCode::OP_POP);

  Expression(Precedence::OR);
  patchJumpDown(endJump);
}
void Compiler::whileStatement() {
  int loopStart = CurrentChunk()->ChunkSize();
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'while'.");
  Expression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");

  int exitJump = emitJumpDown(OpCode::OP_JUMP_IF_FALSE);
  emitByte(OpCode::OP_POP);
  statement();
  emitJumpBack(loopStart);
  patchJumpDown(exitJump);
  emitByte(OpCode::OP_POP);
}
void Compiler::emitJumpBack(int start) {
  int ip_target = start;
  int ip_from = CurrentChunk()->ChunkSize() + 3;  // after OP_JUMP_BACK is executed, ip will pointer to this pos

  emitByte(OpCode::OP_JUMP_BACK);
  int offset = -1 * (ip_target - ip_from);  // always use a positive number, for we store offset into a uint16
  if (offset > UINT16_MAX) error("Loop body too large.");

  emitByte((offset >> 8) & 0xff);
  emitByte(offset & 0xff);
}
void Compiler::forStatement() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'for'.");
  if (MatchAndAdvance(TokenType::SEMICOLON)) {
    // No initializer.
  } else if (MatchAndAdvance(TokenType::VAR)) {
    varDeclaration();
  } else {
    expressionStatement();
  }

  int loopStart = CurrentChunk()->ChunkSize();
  int exitJump = -1;
  if (!MatchAndAdvance(TokenType::SEMICOLON)) {
    Expression();
    Consume(TokenType::SEMICOLON, "Expect ';' after loop condition.");

    // Jump out of the loop if the condition is false.
    exitJump = emitJumpDown(OpCode::OP_JUMP_IF_FALSE);
    emitByte(OpCode::OP_POP);  // Condition.
  }

  if (!MatchAndAdvance(TokenType::RIGHT_PAREN)) {
    int bodyJump = emitJumpDown(OpCode::OP_JUMP);
    int incrementStart = CurrentChunk()->ChunkSize();
    Expression();              // this will leave a value on top of stack
    emitByte(OpCode::OP_POP);  // discard stack top
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after for clauses.");

    emitJumpBack(loopStart);
    loopStart = incrementStart;
    patchJumpDown(bodyJump);
  }
  statement();
  emitJumpBack(loopStart);
  if (exitJump != -1) {
    patchJumpDown(exitJump);
    emitByte(OpCode::OP_POP);  // Condition.
  }
}
void Compiler::breakStatement() {
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  if (loop_nest_level < 0) {
    error("Can not break here");
    return;
  }
  createBreakJump(loop_nest_level);
}
void Compiler::createBreakJump(int level) {
  auto tail = &loop_break_info;
  while (tail->next) {
    tail = tail->next;
  }
  int jump = emitJumpDown(OpCode::OP_JUMP);
  tail->next = new LoopBreak;
  tail->next->offset = jump;
  tail->next->level = level;
}
void Compiler::openBreak() { ++loop_nest_level; }
void Compiler::closeBreak() { --loop_nest_level; }
void Compiler::patchBreaks(int level) {
  if (level < 0) {
    return;
  }
  auto p = &loop_break_info;
  auto new_tail = &loop_break_info;
  while (p && p->level != level) {
    new_tail = p;
    p = p->next;
  }
  while (p) {
    patchJumpDown(p->offset);
    auto tmp = p;
    p = p->next;
    delete tmp;
  }
  new_tail->next = nullptr;
}
}  // namespace vm
}  // namespace lox
