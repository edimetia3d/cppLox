//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/compiler.h"

#include <map>

namespace lox {

namespace vm {
ErrCode Compiler::Compile(Scanner *scanner, Chunk *target) {
  scanner_ = scanner;
  current_trunk_ = target;
  Advance();
  Expression(OperatorType::ASSIGNMENT);
  Consume(TokenType::EOF_TOKEN, "Expect end of expression.");
  endCompiler();
  if (parser_.hadError) {
    return ErrCode::PARSE_FAIL;
  }
  return ErrCode::NO_ERROR;
}
void Compiler::Advance() {
  parser_.current = parser_.next;

  for (;;) {
    auto err = scanner_->ScanOne(&parser_.next);
    if (err.NoError()) break;

    errorAtCurrent(parser_.next->lexeme.c_str());
  }
}
void Compiler::errorAtCurrent(const char *message) { errorAt(parser_.next, message); }
void Compiler::errorAt(Token token, const char *message) {
  if (parser_.panicMode) return;
  parser_.panicMode = true;
  fprintf(stderr, "[line %d] Error", token->line);
  fprintf(stderr, " at '%.*s'", (int)token->lexeme.size(), token->lexeme.c_str());
  fprintf(stderr, ": %s\n", message);
  parser_.hadError = true;
}
void Compiler::Consume(TokenType type, const char *message) {
  if (parser_.next->type == type) {
    Advance();
    return;
  }

  errorAtCurrent(message);
}
Chunk *Compiler::CurrentChunk() { return current_trunk_; }
void Compiler::endCompiler() { emitReturn(); }
void Compiler::emitReturn() { emitOpCode(OpCode::OP_RETURN); }
void Compiler::error(const char *message) { errorAt(parser_.current, message); }
void Compiler::Expression(OperatorType operator_type) {
  Precedence precedence = operator_type;
  Advance();
  auto EmitPrefixFn = getRule(parser_.current)->EmitPrefixFn;
  if (EmitPrefixFn == nullptr) {
    error("Expect expression.");
    return;
  }

  EmitPrefixFn(this);

  while (precedence <= getRule(parser_.next)->operator_type) {
    Advance();
    auto EmitInfixFn = getRule(parser_.current)->EmitInfixFn;
    EmitInfixFn(this);
  }
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
      RULE_ITEM(BANG         , nullptr    , nullptr  , NONE),
      RULE_ITEM(BANG_EQUAL   , nullptr    , nullptr  , NONE),
      RULE_ITEM(EQUAL        , nullptr    , nullptr  , NONE),
      RULE_ITEM(EQUAL_EQUAL  , nullptr    , nullptr  , NONE),
      RULE_ITEM(GREATER      , nullptr    , nullptr  , NONE),
      RULE_ITEM(GREATER_EQUAL, nullptr    , nullptr  , NONE),
      RULE_ITEM(LESS         , nullptr    , nullptr  , NONE),
      RULE_ITEM(LESS_EQUAL   , nullptr    , nullptr  , NONE),
      RULE_ITEM(IDENTIFIER   , nullptr    , nullptr  , NONE),
      RULE_ITEM(STRING       , nullptr    , nullptr  , NONE),
      RULE_ITEM(NUMBER       , M(number)  , nullptr  , NONE),
      RULE_ITEM(AND          , nullptr    , nullptr  , NONE),
      RULE_ITEM(CLASS        , nullptr    , nullptr  , NONE),
      RULE_ITEM(ELSE         , nullptr    , nullptr  , NONE),
      RULE_ITEM(FALSE        , nullptr    , nullptr  , NONE),
      RULE_ITEM(FOR          , nullptr    , nullptr  , NONE),
      RULE_ITEM(FUN          , nullptr    , nullptr  , NONE),
      RULE_ITEM(IF           , nullptr    , nullptr  , NONE),
      RULE_ITEM(NIL          , nullptr    , nullptr  , NONE),
      RULE_ITEM(OR           , nullptr    , nullptr  , NONE),
      RULE_ITEM(PRINT        , nullptr    , nullptr  , NONE),
      RULE_ITEM(RETURN       , nullptr    , nullptr  , NONE),
      RULE_ITEM(THIS         , nullptr    , nullptr  , NONE),
      RULE_ITEM(TRUE         , nullptr    , nullptr  , NONE),
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
  emitOpCode(byte1);
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
  TokenType token_type = parser_.current->type;

  // Compile the operand.
  Expression(OperatorType::NONE);

  // Emit the operator instruction.
  switch (token_type) {
    case TokenType::MINUS:
      emitOpCode(OpCode::OP_NEGATE);
      break;
    default:
      return;  // Unreachable.
  }
}
void Compiler::binary() {
  TokenType token_type = parser_.current->type;
  ParseRule *rule = getRule(token_type);
  Expression((OperatorType)((int)(rule->operator_type) + 1));

  switch (token_type) {
    case TokenType::PLUS:
      emitOpCode(OpCode::OP_ADD);
      break;
    case TokenType::MINUS:
      emitOpCode(OpCode::OP_SUBTRACT);
      break;
    case TokenType::STAR:
      emitOpCode(OpCode::OP_MULTIPLY);
      break;
    case TokenType::SLASH:
      emitOpCode(OpCode::OP_DIVIDE);
      break;
    default:
      return;  // Unreachable.
  }
}
ParseRule *Compiler::getRule(Token token) { return getRule(token->type); }
}  // namespace vm
}  // namespace lox
