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
  Expression();
  Consume(TokenType::EOF_TOKEN, "Expect end of expression.");
  endCompiler();
  if (parser_.hadError) {
    return ErrCode::PARSE_FAIL;
  }
  return ErrCode::NO_ERROR;
}
void Compiler::Advance() {
  parser_.previous = parser_.current;

  for (;;) {
    auto err = scanner_->ScanOne(&parser_.current);
    if (err.NoError()) break;

    errorAtCurrent(parser_.current->lexeme.c_str());
  }
}
void Compiler::errorAtCurrent(const char *message) {
  if (parser_.panicMode) return;
  errorAt(parser_.current, message);
}
void Compiler::errorAt(Token token, const char *message) {
  parser_.panicMode = true;
  fprintf(stderr, "[line %d] Error", token->line);

  if (token->type == TokenType::EOF_TOKEN) {
    fprintf(stderr, " at end");
  } else {
    fprintf(stderr, " at '%.*s'", (int)token->lexeme.size(), token->lexeme.c_str());
  }

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
Chunk *Compiler::CurrentChunk() { return current_trunk_; }
void Compiler::endCompiler() { emitReturn(); }
void Compiler::emitReturn() { emitOpCode(OpCode::OP_RETURN); }
void Compiler::error(const char *message) { errorAt(parser_.previous, message); }
void Compiler::Expression() { parsePrecedence(Precedence::PREC_ASSIGNMENT); }
void Compiler::parsePrecedence(Precedence precedence) {
  Advance();
  auto prefixRule = getRule(parser_.previous)->prefix;
  if (prefixRule == nullptr) {
    error("Expect expression.");
    return;
  }

  prefixRule(this);

  while (precedence <= getRule(parser_.current)->precedence) {
    Advance();
    auto infixRule = getRule(parser_.previous)->infix;
    infixRule(this);
  }
}
ParseRule *Compiler::getRule(TokenType type) {
#define RULE_ITEM(TOKEN_T, MEMBER_FN0, MEMBER_FN1, PRECEDENCE_V)                       \
  {                                                                                    \
    TokenType::TOKEN_T, ParseRule { MEMBER_FN0, MEMBER_FN1, Precedence::PRECEDENCE_V } \
  }
#define M(MEMBER_FN) &Compiler::MEMBER_FN
  // clang-format off
  static std::map<TokenType,ParseRule> rules    = {
      RULE_ITEM(LEFT_PAREN   , M(grouping), nullptr  , PREC_NONE),
      RULE_ITEM(RIGHT_PAREN  , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(LEFT_BRACE   , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(RIGHT_BRACE  , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(COMMA        , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(DOT          , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(MINUS        , M(unary)   , M(binary), PREC_TERM),
      RULE_ITEM(PLUS         , nullptr    , M(binary), PREC_TERM),
      RULE_ITEM(SEMICOLON    , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(SLASH        , nullptr    , M(binary), PREC_FACTOR),
      RULE_ITEM(STAR         , nullptr    , M(binary), PREC_FACTOR),
      RULE_ITEM(BANG         , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(BANG_EQUAL   , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(EQUAL        , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(EQUAL_EQUAL  , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(GREATER      , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(GREATER_EQUAL, nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(LESS         , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(LESS_EQUAL   , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(IDENTIFIER   , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(STRING       , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(NUMBER       , M(number)  , nullptr  , PREC_NONE),
      RULE_ITEM(AND          , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(CLASS        , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(ELSE         , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(FALSE        , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(FOR          , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(FUN          , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(IF           , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(NIL          , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(OR           , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(PRINT        , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(RETURN       , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(THIS         , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(TRUE         , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(VAR          , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(WHILE        , nullptr    , nullptr  , PREC_NONE),
      RULE_ITEM(EOF_TOKEN    , nullptr    , nullptr  , PREC_NONE),
  };
  // clang-format on
#undef RULE_ITEM
#undef M
  return &rules[type];
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
  TokenType operatorType = parser_.previous->type;

  // Compile the operand.
  parsePrecedence(Precedence::PREC_UNARY);

  // Emit the operator instruction.
  switch (operatorType) {
    case TokenType::MINUS:
      emitOpCode(OpCode::OP_NEGATE);
      break;
    default:
      return;  // Unreachable.
  }
}
void Compiler::binary() {
  TokenType operatorType = parser_.previous->type;
  ParseRule *rule = getRule(operatorType);
  parsePrecedence((Precedence)((int)(rule->precedence) + 1));

  switch (operatorType) {
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
