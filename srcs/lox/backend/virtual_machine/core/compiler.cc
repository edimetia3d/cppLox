//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/compiler.h"

namespace lox {

namespace vm {
void DumpToken(Scanner &scanner) {
  int line = -1;
  Token token;
  for (;;) {
    scanner.ScanOne(&token);
    if (token->line != line) {
      printf("%4d ", token->line);
      line = token->line;
    } else {
      printf("   | ");
    }
    printf("%2d '%.*s'\n", (int)token->type, (int)token->lexeme.size(), token->lexeme.c_str());

    if (token->type == TokenType::EOF_TOKEN) break;
  }
}
ErrCode Compiler::Compile(Chunk *target) {
  current_trunk_ = target;
  Advance();
  // Expression();
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
    auto err = scanner_.ScanOne(&parser_.current);
    if (err.NoError() && parser_.current->type != TokenType::EOF_TOKEN) break;

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
}  // namespace vm
}  // namespace lox
