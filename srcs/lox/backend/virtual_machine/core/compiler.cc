//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/core/compiler.h"

namespace lox {

namespace vm {
ErrCode Compiler::Compile(Scanner &scanner, Chunk *target) {
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
    printf("%2d '%.*s'\n", token->type, (int)token->lexeme.size(), token->lexeme.c_str());

    if (token->type == TokenType::EOF_TOKEN) break;
  }
}
}  // namespace vm
}  // namespace lox
