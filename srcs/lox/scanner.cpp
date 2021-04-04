//
// License: MIT
//

#include "scanner.h"
namespace lox {
Error Scanner::Scan() {
  tokens_.emplace_back(TokenType::EOF_TOKEN, *srcs_, 0);
  return err_;
}
void Scanner::scanSinlge() {
  char c = Advance();
  // clang-format off
    switch (c) {
      case '(': AddToken(TokenType::LEFT_PAREN); break;
      case ')': AddToken(TokenType::RIGHT_PAREN); break;
      case '{': AddToken(TokenType::LEFT_BRACE); break;
      case '}': AddToken(TokenType::RIGHT_BRACE); break;
      case ',': AddToken(TokenType::COMMA); break;
      case '.': AddToken(TokenType::DOT); break;
      case '-': AddToken(TokenType::MINUS); break;
      case '+': AddToken(TokenType::PLUS); break;
      case ';': AddToken(TokenType::SEMICOLON); break;
      case '*': AddToken(TokenType::STAR); break;
default: err_.Append(ERR_STR("Unknwon char at line "+std::to_string(line_)));break;
    }
  // clang-format on
}
void Scanner::AddToken(TokenType type) {
  tokens_.emplace_back(
      type,
      std::string(&srcs_->at(start_lex_pos_), &srcs_->at(current_lex_pos_)),
      line_);
  start_lex_pos_ = current_lex_pos_;
}

char Scanner::Advance() { return srcs_->at(current_lex_pos_++); }
}  // namespace lox