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
      case '!': AddToken(match('=') ? TokenType::BANG_EQUAL : TokenType::BANG);break;
      case '=': AddToken(match('=') ? TokenType::EQUAL_EQUAL : TokenType::EQUAL);break;
      case '<': AddToken(match('=') ? TokenType::LESS_EQUAL : TokenType::LESS);break;
      case '>': AddToken(match('=') ? TokenType::GREATER_EQUAL : TokenType::GREATER);break;
      case '/':
        if (match('/')) {
          // A comment goes until the end of the line.
          while (peek() != '\n' && !isAtEnd()) Advance();
        } else {
          AddToken(TokenType::SLASH);
        }
      case ' ':
      case '\r':
      case '\t':
        // Ignore whitespace.
        break;
      case '\n':
        line_++;
        break;
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
bool Scanner::match(char expected) {
  if (isAtEnd()) return false;
  if (srcs_->at(current_lex_pos_) != expected) return false;

  current_lex_pos_++;
  return true;
}
char Scanner::peek() {
  if (isAtEnd()) return '\0';
  return srcs_->at(current_lex_pos_);
}
}  // namespace lox