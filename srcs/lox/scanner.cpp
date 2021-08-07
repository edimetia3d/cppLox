//
// License: MIT
//

#include "lox/scanner.h"

namespace lox {
Error Scanner::Scan() {
  while (!IsAtEnd()) {
    ScanOneToken();
  }
  tokens_.emplace_back(TokenType::EOF_TOKEN, "EOF", line_);
  return err_;
}
void Scanner::ScanOneToken() {
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
      case '!': AddToken(Match('=') ? TokenType::BANG_EQUAL : TokenType::BANG);break;
      case '=': AddToken(Match('=') ? TokenType::EQUAL_EQUAL : TokenType::EQUAL);break;
      case '<': AddToken(Match('=') ? TokenType::LESS_EQUAL : TokenType::LESS);break;
      case '>': AddToken(Match('=') ? TokenType::GREATER_EQUAL : TokenType::GREATER);break;
      case '/':
        if (Match('/')) {
          // A comment goes until the end of the line.
          while (Peek() != '\n' && !IsAtEnd()) Advance();
        } else {
          AddToken(TokenType::SLASH);
        }
      case ' ':
      case '\r':
      case '\t':
        // Ignore whitespace.
        ResetTokenBeg();
        break;
      case '\n':
        line_++;
        ResetTokenBeg();
        break;
      case '"':
        AddStringToken();
        break;
      default:
        if (IsDigit(c)) {
          AddNumToken();
        }
        else if (IsAlpha(c)) {
          AddIdentifierToken();
        }
        else{
          err_.Append(ERR_STR("Unknwon char at line "+std::to_string(line_)));break;
        }
    }
  // clang-format on
}
void Scanner::AddToken(TokenType type) {
  tokens_.emplace_back(type, std::string(srcs_->cbegin() + start_lex_pos_, srcs_->cbegin() + current_lex_pos_), line_);
  ResetTokenBeg();
}
void Scanner::ResetTokenBeg() { start_lex_pos_ = current_lex_pos_; }

char Scanner::Advance() { return srcs_->at(current_lex_pos_++); }
bool Scanner::Match(char expected) {
  if (IsAtEnd()) return false;
  if (srcs_->at(current_lex_pos_) != expected) return false;

  current_lex_pos_++;
  return true;
}
char Scanner::Peek(int offset) {
  if (IsAtEnd()) return '\0';
  return srcs_->at(current_lex_pos_ + offset);
}
void Scanner::AddStringToken() {
  while (Peek() != '"' && !IsAtEnd()) {
    if (Peek() == '\n') line_++;
    Advance();
  }

  if (IsAtEnd()) {
    err_.Append(ERR_STR("Unterminated string @line" + std::to_string(line_)));
    return;
  }

  // The closing ".
  Advance();

  AddToken(TokenType::STRING);
}
void Scanner::AddNumToken() {
  while (IsDigit(Peek())) Advance();

  // Look for a fractional part.
  if (Peek() == '.') {
    if (IsDigit(Peek(1))) {
      // Consume the "."
      Advance();

      while (IsDigit(Peek())) Advance();

      AddToken(TokenType::NUMBER);
    } else {
      err_.Append(ERR_STR("wrong number format @line" + std::to_string(line_)));
    }
  } else {
    AddToken(TokenType::NUMBER);
  }
}
void Scanner::AddIdentifierToken() {
  while (IsAlphaNumeric(Peek())) Advance();

  AddToken(Token::GetIdentifierType(std::string(
      srcs_->cbegin() + start_lex_pos_, srcs_->cbegin() + current_lex_pos_)));
}
}  // namespace lox
