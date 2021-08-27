//
// License: MIT
//

#include "lox/frontend/scanner.h"

namespace lox {
LoxError Scanner::ScanAll(std::vector<Token>* output) {
  ResetErr();
  while (!IsAtEnd()) {
    ScanOne();
    output->push_back(last_scan_);
  }
  output->push_back(MakeToken(TokenType::EOF_TOKEN, "EOF", line_));
  return err_;
}

LoxError Scanner::ScanOne(Token* output) {
  ResetErr();
  if (!IsAtEnd()) {
    ScanOne();
    *output = last_scan_;
  } else {
    *output = MakeToken(TokenType::EOF_TOKEN, "EOF", line_);
  }

  return err_;
}

void Scanner::ScanOne() {
  new_token_scaned_ = false;
TRY_SCAN:
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
        err_.Merge(LoxError("Unknwon char at line "+std::to_string(line_)));return;
      }
  }
  // clang-format on
  if (!new_token_scaned_) {
    goto TRY_SCAN;
  }
}
void Scanner::AddToken(TokenType type) {
  last_scan_ =
      MakeToken(type, std::string(srcs_->cbegin() + start_lex_pos_, srcs_->cbegin() + current_lex_pos_), line_);
  new_token_scaned_ = true;
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
    err_.Merge(LoxError("Unterminated string @line" + std::to_string(line_)));
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
      err_.Merge(LoxError("wrong number format @line" + std::to_string(line_)));
    }
  } else {
    AddToken(TokenType::NUMBER);
  }
}
void Scanner::AddIdentifierToken() {
  while (IsAlphaNumeric(Peek())) Advance();

  AddToken(
      TokenBase::GetIdentifierType(std::string(srcs_->cbegin() + start_lex_pos_, srcs_->cbegin() + current_lex_pos_)));
}
void Scanner::ResetErr() { err_ = LoxError(); }

}  // namespace lox
