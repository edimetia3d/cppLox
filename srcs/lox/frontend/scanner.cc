//
// License: MIT
//

#include "lox/frontend/scanner.h"

#include "lox/common/lox_error.h"

namespace lox {

std::vector<Token> Scanner::ScanAll() {
  std::vector<Token> output;
  while (output.empty() || output.back()->type != TokenType::EOF_TOKEN) {
    output.push_back(ScanOne());
  }
  return output;
}

Token Scanner::ScanOne() {
  if (IsAtEnd()) {
    return Token(TokenType::EOF_TOKEN, "EOF", line_, col_, src_file_name);
  }
  ResetTokenBeg();
  char c = Advance();
  // clang-format off
  switch (c) {
    case '(': return AddToken(TokenType::LEFT_PAREN);
    case ')': return AddToken(TokenType::RIGHT_PAREN);
    case '[': return AddToken(TokenType::LEFT_SQUARE);
    case ']': return AddToken(TokenType::RIGHT_SQUARE);
    case '{': return AddToken(TokenType::LEFT_BRACE);
    case '}': return AddToken(TokenType::RIGHT_BRACE);
    case ',': return AddToken(TokenType::COMMA);
    case '.': return AddToken(TokenType::DOT);
    case '-': return AddToken(TokenType::MINUS);
    case '+': return AddToken(TokenType::PLUS);
    case ':': return AddToken(TokenType::COLON);
    case ';': return AddToken(TokenType::SEMICOLON);
    case '*': return AddToken(TokenType::STAR);
    case '!': return AddToken(MatchAndAdvance('=') ? TokenType::BANG_EQUAL : TokenType::BANG);
    case '=': return AddToken(MatchAndAdvance('=') ? TokenType::EQUAL_EQUAL : TokenType::EQUAL);
    case '<': return AddToken(MatchAndAdvance('=') ? TokenType::LESS_EQUAL : TokenType::LESS);
    case '>': return AddToken(MatchAndAdvance('=') ? TokenType::GREATER_EQUAL : TokenType::GREATER);
    case '/':{
        if (MatchAndAdvance('/')) {
          // A comment goes until the end of the line.
          while (Peek() != '\n' && !IsAtEnd()) Advance();
          return ScanOne();
        } else {
          return AddToken(TokenType::SLASH);
        }
      }
    case ' ':
    case '\r':
    case '\t':
      return ScanOne();
    case '\n':
      StartNewLine();
      return ScanOne();
    case '"':
      return AddStringToken();
    default:
      if (IsDigit(c)) {
        return AddNumToken();
      }
      else if (IsAlpha(c)) {
        return AddIdentifierToken();
      }
      else{
         Error("Unexpected character.");
      }
  }
  // clang-format on
  Error("Unknown error");
  return Token();
}
Token Scanner::AddToken(TokenType type) {
  auto ret = Token(type, std::string(srcs_->cbegin() + start_lex_pos_, srcs_->cbegin() + current_lex_pos_), line_, col_,
                   src_file_name);
  ResetTokenBeg();
  return ret;
}
void Scanner::ResetTokenBeg() { start_lex_pos_ = current_lex_pos_; }

char Scanner::Advance() {
  auto ret = LastChar();
  ++current_lex_pos_;
  ++col_;
  return ret;
}

bool Scanner::MatchAndAdvance(char expected) {
  if (IsAtEnd()) return false;
  if (LastChar() != expected) return false;
  Advance();
  return true;
}

char Scanner::Peek(int offset) {
  if ((current_lex_pos_ + offset) >= srcs_->size()) return '\0';
  return srcs_->at(current_lex_pos_ + offset);
}

Token Scanner::AddStringToken() {
  while (Peek() != '"' && !IsAtEnd()) {
    if (Peek() == '\n') StartNewLine();
    Advance();
  }

  if (IsAtEnd()) {
    Error("Unterminated string.");
  }

  // The closing ".
  Advance();

  return AddToken(TokenType::STRING);
}
Token Scanner::AddNumToken() {
  while (IsDigit(Peek())) Advance();

  // Look for a fractional part.
  if ((Peek() == '.') && IsDigit(Peek(1))) {
    // Consume the "."
    Advance();

    while (IsDigit(Peek())) Advance();

    return AddToken(TokenType::NUMBER);
  } else {
    return AddToken(TokenType::NUMBER);
  }
}

Token Scanner::AddIdentifierToken() {
  while (IsAlphaNumeric(Peek())) Advance();

  return AddToken(
      Token::GetIdentifierType(std::string(srcs_->cbegin() + start_lex_pos_, srcs_->cbegin() + current_lex_pos_)));
}

char Scanner::LastChar() { return srcs_->at(current_lex_pos_); }

void Scanner::Error(const std::string &msg) {
  std::vector<char> buf(100);
  snprintf(buf.data(), buf.size(), "[line %d] Error: %s", line_ + 1, msg.c_str());
  throw ScannerError(buf.data());
}

}  // namespace lox
