//
// License: MIT
//

#include "lox/frontend/scanner.h"

#include "lox/common/lox_error.h"

namespace lox {

Token Scanner::ScanOne() {
  if (input_->IsAtEnd()) {
    auto beg = "EOF";
    auto end = beg + 3;
    return {TokenType::EOF_TOKEN, RefString(beg, end), Location(input_, line_, input_->Pos() - 1)};
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
          while (Peek() != '\n' && !input_->IsAtEnd()) Advance();
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
  auto str_beg = input_->Data() + start_lex_pos_;
  auto str_end = input_->Data() + input_->Pos();
  auto ret = Token(type, RefString(str_beg, str_end), Location(input_, line_, start_lex_pos_));
  ResetTokenBeg();
  return ret;
}
void Scanner::ResetTokenBeg() { start_lex_pos_ = input_->Pos(); }

char Scanner::Advance() { return input_->Read(); }

bool Scanner::MatchAndAdvance(char expected) {
  if (input_->IsAtEnd())
    return false;
  if (input_->Peek() != expected)
    return false;
  Advance();
  return true;
}

char Scanner::Peek(int offset) {
  auto read_pos = input_->Pos() + offset;
  if (read_pos >= input_->Size())
    return '\0';
  return input_->At(read_pos);
}

Token Scanner::AddStringToken() {
  while (Peek() != '"' && !input_->IsAtEnd()) {
    if (Peek() == '\n')
      StartNewLine();
    Advance();
  }

  if (input_->IsAtEnd()) {
    Error("Unterminated string.");
  }

  // The closing ".
  Advance();

  return AddToken(TokenType::STRING);
}
Token Scanner::AddNumToken() {
  while (IsDigit(Peek()))
    Advance();

  // Look for a fractional part.
  if ((Peek() == '.') && IsDigit(Peek(1))) {
    // Consume the "."
    Advance();

    while (IsDigit(Peek()))
      Advance();

    return AddToken(TokenType::NUMBER);
  } else {
    return AddToken(TokenType::NUMBER);
  }
}

Token Scanner::AddIdentifierToken() {
  while (IsAlphaNumeric(Peek()))
    Advance();
  auto beg = input_->Data() + start_lex_pos_;
  auto end = input_->Data() + input_->Pos();
  return AddToken(Token::GetIdentifierType(std::string(beg, end)));
}

void Scanner::Error(const std::string &msg) {
  std::vector<char> buf(100);
  snprintf(buf.data(), buf.size(), "[line %d] Error: %s", line_ + 1, msg.c_str());
  throw ScannerError(buf.data());
}

} // namespace lox
