//
// LICENSE: MIT
//
#include "function_unit.h"

namespace lox::vm {
FunctionUnit::FunctionUnit(FunctionUnit *enclosing, FunctionType type, const std::string &name, LineInfoCB line_info)
    : type(type), line_info(line_info) {
  enclosing_ = enclosing;
  func = new ObjFunction();  // object function will get gc cleaned, so we only new , not delete
  func->name = name;
  // the function object will be pushed to stack at runtime, so locals[0] is occupied here
  locals.resize(locals.size() + 1);
  auto &function_self = locals.back();
  if (type == FunctionType::METHOD || type == FunctionType::INITIALIZER) {
    function_self.name = "this";
  } else {
    function_self.name = name;
  }
  if (enclosing_) {
    function_self.semantic_scope_depth = enclosing_->semantic_scope_depth;
  } else {
    function_self.semantic_scope_depth = 0;
  }
}
FunctionUnit::UpValue *FunctionUnit::AddUpValue(NamedValue *some_value, bool is_on_stack) {
  // check if upvalue is already added
  for (auto &uv : upvalues) {
    FunctionUnit::UpValue *upvalue = &uv;
    if (upvalue->corresponding_position == some_value->position && upvalue->is_on_stack_at_begin == is_on_stack) {
      return upvalue;
    }
  }

  if (upvalues.size() == UPVALUE_LOOKUP_MAX) {
    SPDLOG_ERROR("Upvalue limit reached.");
    return nullptr;
  }
  upvalues.resize(upvalues.size() + 1);

  upvalues.back().is_on_stack_at_begin = is_on_stack;
  upvalues.back().position = upvalues.size() - 1;
  upvalues.back().corresponding_position = some_value->position;
  return &upvalues.back();
}
FunctionUnit::UpValue *FunctionUnit::TryResolveUpValue(Token varaible_name) {
  /**
   * In enclosing, only direct enclosing is surely alive, other indirect enclosing maybe out of stack
   */
  if (enclosing_ == nullptr) return nullptr;

  if (Local *enclosing_local = enclosing_->TryResolveLocal(varaible_name)) {
    // the value is still on stack when creating closure at runtime

    enclosing_local->is_captured = true;  // mark enclosing_local as captured to emit CLOSE_UPVALUE instead of POP, when
                                          // the enclosing_local go out of scope.
    return AddUpValue(enclosing_local, true);
  } else {
    // the value is not on stack, make the enclosing_ to capture it, and we use the one captured by enclosing_.
    UpValue *enclosing_upvalue = enclosing_->TryResolveUpValue(varaible_name);
    if (enclosing_upvalue) {
      return AddUpValue(enclosing_upvalue, false);
    }
  }

  return nullptr;
}
FunctionUnit::Local *FunctionUnit::TryResolveLocal(Token varaible_name) {
  for (auto r_iter = locals.rbegin(); r_iter != locals.rend(); r_iter++) {
    FunctionUnit::Local *local = &*r_iter;
    if (varaible_name->lexeme == local->name) {
      return local;
    }
  }
  return nullptr;
}
void FunctionUnit::EmitByte(uint8_t byte) { Chunk()->WriteUInt8(byte, line_info()); }
void FunctionUnit::EmitBytes(OpCode byte1, uint8_t byte2) {
  EmitByte(byte1);
  EmitByte(byte2);
}
void FunctionUnit::EmitByte(OpCode opcode) { EmitByte(static_cast<uint8_t>(opcode)); }
void FunctionUnit::EmitBytes(OpCode opcode0, OpCode opcode1) { EmitBytes(opcode0, static_cast<uint8_t>(opcode1)); }

void FunctionUnit::EmitJumpBack(int start) {
  int ip_target = start;
  int ip_from = Chunk()->ChunkSize() + 3;  // after OP_JUMP_BACK is executed, ip will pointer to this pos

  EmitByte(OpCode::OP_JUMP_BACK);
  int offset = -1 * (ip_target - ip_from);  // always use a positive number, for we store offset into a uint16
  if (offset > UINT16_MAX) {
    SPDLOG_ERROR("Loop body too large.");
  }

  EmitByte((offset >> 8) & 0xff);
  EmitByte(offset & 0xff);
}

FunctionUnit::JumpDownHole FunctionUnit::CreateJumpDownHole(OpCode jump_cmd) {
  int initial_size = Chunk()->ChunkSize();
  EmitByte(0xff);
  EmitByte(0xff);
  EmitByte(0xff);
  int hole_size = Chunk()->ChunkSize() - initial_size;
  return JumpDownHole{jump_cmd, initial_size, hole_size};
}
void FunctionUnit::JumpHerePatch(FunctionUnit::JumpDownHole hole) {
  int beg_addr = hole.beg_offset;
  int frist_offset_after_jump = hole.beg_offset + hole.hole_size;
  // at runtime, jump will load 1 byte of OPCode and 2 byte of position, so ip will pointer to `BASE + offset + 2`
  int ip_target = Chunk()->ChunkSize();
  int jump_diff = ip_target - frist_offset_after_jump;

  if (jump_diff > UINT16_MAX) {
    SPDLOG_ERROR("Too much code to jump_diff over.");
  }

  Chunk()->code[beg_addr] = (uint8_t)hole.jump_type;
  Chunk()->code[beg_addr + 1] = (jump_diff >> 8) & 0xff;
  Chunk()->code[beg_addr + 2] = jump_diff & 0xff;
}

bool FunctionUnit::IsLocalAtOuterScope(const FunctionUnit::Local *local) const {
  return local->semantic_scope_depth < semantic_scope_depth;
}
bool FunctionUnit::IsGlobalScope() const { return semantic_scope_depth == 0; }

void FunctionUnit::CleanUpLocals(int local_var_num) {
  for (int i = 0; i < local_var_num; ++i) {
    if (locals.back().is_captured) {
      EmitByte(OpCode::OP_CLOSE_UPVALUE);
    } else {
      EmitByte(OpCode::OP_POP);
    }
    locals.pop_back();
  }
}

uint8_t FunctionUnit::StoreTokenLexmeToConstant(Token token) {
  return MakeConstant(Value(Symbol::Intern(token->lexeme)));
}

uint8_t FunctionUnit::MakeConstant(Value value) {
  int constant = Chunk()->AddConstant(value);
  if (constant > UINT8_MAX) {
    SPDLOG_ERROR("Too many constants in one chunk.");
  }

  return (uint8_t)constant;
}

void FunctionUnit::EmitDefaultReturn() {
  if (type == FunctionType::INITIALIZER) {
    EmitBytes(OpCode::OP_GET_LOCAL, 0);
  } else {
    EmitByte(OpCode::OP_NIL);
  }
  EmitByte(OpCode::OP_RETURN);
}

void FunctionUnit::EmitUnary(const TokenType &token_type) {
  switch (token_type) {
    case TokenType::MINUS:
      EmitByte(OpCode::OP_NEGATE);
      break;
    case TokenType::BANG:
      EmitByte(OpCode::OP_NOT);
      break;
    default:
      return;  // Unreachable.
  }
}
void FunctionUnit::EmitBinary(const TokenType &token_type) {
  switch (token_type) {
    case TokenType::BANG_EQUAL:
      EmitBytes(OpCode::OP_EQUAL, OpCode::OP_NOT);
      break;
    case TokenType::EQUAL_EQUAL:
      EmitByte(OpCode::OP_EQUAL);
      break;
    case TokenType::GREATER:
      EmitByte(OpCode::OP_GREATER);
      break;
    case TokenType::GREATER_EQUAL:
      EmitBytes(OpCode::OP_LESS, OpCode::OP_NOT);
      break;
    case TokenType::LESS:
      EmitByte(OpCode::OP_LESS);
      break;
    case TokenType::LESS_EQUAL:
      EmitBytes(OpCode::OP_GREATER, OpCode::OP_NOT);
      break;
    case TokenType::PLUS:
      EmitByte(OpCode::OP_ADD);
      break;
    case TokenType::MINUS:
      EmitByte(OpCode::OP_SUBTRACT);
      break;
    case TokenType::STAR:
      EmitByte(OpCode::OP_MULTIPLY);
      break;
    case TokenType::SLASH:
      EmitByte(OpCode::OP_DIVIDE);
      break;
    default:
      return;  // Unreachable.
  }
}

void FunctionUnit::EmitLiteral(TokenType token_type) {
  switch (token_type) {
    case TokenType::FALSE:
      EmitByte(OpCode::OP_FALSE);
      break;
    case TokenType::NIL:
      EmitByte(OpCode::OP_NIL);
      break;
    case TokenType::TRUE:
      EmitByte(OpCode::OP_TRUE);
      break;
    default:
      return;  // Unreachable.
  }
}
FunctionUnit::NamedValue *FunctionUnit::DeclNamedValue(Token var_name) {
  if (IsGlobalScope()) {
    // global var will be defined at runtime in global table, we only store the name to constants when decl it.
    for (auto &global : globals) {
      if (global.name == var_name->lexeme) {
        SPDLOG_ERROR("Redefine global not allowed.");
        return nullptr;
      }
    }

    if (globals.size() == GLOBAL_LOOKUP_MAX) {
      SPDLOG_ERROR("Too many global variables in script");
      return nullptr;
    }

    globals.resize(globals.size() + 1);
    Global *new_global = &globals.back();
    new_global->name = var_name->lexeme;
    new_global->position = StoreTokenLexmeToConstant(var_name);
    return new_global;
  } else {
    // Look up from the inner scope to outer scope by backward-visiting to do a redefinition check
    for (auto ref_loca = locals.rbegin(); ref_loca != locals.rend(); ++ref_loca) {
      auto Local = &*ref_loca;
      if (Local->is_inited && IsLocalAtOuterScope(Local)) {
        // If we went here, we could always create a new local variable to hide the one in outer scope
        break;
      } else {
        if (var_name->lexeme == Local->name) {
          SPDLOG_ERROR("Re-definition in same scope not allowed");
          return nullptr;
        }
      }
    }
    if (locals.size() == STACK_LOOKUP_MAX) {
      SPDLOG_ERROR("Too many local variables in function.");
      return nullptr;
    }
    locals.resize(locals.size() + 1);
    Local *new_local = &locals.back();
    new_local->name = var_name->lexeme;
    new_local->semantic_scope_depth = semantic_scope_depth;
    new_local->position = locals.size() - 1;
    return new_local;
  }
}

void FunctionUnit::DefineNamedValue(NamedValue *handle) {
  if (IsGlobalScope()) {
    assert(handle->position < GLOBAL_LOOKUP_MAX);
    // A global var will be at the stack top value, we just move it to global table.
    EmitBytes(OpCode::OP_DEFINE_GLOBAL, handle->position);

    // global var def and decl must be paired.
    assert(handle->position == (globals.size() - 1));

  } else {
    // Stack value do not need to move at runtime.

    // local var def and decl must be paired.
    assert(handle->position == (locals.size() - 1));
  }
  handle->is_inited = true;
}
FunctionUnit::Global *FunctionUnit::TryResolveGlobal(Token varaible_name) {
  for (auto &global : globals) {
    if (global.name == varaible_name->lexeme) {
      return &global;
    }
  }
  return nullptr;
}
}  // namespace lox::vm