//
// LICENSE: MIT
//
#include "lox/backend/virtual_machine/core/function_unit.h"

#include "lox/backend/virtual_machine/errors.h"
#include <spdlog/spdlog.h>

namespace lox::vm {
std::vector<FunctionUnit::Global> FunctionUnit::globals;

FunctionUnit::FunctionUnit(FunctionUnit *enclosing, FunctionType type, const std::string &name, LineInfoCB line_info,
                           ErrorCB error_cb)
    : enclosing(enclosing), type(type), line_info_callback(line_info), error_callback(error_cb) {
  // these three reserve are very important, because se will use pointer to point to the elements
  // we must use reserve to avoid dangled pointer caused by reallocation.
  locals.reserve(STACK_COUNT_LIMIT);
  upvalues.reserve(UPVALUE_COUNT_LIMIT);
  globals.reserve(CONSTANT_COUNT_LIMIT);
  func = new ObjFunction(name);  // object function will get gc cleaned, so we only new , not delete
  if (enclosing) {
    current_semantic_scope_level = enclosing->current_semantic_scope_level;
  } else {
    current_semantic_scope_level = 0;
  }

  // slots[0] is reserved for support "this" argument, and for other function, it will be filled as the closure itself
  locals.resize(locals.size() + 1);
  auto &function_self = locals.back();

  if (type == FunctionType::METHOD || type == FunctionType::INITIALIZER) {
    function_self.name = "this";
  } else {
    function_self.name = name;
  }
  function_self.is_inited = true;  // a hack that treat `this` as always inited, so we can reference self recursively.
  function_self.position = locals.size() - 1;
  function_self.semantic_scope_depth = current_semantic_scope_level;
}

FunctionUnit::UpValue *FunctionUnit::DoAddUpValue(NamedValue *some_value, UpValueSrc beg_src) {
  for (auto &uv : upvalues) {
    FunctionUnit::UpValue *upvalue = &uv;
    if (upvalue->name == some_value->name && upvalue->src_at_begin == UpValueSrc::ON_STACK_TAIL) {
      return upvalue;
    }
  }

  for (auto &uv : upvalues) {
    FunctionUnit::UpValue *upvalue = &uv;
    if (upvalue->position_at_begin == some_value->position && upvalue->src_at_begin == beg_src) {
      return upvalue;
    }
  }

  if (upvalues.size() == UPVALUE_COUNT_LIMIT) {
    Error("Too many closure variables in function.");
  }
  upvalues.resize(upvalues.size() + 1);
  upvalues.back().name = some_value->name;
  upvalues.back().src_at_begin = beg_src;
  upvalues.back().position = upvalues.size() - 1;
  upvalues.back().position_at_begin = some_value->position;
  upvalues.back().is_inited = true;  // all upvalues are already inited
  return &upvalues.back();
}
FunctionUnit::UpValue *FunctionUnit::AddUpValueFromEnclosingStack(Local *some_value) {
  return DoAddUpValue(some_value, UpValueSrc::ON_SLOT_BEGIN);
}

FunctionUnit::UpValue *FunctionUnit::AddUpValueFromEnclosingUpValue(FunctionUnit::UpValue *some_value) {
  return DoAddUpValue(some_value, UpValueSrc::ON_ENCLOSING_UPVALUE);
}

FunctionUnit::UpValue *FunctionUnit::TryResolveUpValue(Token varaible_name) {
  /**
   * In enclosing, only direct enclosing is surely alive, other indirect enclosing maybe out of stack
   */
  if (enclosing == nullptr) return nullptr;

  if (Local *enclosing_local = enclosing->TryResolveLocal(varaible_name)) {
    // the value is still on stack when creating closure at runtime

    enclosing_local->is_captured = true;  // mark enclosing_local as captured to emit CLOSE_UPVALUE instead of POP, when
                                          // the enclosing_local go out of scope.
    return AddUpValueFromEnclosingStack(enclosing_local);
  } else {
    // the value is not on stack, make the enclosing_ to capture it, and we use the one captured by enclosing_.
    UpValue *enclosing_upvalue = enclosing->TryResolveUpValue(varaible_name);
    if (enclosing_upvalue) {
      return AddUpValueFromEnclosingUpValue(enclosing_upvalue);
    }
  }

  return nullptr;
}

FunctionUnit::Local *FunctionUnit::TryResolveLocal(Token varaible_name) {
  for (auto r_iter = locals.rbegin(); r_iter != locals.rend(); r_iter++) {
    if (varaible_name->lexeme == r_iter->name) {
      return &*r_iter;
    }
  }
  return nullptr;
}

void FunctionUnit::EmitByte(uint8_t byte) { Chunk()->WriteUInt8(byte, line_info_callback()); }

void FunctionUnit::EmitBytes(OpCode byte1, uint8_t byte2) {
  EmitByte(byte1);
  EmitByte(byte2);
}
void FunctionUnit::EmitByte(OpCode opcode) { EmitByte(static_cast<uint8_t>(opcode)); }
void FunctionUnit::EmitBytes(OpCode opcode0, OpCode opcode1) { EmitBytes(opcode0, static_cast<uint8_t>(opcode1)); }

void FunctionUnit::EmitJumpBack(int start) {
  int ip_target = start;
  int ip_from = Chunk()->ChunkSize() + 3;  // after OP_JUMP_BACK is consumed by VM, ip will pointer to this pos

  EmitByte(OpCode::OP_JUMP_BACK);
  int offset = -1 * (ip_target - ip_from);  // always use a positive number to get longer jump range, that's why we
                                            // create a new OP_JUMP_BACK instruction
  if (offset > UINT16_MAX) {
    Error("Loop body too large.");
  }

  EmitByte((offset >> 8) & 0xff);
  EmitByte(offset & 0xff);
}

FunctionUnit::JumpDownHole FunctionUnit::CreateJumpDownHole(OpCode jump_cmd) {
  int instruction_beg_offset = Chunk()->ChunkSize();
  EmitByte(jump_cmd);
  EmitByte(0xff);
  EmitByte(0xff);
  int hole_size = Chunk()->ChunkSize() - instruction_beg_offset;
  return JumpDownHole{jump_cmd, instruction_beg_offset, hole_size};
}
void FunctionUnit::JumpHerePatch(FunctionUnit::JumpDownHole hole) {
  int beg_addr = hole.beg_offset;
  int ip_from = hole.beg_offset + hole.hole_size;
  int ip_target = Chunk()->ChunkSize();
  int jump_diff = ip_target - ip_from;

  if (jump_diff > UINT16_MAX) {
    Error("Jump too far.");
  }

  Chunk()->code[beg_addr] = (uint8_t)hole.jump_type;
  Chunk()->code[beg_addr + 1] = (jump_diff >> 8) & 0xff;
  Chunk()->code[beg_addr + 2] = jump_diff & 0xff;
}

bool FunctionUnit::IsGlobalScope() const { return current_semantic_scope_level == 0; }

void FunctionUnit::CleanUpNLocalFromTail(int local_var_num) {
  for (int i = 0; i < local_var_num; ++i) {
    if (locals.back().is_captured) {
      EmitByte(OpCode::OP_CLOSE_UPVALUE);
    } else {
      EmitByte(OpCode::OP_POP);
    }
    locals.pop_back();
  }
}

uint8_t FunctionUnit::AddValueConstant(Value value) {
  if ((value.IsObject() && value.AsObject()->DynAs<Symbol>())) {
    return GetSymbolConstant(value.AsObject()->DynAs<Symbol>()->c_str());
  }
  if (Chunk()->constants.size() == CONSTANT_COUNT_LIMIT) {
    Error("Too many constants in one chunk.");
  }
  return Chunk()->AddConstant(value);
}

void FunctionUnit::EmitDefaultReturn() {
  if (type == FunctionType::INITIALIZER) {
    EmitBytes(OpCode::OP_GET_LOCAL, 0);  // instance `this` will be at this slot
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
      Error("Unknown unary operator.");
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
      Error("Unknown binary operator.");
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
      Error("Unknown literal.");
      return;  // Unreachable.
  }
}
FunctionUnit::NamedValue *FunctionUnit::DeclNamedValue(Token var_name) {
  if (IsGlobalScope()) {
    // global var need to be moved to vm's global list at runtime by a instruction

    // check redifinition
    for (auto &global : globals) {
      if (global.name == var_name->lexeme) {
        SPDLOG_DEBUG("Redefine global of {} detected.", var_name->lexeme);
      }
    }

    if (globals.size() == CONSTANT_COUNT_LIMIT) {
      Error("Too many global variables in script");
    }

    globals.resize(globals.size() + 1);
    Global *new_global = &globals.back();
    new_global->name = var_name->lexeme;
    return new_global;
  } else {
    // check redifinition
    for (auto r_iter = locals.rbegin(); r_iter != locals.rend(); ++r_iter) {
      if (r_iter->is_inited && r_iter->semantic_scope_depth < current_semantic_scope_level) {
        // If we went here, the local variable is defined in a parent scope, we are fine to create a new one
        // to hide the outer one.
        break;
      } else {
        if (var_name->lexeme == r_iter->name) {
          Error("Already a variable with this name in this scope.");
        }
      }
    }
    if (locals.size() == STACK_COUNT_LIMIT) {
      Error("Too many local variables in function.");
    }
    locals.resize(locals.size() + 1);
    Local *new_local = &locals.back();
    new_local->name = var_name->lexeme;
    new_local->semantic_scope_depth = current_semantic_scope_level;
    new_local->position = locals.size() - 1;
    return new_local;
  }
}

void FunctionUnit::DefineNamedValue(NamedValue *value) {
  if (IsGlobalScope()) {
    // move the stack top to vm's global by it's name
    EmitBytes(OpCode::OP_DEFINE_GLOBAL, GetSymbolConstant(value->name));

    // global var def and decl must be paired.
    assert(value == &globals.back());

  } else {
    // stack var will just leave on stack

    // local var def and decl must be paired.
    assert(value->position == (locals.size() - 1));
  }
  value->is_inited = true;
}
std::unique_ptr<FunctionUnit::Global> FunctionUnit::TryResolveGlobal(Token varaible_name) {
  for (auto &global : globals) {
    if (global.name == varaible_name->lexeme) {
      if (global.position != -1) {
        Error("Unknown global access error");
      }
      auto ret = std::make_unique<FunctionUnit::Global>();
      *ret = global;
      ret->position = GetSymbolConstant(varaible_name->lexeme);
    }
  }
  return nullptr;
}
void FunctionUnit::EmitConstant(Value value) { EmitBytes(OpCode::OP_CONSTANT, AddValueConstant(value)); }
void FunctionUnit::EmitOpClosure(ObjFunction *func, std::vector<UpValue> upvalues_of_func, int extra_closed_value) {
  EmitBytes(OpCode::OP_CLOSURE, AddValueConstant(Value(func)));
  EmitByte((uint8_t)upvalues_of_func.size());
  EmitByte(extra_closed_value);
  for (auto &upvalue : upvalues_of_func) {
    EmitByte(static_cast<uint8_t>(upvalue.src_at_begin));
    EmitByte(upvalue.position_at_begin);
  }
  locals.resize(locals.size() - extra_closed_value);
}
uint8_t FunctionUnit::GetSymbolConstant(const std::string &str) {
  if (!used_symbol_constants.contains(str)) {
    if (Chunk()->constants.size() == CONSTANT_COUNT_LIMIT) {
      Error("Too many constants in one chunk.");
    }
    auto constant = Chunk()->AddConstant(Value(Symbol::Intern(str)));
    used_symbol_constants[str] = constant;
    return constant;
  }
  return used_symbol_constants[str];
}

void FunctionUnit::Error(const std::string &msg) {
  SPDLOG_DEBUG(msg);
  error_callback(msg.c_str());
}

}  // namespace lox::vm