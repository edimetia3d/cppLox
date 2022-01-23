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
      return ret;
    }
  }
  return nullptr;
}
void FunctionUnit::EmitConstant(Value value) { EmitBytes(OpCode::OP_CONSTANT, AddValueConstant(value)); }
void FunctionUnit::EmitOpClosure(ObjFunction *func, const std::vector<UpValue> &upvalues_of_func,
                                 std::map<std::string, UpValue *> value_need_to_force_closed) {
  int extra_count = 0;
  for (auto &pair : value_need_to_force_closed) {
    auto handle = ResolveNamedValue(MakeToken(TokenType::IDENTIFIER, pair.first, -1));
    EmitBytes(handle.get_op, handle.reslove.position);
    pair.second->position_at_begin = value_need_to_force_closed.size() - extra_count - 1;
    ++extra_count;
  }
  EmitBytes(OpCode::OP_CLOSURE, AddValueConstant(Value(func)));
  EmitByte((uint8_t)upvalues_of_func.size());
  for (auto &upvalue : upvalues_of_func) {
    EmitByte(static_cast<uint8_t>(upvalue.src_at_begin));
    EmitByte(upvalue.position_at_begin);
  }
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

void FunctionUnit::ForceCloseValue(Token name_in_outer_scope) {
  if (!force_closed_values.contains(name_in_outer_scope->lexeme)) {
    Token new_name = MakeToken(TokenType::IDENTIFIER, "__closed_" + name_in_outer_scope->lexeme, line_info_callback());
    if (!enclosing) {
      Error("Cannot force close in global scope");
    }

    if (upvalues.size() == UPVALUE_COUNT_LIMIT) {
      Error("Too many closure variables in function.");
    }
    upvalues.resize(upvalues.size() + 1);
    upvalues.back().name = new_name->lexeme;
    upvalues.back().src_at_begin = UpValueSrc::ON_STACK_TOP;
    upvalues.back().position = upvalues.size() - 1;
    upvalues.back().position_at_begin = -1;
    upvalues.back().is_inited = true;  // all upvalues are already inited

    force_closed_values[name_in_outer_scope->lexeme] = &upvalues.back();
  }
  auto up_value = force_closed_values[name_in_outer_scope->lexeme];
  EmitBytes(OpCode::OP_GET_UPVALUE, up_value->position);
}

FunctionUnit::NamedValueOperator FunctionUnit::ResolveNamedValue(Token varaible_name) {
  /**
   * About UpValue:
   *
   * At runtime, If the closed value we want to access is still on the stack, we should access the value on stack. If
   * the closed value is not on stack, we should access the copied one. e.g. we define a closure, and then call it
   * immediately, we should access the raw value that is still on stack.
   *
   * To support this feature:
   *    1. compiler and virtual machine will have to work together.
   *    2. The closed value at runtime will be a pointer point to stack initially, and it will point to a new location
   *       when the stack position it points to is popped.
   *
   * Note that the global level function will not have any upvalue, all unresolved symbol in global level will go to
   * global directly. That is , the global level function is a closure that closes nothing.
   *
   * To a normal closure:
   *    1. The ObjFunction of closure is already created by compiler at compilation time, and we can do nothing to it.
   *       (It may contains some GET_UPVALUE, SET_UPVALUE op, but not very important.)
   *    2. There will always be a "outer function", act as the runtime creator of the closure.
   *
   * The ClosureCreator will do all the magic things at runtime to make the inner defined ObjFunction a real closure.
   *    1. At compile time, just after the inner function is defined (a `PopCU()` is executed), a `OP_CLOUSRE` will be
   * emitted. At runtime, `OP_CLOUSRE` will create a ObjClosure, and update the upvalues of it to make the upvalues
   * point to correct position, and for the upvalues that point to stack, we will track them as `opened-upvalues`, they
   * will get updated later.
   *    2. At compile time, normally, when a var out of scope, a OP_POP will be emit, but if it is needed by some inner
   * function, a `OP_CLOSE_UPVALUE` will be emitted, this OP will not only do the pop, but also do some upvalue updating
   *       job at runtime. At runtime, `OP_CLOSE_UPVALUE` will try to close all opened-upvalues that point to a invalid
   *       stack position. Normally, a `out of scope` will move the sp to correct place, and all value on stack after
   *       the correct place will be treated as `poped`, thus, it is very easy to determine whether a opened-upvalue
   * point to a invalid place. And also, OP_CLOSE_UPVALUE just do the `close`, it doesnt care which function the upvalue
   *       belongs to.
   *
   * Note that:
   *    1. the truth that closed values are get "copied" (or captured, closed) when they went out of scope, it may cause
   * some logical misleading, e.g. creating closure in loops will make every closure closed to the same value.
   *    2. we will also support the chain style upvalue, if a nested closure is created, and they need to access same
   *    variable, then same value will always be used at runtime.
   *
   *
   * To support the code gen of `OP_CLOUSRE` and `OP_CLOSE_UPVALUE`, we need to track the upvalue at compile time. And
   * all these tracking will be done by the TryResolveUpValue, obviously, upvalue is only known by compiler, user code
   * will not decl/define upvalue.
   */

  OpCode get_op, set_op;
  FunctionUnit::NamedValue *p_resolve = nullptr;
  std::unique_ptr<FunctionUnit::Global> global_resolve;
  if ((p_resolve = TryResolveLocal(varaible_name))) {
    get_op = OpCode::OP_GET_LOCAL;
    set_op = OpCode::OP_SET_LOCAL;
    assert(p_resolve->position < STACK_COUNT_LIMIT);
  } else if ((p_resolve = TryResolveUpValue(varaible_name))) {
    get_op = OpCode::OP_GET_UPVALUE;
    set_op = OpCode::OP_SET_UPVALUE;
    assert(p_resolve->position < UPVALUE_COUNT_LIMIT);
  } else if ((global_resolve = TryResolveGlobal(varaible_name))) {
    get_op = OpCode::OP_GET_GLOBAL;
    set_op = OpCode::OP_SET_GLOBAL;
    p_resolve = global_resolve.get();
    assert(p_resolve->position < CONSTANT_COUNT_LIMIT);
  } else {
    // We will treat all unknown variable as global variable, and delay the error to runtime.
    // because new global variable might be created at runtime before we actually access it.

    get_op = OpCode::OP_GET_GLOBAL;
    set_op = OpCode::OP_SET_GLOBAL;
    global_resolve = std::make_unique<FunctionUnit::Global>();
    global_resolve->name = varaible_name->lexeme;
    global_resolve->is_inited = true;
    global_resolve->position = GetSymbolConstant(varaible_name->lexeme);
    assert(global_resolve->position < CONSTANT_COUNT_LIMIT);
    p_resolve = global_resolve.get();
  }
  if (!p_resolve->is_inited) {
    Error("Can't read local variable in its own initializer.");
  }
  return NamedValueOperator{
      .get_op = get_op,
      .set_op = set_op,
      .reslove = *p_resolve,
  };
}

}  // namespace lox::vm