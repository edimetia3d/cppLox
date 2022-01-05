//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_FUNCTION_UNIT_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_FUNCTION_UNIT_H

#include <string>

#include "lox/backend/virtual_machine/core/chunk.h"
#include "lox/backend/virtual_machine/object/object.h"
#include "lox/token/token.h"

namespace lox::vm {

using LineInfoCB = std::function<int()>;
enum class FunctionType { UNKNOWN, FUNCTION, METHOD, INITIALIZER, SCRIPT };

/**
 * Function compilation unit is the core of the compiling process. For we are always creating nested FunctionUnit.
 *
 * For ObjFunction only contains things needed at runtime, we need a wrapper to help ObjFunction creation at compile
 * time.
 */
struct FunctionUnit {
  FunctionUnit(FunctionUnit* enclosing, FunctionType type, const std::string& name, LineInfoCB line_info);

  ///////////////////////////////////////////// NAME RESOLVE SUPPORT BEG////////////////////////////////////////////////

  /**
   * Name declaration/definition/resolve and closure support are highly related.
   *
   * Every obj that has a `name` is a named value, eg. a variable, a function, a class, etc.
   *
   * In function unit, we will:
   *    1. track active stack values of this function along the compilation process, stack values will going in and out.
   *    2. track all upvalues needed by this function and this function's nested closure, only new items will be added.
   *    3. track all globals needed by this function, only new items will be added.
   *
   * DeclNamedValue will make a named value entry with un-inited state.
   * DefineNamedValue will convert the named value entry to inited state.
   * TryResolveXXX will try to resolve a named value entry, if not found, it will return nullptr.
   *
   * Note that only local/global named value could be decl/defined, upvalue could only be resolved.
   */

  struct NamedValue {
    std::string name;
    bool is_inited = false;
    int position = -1;  // have different meaning for local/global/upvalue, see their comment.
  };

  NamedValue* DeclNamedValue(Token var_name);
  void DefineNamedValue(NamedValue* handle);

  /**
   * A Local var will be stored in semantic depth 1 or deeper, any thing create a new semantic scope will deeper the
   * depth. eg. a nested function def, a if-else block, a for-loop, etc.
   */
  struct Local : public NamedValue {
    // position of local is offset on stack
    int semantic_scope_depth;
    bool is_captured = false;  // if the local is captured by a closure, it will get closed when it goes out of scope.
  };
  std::vector<Local> locals;  // contains currently active local vars, that is, not destroyed by "out of scoped" yet
  Local* TryResolveLocal(Token varaible_name);

  struct UpValue : public NamedValue {
    // position of upvalue is :
    //   1. offset in parent's stack slot, if it is is_on_stack_at_begin
    //   2. offset in parent's upvalues, if it is not is_on_stack_at_begin
    bool is_on_stack_at_begin = false;
    int corresponding_position = -1;
  };
  std::vector<UpValue> upvalues;  // contains upvalues current function will used
  UpValue* TryResolveUpValue(Token varaible_name);

  struct Global : public NamedValue {
    // position of global is its name's offset in constant table.
  };
  std::vector<Global> globals;  // contains globals current function will used
  Global* TryResolveGlobal(Token varaible_name);

  ///////////////////////////////////////////// NAME RESOLVE SUPPORT END////////////////////////////////////////////////
  struct JumpDownHole {
    OpCode jump_type;
    int beg_offset = -1;
    int hole_size = -1;  // entire hole size (contains the opcode)
  };

  struct LoopInfo {
    int initial_stack_size = 0;  // stack size at the beginning of the loop, when the loop ends, the stack size will be
                                 // restored to this value.
    bool contains_init_value = false;  // used to determine if a for-loop contains a newly created value ,that should
                                       // not be destroyed when continue.
    int beg_offset = -1;               // the loops first instruction offset
    std::vector<JumpDownHole> breaks;  // all the breaks in this loop that should be patched
  };
  std::vector<LoopInfo> loop_infos;  // loop could also be nested, so we need a vector to track all loops.

  FunctionUnit* enclosing_ = nullptr;
  ObjFunction* func;  // func is created by us, and managed by GC
  FunctionType type = FunctionType::UNKNOWN;
  int semantic_scope_depth = 0;  // the current tokens semantic scope depth, 0 means the global scope.
  // Note that 1. function name belongs to outer scope, function body is in the inner scope. 2. In our implementation,
  // class do not create new scope, and class can only define methods, and these methods will not be populated as named
  // value in scope.

  UpValue* AddUpValue(NamedValue* some_value, bool is_on_stack);
  Chunk* Chunk() { return func->chunk.get(); }
  void EmitByte(uint8_t byte);
  void EmitBytes(OpCode byte1, uint8_t byte2);
  void EmitByte(OpCode opcode);
  void EmitBytes(OpCode opcode0, OpCode opcode1);
  void EmitDefaultReturn();
  uint8_t MakeConstant(Value value);
  void EmitConstant(Value value) { EmitBytes(OpCode::OP_CONSTANT, MakeConstant(value)); }
  JumpDownHole CreateJumpDownHole(OpCode jump_cmd);
  void JumpHerePatch(JumpDownHole hole);
  void EmitJumpBack(int start);
  void CleanUpLocals(int local_var_num);

  uint8_t StoreTokenLexmeToConstant(Token token);

  void EmitUnary(const TokenType& token_type);
  void EmitBinary(const TokenType& token_type);
  void EmitLiteral(TokenType token_type);
  LineInfoCB line_info;
  bool IsGlobalScope() const;
  bool IsLocalAtOuterScope(const Local* local) const;
};

}  // namespace lox::vm

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_CORE_FUNCTION_UNIT_H
