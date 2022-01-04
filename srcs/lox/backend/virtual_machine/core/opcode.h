//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BYTECODE_OPCODE_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BYTECODE_OPCODE_H

#include <cstdint>
namespace lox::vm {
enum class OpCode : uint8_t {
  OP_CONSTANT,  // 2 Bytes of [op_code, constant_index]
  OP_NIL,
  OP_TRUE,
  OP_FALSE,
  OP_POP,
  OP_GET_LOCAL,      // 2 Bytes of [op_code, offset_from_function_slots]
  OP_SET_LOCAL,      // 2 Bytes of [op_code, offset_from_function_slots]
  OP_GET_GLOBAL,     // 2 Bytes of [op_code, constant_index]
  OP_DEFINE_GLOBAL,  // 2 Bytes of [op_code, global_var_name_cst_index], read global var name from constant, move the
                     // stack top value to the global
  OP_SET_GLOBAL,     // 2 Bytes of [op_code, constant_index]
  OP_GET_UPVALUE,    // 2 Bytes of [op_code, offset_of_upvalue]
  OP_SET_UPVALUE,    // 2 Bytes of [op_code, offset_of_upvalue]
  OP_EQUAL,
  OP_GREATER,
  OP_LESS,
  OP_NOT,
  OP_NEGATE,
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_DIVIDE,
  OP_PRINT,
  OP_JUMP,           // 3 bytes of [opcode,[hi_jump_offset,lo_jump_offset]]
  OP_JUMP_IF_FALSE,  // 3 bytes of [opcode,[hi_jump_offset,lo_jump_offset]]
  OP_JUMP_BACK,      // 3 bytes of [opcode,[hi_jump_offset,lo_jump_offset]]
  OP_CALL,           // 2 Bytes of [op_code,arg_count]
  OP_CLOSURE,        // dynamic size of [op_code,upvalue_count, N*[isLocal,offset_of_slot_or_upvalue]]
  OP_CLOSE_UPVALUE,  // move a stack value to heap
  OP_RETURN,
  OP_CLASS,     // 2 Bytes of [op_code, class_name_cst_index], Read class name from constant, and leave a class obj on
                // stack
  OP_SET_ATTR,  // 2 Bytes of [op_code, constant_index]
  OP_GET_ATTR,  // 2 Bytes of [op_code, constant_index]
  OP_METHOD,    // 2 Bytes of [op_code, constant_index]
  OP_INVOKE,    // 3 Bytes of [op_code, method_name_constant_index,arg_count]
  OP_INHERIT,
};
}  // namespace lox::vm

// Macros that directly related to OpCode
#define U8_LOOKUP_MAX (UINT8_MAX + 1)
#define U16_LOOKUP_MAX (UINT16_MAX + 1)

#define STACK_LOOKUP_MAX U8_LOOKUP_MAX
#define UPVALUE_LOOKUP_MAX U8_LOOKUP_MAX
#define ARG_COUNT_MAX STACK_LOOKUP_MAX
#define GLOBAL_LOOKUP_MAX U8_LOOKUP_MAX
// Stack uses relative address , and command has a 256 limit,
// so a command can find only in [frame_pointer,frame_pointer+256]

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BYTECODE_OPCODE_H
