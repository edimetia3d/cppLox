//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BYTECODE_OPCODE_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BYTECODE_OPCODE_H

#include <cstdint>
namespace lox::vm {
enum class OpCode : uint8_t {
  OP_CONSTANT,   // 2 Bytes of [op_code, constant_index], read a global by str name defined by constant_index
  OP_NIL,        // Push a nil to stack
  OP_TRUE,       // Push a true to stack
  OP_FALSE,      // Push a false to stack
  OP_POP,        // Pop a value from stack
  OP_GET_LOCAL,  // 2 Bytes of [op_code, offset_from_function_slots], read a local by offset_from_function_slots, and
                 // push to stack
  OP_SET_LOCAL,  // 2 Bytes of [op_code, offset_from_function_slots], write a local by offset_from_function_slots,and
                 // push to stack
  OP_GET_GLOBAL, // 2 Bytes of [op_code, constant_index], read a global by str name defined by constant_index,and push
                 // to stack
  OP_DEFINE_GLOBAL, // 2 Bytes of [op_code, global_var_name_cst_index], read global var name from constant, pop and
                    // move the stack top value to the global
  OP_SET_GLOBAL,    // 2 Bytes of [op_code, constant_index]
  OP_GET_UPVALUE,   // 2 Bytes of [op_code, offset_of_upvalue]
  OP_SET_UPVALUE,   // 2 Bytes of [op_code, offset_of_upvalue]
  OP_EQUAL,         // Pop two values on stack and compare them, if equal, push true, else push false
  OP_GREATER,  // Pop two values on stack and compare them, if first is greater than second, push true, else push false
  OP_LESS,     // Pop two values on stack and compare them, if first is less than second, push true, else push false
  OP_NOT,      // Pop a value on stack, if it is true, push false, else push true
  OP_NEGATE,   // Pop a value on stack, and push its negation
  OP_ADD,      // Pop two values on stack, and push their sum
  OP_SUBTRACT, // Pop two values b (first) and a on stack, and push their difference a-b
  OP_MULTIPLY, // Pop two values on stack, and push their product
  OP_DIVIDE,   // Pop two values b (first) and a on stack, and push their quotient a/b
  OP_PRINT,    // Pop a value on stack, and print it
  OP_JUMP,     // 3 bytes of [opcode,[hi_jump_offset,lo_jump_offset]]
  OP_JUMP_IF_FALSE, // 3 bytes of [opcode,[hi_jump_offset,lo_jump_offset]]
  OP_JUMP_BACK,     // 3 bytes of [opcode,[hi_jump_offset,lo_jump_offset]]
  OP_CALL,          // 2 Bytes of [op_code,arg_count]
  OP_CLOSURE,       // dynamic size of [op_code, upvalue_count, N*[isLocal,offset_of_slot_or_upvalue]]
  OP_CLOSE_UPVALUE, // move a stack value to heap
  OP_RETURN,        // Pop a value on stack, and switch call frame to caller, and push the value to the caller's stack
  OP_CLASS,    // 2 Bytes of [op_code, class_name_cst_index], Read class name from constant, and leave a class obj on
               // stack
  OP_SET_ATTR, // 2 Bytes of [op_code, constant_index]
  OP_GET_ATTR, // 2 Bytes of [op_code, constant_index]
  OP_METHOD,   // 2 Bytes of [op_code, constant_index]
  OP_INVOKE,   // 3 Bytes of [op_code, method_name_constant_index,arg_count]
  OP_INHERIT,  // Pop the baseclass (first) and the subclass (second) on stack, and update baseclass's dict, then push
               // the updated subclass to the stack
  OP_INSTANCE_TYPE_CAST // 1 bytes command ,read class and instance from stack
};
} // namespace lox::vm

// Macros that directly related to OpCode
#define U8_LOOKUP_UP_LIMIT (UINT8_MAX + 1)
#define U16_LOOKUP_LIMIT (UINT16_MAX + 1)

#define STACK_COUNT_LIMIT U8_LOOKUP_UP_LIMIT
#define CONSTANT_COUNT_LIMIT U8_LOOKUP_UP_LIMIT
#define UPVALUE_COUNT_LIMIT U8_LOOKUP_UP_LIMIT

#endif // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BYTECODE_OPCODE_H
