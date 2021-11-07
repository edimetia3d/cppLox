//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/common/clox_value.h"

#include <spdlog/spdlog.h>

#include <cstdio>

#include "lox/backend/virtual_machine/bytecode/chunk.h"

namespace lox {

void printValue(const vm::Object &value, bool print_to_debug) {
#define q_printf(...)          \
  if (print_to_debug) {        \
    SPDLOG_DEBUG(__VA_ARGS__); \
  } else {                     \
    printf(__VA_ARGS__);       \
  };                           \
  break
  switch (value.Type()) {
    case vm::ObjectType::BOOL:
      q_printf((value.AsBool() ? "true" : "false"));
    case vm::ObjectType::NUMBER:
      q_printf("%f", value.AsNumber());
    case vm::ObjectType::NIL:
      q_printf("nil");
    case vm::ObjectType::OBJ_HANDLE:
      value.AsHandle()->Print(print_to_debug);
      break;
    default:
      q_printf("Unkown types");
  }
#undef q_printf
}

namespace vm {

static uint32_t fnv_1a(uint8_t *data, int size) {
  uint32_t new_hash = 2166136261u;
  for (int i = 0; i < size; i++) {
    new_hash ^= data[i];
    new_hash *= 16777619;
  }
  return new_hash;
}
bool lox::vm::ObjHandle::Equal(const lox::vm::ObjHandle *rhs) const {
  return this == rhs;  // all string are interned, so we can compare directly
}
void ObjHandle::Print(bool print_to_debug) const {
#define q_printf(...)          \
  if (print_to_debug) {        \
    SPDLOG_DEBUG(__VA_ARGS__); \
  } else {                     \
    printf(__VA_ARGS__);       \
  };                           \
  break
  switch (type) {
    case ObjType::OBJ_STRING:
      q_printf("%s", As<ObjInternedString>()->c_str());
    case ObjType::OBJ_FUNCTION:
      q_printf("<fn %s>", As<ObjFunction>()->name.c_str());
    case ObjType::OBJ_NATIVE_FUNCTION:
      q_printf("<native fn>");
    case ObjType::OBJ_RUNTIME_FUNCTION: {
      auto p = As<ObjRuntimeFunction>();
      if (p->isClosure()) {
        q_printf("<closure %s>", As<ObjRuntimeFunction>()->function->name.c_str());
      } else {
        q_printf("<fn %s>", As<ObjRuntimeFunction>()->function->name.c_str());
      }
    }
    case ObjType::OBJ_CLASS:
      q_printf("<class %s>", As<ObjClass>()->name.c_str());
    case ObjType::OBJ_UPVALUE:
      q_printf("upvalue");
    case ObjType::OBJ_INSTANCE:
      q_printf("%s instance", As<ObjInstance>()->klass->name.c_str());
    case ObjType::OBJ_BOUND_METHOD:
      q_printf("<bound method %s>", As<ObjBoundMethod>()->method->function->name.c_str());
    default:
      q_printf("Unknown ObjHandle type");
  }
#undef q_printf
}
std::set<ObjHandle *> &ObjHandle::AllCreatedObj() {
  static std::set<ObjHandle *> ret;
  return ret;
}
int &ObjHandle::ObjCount() {
  static int value;
  return value;
}
uint32_t ObjInternedString::UpdateHash() {
  hash = fnv_1a((uint8_t *)c_str(), size());
  return hash;
}
ObjInternedString *ObjInternedString::Concat(const ObjInternedString *lhs, const ObjInternedString *rhs) {
  Buffer<char> buf;
  buf.reserve(lhs->size() + rhs->size() + 1);
  buf.push_buffer(lhs->c_str(), lhs->size());
  buf.push_buffer(rhs->c_str(), rhs->size());
  buf.push_back('\0');
  InternView view{.data = buf.data(), .hash = fnv_1a((uint8_t *)buf.data(), buf.size() - 1), .size = (buf.size() - 1)};
  auto entry = GetInternMap().Get(view);
  if (entry) {
    return entry->value;
  } else {
    return new ObjInternedString(std::move(buf));
  }
}
int ObjInternedString::size() const {
  return data.size() - 1;  // this is a c style str
}
ObjInternedString::ObjInternedString(const char *buf, int size) {
  data.push_buffer(buf, size);
  data.push_back('\0');
  UpdateHash();
  TryInternThis();
}
void ObjInternedString::TryInternThis() {
  if (!GetInternMap().Get(GetInternView())) {
    GetInternMap().Set(GetInternView(), this);
  } else {
    printf("Warning: Get repeated intern string\n");
  }
}
ObjInternedString::ObjInternedString(Buffer<char> &&buffer) {
  data = std::move(buffer);
  UpdateHash();
  TryInternThis();
}
ObjInternedString::InternMap &ObjInternedString::GetInternMap() {
  static InternMap interning_map(32);
  return interning_map;
}
ObjInternedString *ObjInternedString::Make(const char *data, int size) {
  InternView view{.data = data, .hash = fnv_1a((uint8_t *)data, size), .size = size};
  auto entry = GetInternMap().Get(view);
  if (entry) {
    return entry->value;
  } else {
    return new ObjInternedString(data, size);
  }
}
ObjInternedString::~ObjInternedString() { GetInternMap().Del(GetInternView()); }

void ObjHandle::Destroy(ObjHandle *obj) {
  SPDLOG_DEBUG("Object [%p] with type [%d] deleted.", obj, obj->type);
  switch (obj->type) {
    case ObjType::OBJ_STRING:
      delete obj->As<ObjInternedString>();
      break;
    case ObjType::OBJ_FUNCTION:
      delete obj->As<ObjFunction>();
      break;
    case ObjType::OBJ_NATIVE_FUNCTION:
      delete obj->As<ObjNativeFunction>();
      break;
    case ObjType::OBJ_RUNTIME_FUNCTION:
      delete obj->As<ObjRuntimeFunction>();
      break;
    case ObjType::OBJ_UPVALUE:
      delete obj->As<ObjUpvalue>();
      break;
    case ObjType::OBJ_CLASS: {
      delete obj->As<ObjClass>();
      break;
    }
    case ObjType::OBJ_INSTANCE: {
      delete obj->As<ObjInstance>();
      break;
    }
    case ObjType::OBJ_BOUND_METHOD:
      delete obj->As<ObjBoundMethod>();
      break;
    default:
      printf("Destroying Unknown Type.\n");
  }
}
ObjHandle::ObjHandle(ObjType type) : type(type), isMarked(false) {
  SPDLOG_DEBUG("Object [%p] with type [%d] created", this, type);
  AllCreatedObj().insert(this);
  ++ObjCount();
}
ObjHandle::~ObjHandle() {
  --ObjCount();
  AllCreatedObj().erase(this);
}
void ObjHandle::MarkReference(ObjHandle *obj) {
  auto &gc = GC::Instance();
  switch (obj->type) {
    case ObjType::OBJ_STRING:
      break;
    case ObjType::OBJ_FUNCTION: {
      int n = obj->As<ObjFunction>()->chunk->constants.size();
      auto p = obj->As<ObjFunction>()->chunk->constants.data();
      for (int i = 0; i < n; ++i) {
        gc.mark(p[i]);
      }
      break;
    }
    case ObjType::OBJ_NATIVE_FUNCTION:
      break;
    case ObjType::OBJ_RUNTIME_FUNCTION: {
      gc.mark(obj->As<ObjRuntimeFunction>()->function);
      for (int i = 0; i < obj->As<ObjRuntimeFunction>()->upvalueCount; ++i) {
        gc.mark(obj->As<ObjRuntimeFunction>()->upvalues[i]);
      }
      break;
    }
    case ObjType::OBJ_UPVALUE: {
      auto p = obj->As<ObjUpvalue>();
      gc.mark(*p->location);
      break;
    }
    case ObjType::OBJ_CLASS: {
      ObjClass *klass = obj->As<ObjClass>();
      gc.mark(klass->methods);
      if (klass->superclass) {
        gc.mark(klass->superclass);
      }
      break;
    }
    case ObjType::OBJ_INSTANCE: {
      ObjInstance *instance = obj->As<ObjInstance>();
      gc.mark(instance->klass);
      gc.mark(instance->dict);
      break;
    }
    case ObjType::OBJ_BOUND_METHOD: {
      ObjBoundMethod *bound = obj->As<ObjBoundMethod>();
      gc.mark(bound->receiver);
      gc.mark(bound->method);
      break;
    }
    default:
      break;
  }
}

ObjFunction::ObjFunction() { chunk = new Chunk(); }
ObjFunction::~ObjFunction() { delete chunk; }
bool Object::Equal(Object rhs) {
  if (type != rhs.type) return false;
  switch (type) {
    case ObjectType::BOOL:
      return AsBool() == rhs.AsBool();
    case ObjectType::NIL:
      return true;
    case ObjectType::NUMBER:
      return AsNumber() == rhs.AsNumber();
    case ObjectType::OBJ_HANDLE:
      return rhs.IsHandle() && AsHandle()->Equal(rhs.AsHandle());
    default:
      return false;  // Unreachable.
  }
}
GC &GC::Instance() {
  static GC obj;
  return obj;
}
void GC::collectGarbage() {
  SPDLOG_DEBUG("-- gc begin");
  markRoots();
  Sweep();
  SPDLOG_DEBUG("-- gc end");
}
void GC::mark(Object value) {
  if (value.IsHandle()) mark(value.AsHandle());
}
void GC::mark(ObjHandle *object) {
  if (object == nullptr || object->isMarked) return;
  SPDLOG_DEBUG("%p mark ", (void *)object);
  lox::printValue(Object(object), true);
  object->isMarked = true;
  ObjHandle::MarkReference(object);
}
void GC::mark(HashMap<ObjInternedString *, Object, ObjInternedString::Hash> *table) {
  auto iter = table->GetAllItem();
  while (auto entry = iter.next()) {
    mark(entry->key);
    mark(entry->value);
  }
}
void GC::Sweep() {
  auto &list = ObjHandle::AllCreatedObj();
  auto iter = list.begin();
  std::set<ObjHandle *> to_del;
  while (iter != list.end()) {
    if ((*iter)->isMarked) {
      (*iter)->isMarked = false;
    } else {
      to_del.insert((*iter));
    }
    ++iter;
  }
  iter = to_del.begin();
  while (iter != to_del.end()) {
    ObjHandle::Destroy((*iter));
    ++iter;
  }
}
}  // namespace vm
}  // namespace lox
