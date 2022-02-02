//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_TREE_WALKER_EVALUATOR_RUNTIME_OBJECT_H
#define LOX_SRCS_LOX_BACKEND_TREE_WALKER_EVALUATOR_RUNTIME_OBJECT_H

#include "lox/ast/ast.h"
#include "lox/backend/tree_walker/evaluator/environment.h"
#include "lox/object/object.h"
#include "runtime_error.h"

namespace lox::twalker {

template <class DataType_>
class TWObject : public lox::Object {
 public:
  using DataType = DataType_;
  explicit TWObject(const DataType_& data_) : data(data_) {}
  explicit TWObject(DataType_&& data_) : data(std::move(data_)) {}

  std::vector<Object*> References() override { return {}; }

  DataType_ data;
};

struct ReturnValueException : public std::exception {
  explicit ReturnValueException(ObjectPtr obj) : ret(std::move(obj)) {}

  ObjectPtr ret;
};

class Evaluator;
class ICallable {
 public:
  virtual int Arity() = 0;
  /**
   * A callable could:
   * 1. return value by throw an ReturnValueException
   * 2. return value by return a value
   *
   * note: for we did ot use enable_shared_from_this, we can not get shared this from definition,
   * the param this_in_sp should be given by caller
   */
  virtual ObjectPtr Call(Evaluator* evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) = 0;
};

class Bool : public TWObject<bool> {
 public:
  using TWObject<DataType>::TWObject;
  bool IsTrue() const override { return data; }
  std::string Str() const override { return (data ? "true" : "false"); }
};

class Number : public TWObject<double> {
 public:
  using TWObject<DataType>::TWObject;
  bool IsTrue() const override { return data; }
  std::string Str() const override {
    std::vector<char> buf(30);
    snprintf(buf.data(), buf.size(), "%g", data);
    return buf.data();
  }
};

class String : public TWObject<std::string> {
 public:
  using TWObject<DataType>::TWObject;
  bool IsTrue() const override { return !data.empty(); }
  std::string Str() const override { return data; }
};

class Nil : public lox::Object {
 public:
  bool IsTrue() const override { return false; }
  std::string Str() const override { return "nil"; }
  std::vector<Object*> References() override { return {}; }
};

class Klass;
struct InstanceData {
  using DictT = std::map<std::string, ObjectPtr>;
  InstanceData() noexcept;
  ObjectPtr klass;
  DictT& dict() { return *dict_; }

 private:
  std::shared_ptr<DictT> dict_;
};

class Instance : public TWObject<InstanceData> {
 public:
  using TWObject<DataType>::TWObject;
  bool IsTrue() const override { return static_cast<bool>(this); }
  std::string Str() const override;
};

class Environment;
struct ClosureData {
  EnvPtr closed;  // note that when created, `closed` is a capture of the caller's environment
  FunctionStmt* function;
};

class Closure : public TWObject<ClosureData>, public ICallable {
 public:
  using TWObject<DataType>::TWObject;
  ObjectPtr Call(Evaluator* evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) override;
  int Arity() override;
  std::string Str() const override;
};

class Klass;
struct KlassData {
  std::string name;
  ObjectPtr superclass;
  std::map<std::string, ObjectPtr> methods;
};

class Klass : public TWObject<KlassData>, public ICallable {
 public:
  using TWObject<DataType>::TWObject;
  int Arity() override;
  ObjectPtr Call(Evaluator* evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) override;
  std::string Str() const override;

  /**
   * GetMethod will always return a bounded method (closure object will "this" in environment)
   */
  ObjectPtr GetMethod(ObjectPtr object_this, const std::string& name);

  void AddMethod(const std::string& name, ObjectPtr method);
};

}  // namespace lox::twalker

#endif  // LOX_SRCS_LOX_BACKEND_TREE_WALKER_EVALUATOR_RUNTIME_OBJECT_H
