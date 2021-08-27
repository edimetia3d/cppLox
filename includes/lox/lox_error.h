//
// LICENSE: MIT
//

#ifndef LOX_INCLUDES_LOX_LOX_ERROR_H_
#define LOX_INCLUDES_LOX_LOX_ERROR_H_
#include <memory>
#include <string>
#include <vector>
namespace lox {
class LoxError {
 public:
  using ErrorNode = std::shared_ptr<LoxError>;
  LoxError();
  explicit LoxError(const std::string &message);
  std::string Message() const;

  int ToErrCode() const;
  void Merge(const LoxError &new_err);
  bool NoError();

 private:
  std::string message_{};
  std::string RecursiveMessage(int level) const;
  std::vector<ErrorNode> sub_errors;
};
}  // namespace lox

#endif  // LOX_INCLUDES_LOX_LOX_ERROR_H_
