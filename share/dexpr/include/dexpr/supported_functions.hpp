/**
 * @file supported_functions.hpp
 * @brief The set of callables a component makes available to expressions.
 *
 * dexpr owns the grammar; a component owns the vocabulary. The parser accepts
 * any call syntactically -- `foo(a, b=c)` parses the same whether or not `foo`
 * exists -- and a component declares what it can actually evaluate by filling a
 * FunctionRegistry at init and running validate_calls() over the AST.
 *
 * That split is deliberate: parsing stays independent of who is asking, so the
 * same expression can be checked against different components' vocabularies,
 * and a component adds a callable without editing anything in this library.
 */
#ifndef DEXPR_SUPPORTED_FUNCTIONS_HPP
#define DEXPR_SUPPORTED_FUNCTIONS_HPP

#include <dexpr/ast.hpp>

#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dexpr {

// A keyword argument a function accepts, e.g. the `dims` of
// `derivative(dx, dims=['col'])`.
struct ParamSpec {
  std::string name;
  bool required = false;
};

// How a call may be written. Most component functions read best in one form
// only: `where(x)` free-standing is meaningless without an operand, while
// `x.tend()` reads wrong as `tend(x)`.
enum class CallForm {
  Any,     // both `f(x)` and `x.f()`
  Free,    // only `f(x)`
  Method,  // only `x.f()`
};

struct FunctionSpec {
  std::string name;
  std::string desc;

  // Positional arity, not counting keyword arguments. -1 max means variadic.
  int min_positional = 0;
  int max_positional = 0;

  std::vector<ParamSpec> keywords;

  CallForm form = CallForm::Any;

  // "name(arg, kw=..)\n--- desc", the listing the `dexpr` tool prints.
  std::string to_string() const;
};

inline std::ostream& operator<<(std::ostream& os, const FunctionSpec& function) {
  return os << function.to_string() << '\n';
}

// The callables one component makes available. Ordered, so listings and error
// messages are stable rather than hash-order.
class FunctionRegistry {
public:
  // Throws std::invalid_argument if the name is already registered: two specs
  // for one name means one of them is silently dead.
  void add(FunctionSpec spec);

  const FunctionSpec* find(std::string_view name) const;
  bool contains(std::string_view name) const { return find(name) != nullptr; }

  std::vector<std::string> names() const;
  bool empty() const { return fns_.empty(); }
  std::size_t size() const { return fns_.size(); }

  auto begin() const { return fns_.begin(); }
  auto end() const { return fns_.end(); }

  // Every spec, one per line, for help text and error messages.
  std::string to_string() const;

private:
  // std::less<> so lookups take a string_view without allocating.
  std::map<std::string, FunctionSpec, std::less<>> fns_;
};

// The generic callables that used to be hard-coded here. A component may seed
// its registry from this or start empty; nothing consults it implicitly.
const FunctionRegistry& builtin_functions();

class ValidationError : public std::runtime_error {
public:
  explicit ValidationError(const std::vector<std::string>& errors);
};

// Checks every call in the tree against `reg`: that the callee is a plain name,
// that the name is registered, that positional arity fits, that each keyword
// argument names a declared parameter, that required keywords are present, and
// that the call form matches. Collects every problem and throws once.
//
// Kept out of the parser on purpose -- see the file comment.
void validate_calls(const ast::Expression& root, const FunctionRegistry& reg);

} // namespace dexpr

#endif
