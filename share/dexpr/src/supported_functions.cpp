#include <dexpr/supported_functions.hpp>

#include <dexpr/tokens.hpp>

#include <algorithm>
#include <set>
#include <type_traits>
#include <utility>

/**
 * @file supported_functions.cpp
 * @brief FunctionRegistry, the builtin seed, and the call-validation visitor.
 */
namespace dexpr {

std::string FunctionSpec::to_string() const {
  std::string str_{name};
  str_ += "(";

  bool first = true;
  for (int i = 0; i < min_positional; ++i) {
    if (!first) {
      str_ += ", ";
    }
    first = false;
    str_ += "<arg>";
  }
  for (int i = min_positional; i < max_positional; ++i) {
    if (!first) {
      str_ += ", ";
    }
    first = false;
    str_ += "[<arg>]";
  }
  for (const auto& kw : keywords) {
    if (!first) {
      str_ += ", ";
    }
    first = false;
    str_ += kw.required ? kw.name + "=.." : "[" + kw.name + "=..]";
  }

  str_ += ")";
  str_ += "\n--- " + desc;

  return str_;
}

void FunctionRegistry::add(FunctionSpec spec) {
  if (spec.name.empty()) {
    throw std::invalid_argument("dexpr: cannot register a function with no name");
  }
  // A silently-replaced spec is a bug that only shows up as a confusing
  // validation message much later, so refuse rather than overwrite.
  if (fns_.count(spec.name) > 0) {
    throw std::invalid_argument("dexpr: function '" + spec.name +
                                "' is already registered");
  }
  const auto name = spec.name;
  fns_.emplace(name, std::move(spec));
}

const FunctionSpec* FunctionRegistry::find(std::string_view name) const {
  const auto it = fns_.find(name);
  return it == fns_.end() ? nullptr : &it->second;
}

std::vector<std::string> FunctionRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(fns_.size());
  for (auto it = fns_.begin(); it != fns_.end(); ++it) {
    out.push_back(it->first);
  }
  return out;
}

const FunctionRegistry& builtin_functions() {
  static const FunctionRegistry reg = [] {
    FunctionRegistry r;
    r.add({.name = "where",
           .desc = "applies condition to operand",
           .min_positional = 1,
           .max_positional = 1,
           .keywords = {}});
    r.add({.name = "sum",
           .desc = "sums operand over designated indices (int or name)",
           .min_positional = 0,
           .max_positional = 0,
           .keywords = {{"dims", true}}});
    r.add({.name = "derivative",
           .desc = "takes derivative w.r.t. `dx` over designated dimension",
           .min_positional = 1,
           .max_positional = 1,
           .keywords = {{"dims", false}}});
    r.add({.name = "tend",
           .desc = "calculates the tendency of a variable over time",
           .min_positional = 0,
           .max_positional = 0,
           .keywords = {}});
    return r;
  }();
  return reg;
}

ValidationError::ValidationError(const std::vector<std::string>& errors)
    : std::runtime_error([&] {
        std::string result = "Validation errors:\n";
        for (const auto& error : errors) {
          result += "  - " + error + '\n';
        }
        return result;
      }()) {}

namespace {

// The callee of a call. `f(x)` puts an Identifier directly under
// FuncExpression::function; `x.f(y)` puts a BinaryExpression{Dot} there, whose
// right child is the name and whose left child is the receiver.
struct Callee {
  const std::string* name = nullptr; // null when the callee is not a plain name
  const ast::Expression* receiver = nullptr; // non-null for the method form
};

struct CalleeVisitor {
  Callee operator()(const ast::Identifier& e) const { return {&e.value, nullptr}; }
  Callee operator()(const ast::BinaryExpression& e) const {
    if (e.op != TokenTypes::Dot) {
      return {};
    }
    // The method name must itself be a plain identifier: `x.(a+b)()` is not a
    // call we can check, and `x.y.f()` recurses through the receiver instead.
    return e.right->visit([&](const auto& node) -> Callee {
      using T = std::decay_t<decltype(node)>;
      if constexpr (std::is_same_v<T, ast::Identifier>) {
        return {&node.value, e.left.get()};
      } else {
        return {};
      }
    });
  }
  template <typename T> Callee operator()(const T&) const { return {}; }
};

// Keyword arguments are Assign binary expressions in the argument list, so the
// two kinds are separated after parsing rather than during it.
const std::string* keyword_name(const ast::Expression& arg) {
  return arg.visit([](const auto& node) -> const std::string* {
    using T = std::decay_t<decltype(node)>;
    if constexpr (std::is_same_v<T, ast::BinaryExpression>) {
      if (node.op == TokenTypes::Assign) {
        return node.left->visit([](const auto& lhs) -> const std::string* {
          using L = std::decay_t<decltype(lhs)>;
          if constexpr (std::is_same_v<L, ast::Identifier>) {
            return &lhs.value;
          } else {
            return nullptr;
          }
        });
      }
    }
    return nullptr;
  });
}

class Validator {
public:
  explicit Validator(const FunctionRegistry& reg) : reg_(reg) {}

  void run(const ast::Expression& expr) {
    expr.visit(*this);
  }

  std::vector<std::string> take_errors() { return std::move(errors_); }

  void operator()(const ast::Identifier&) const {}
  void operator()(const ast::StringLiteral&) const {}
  void operator()(const ast::FloatLiteral&) const {}
  void operator()(const ast::IntegerLiteral&) const {}

  void operator()(const ast::UnaryExpression& e) { run(*e.right); }

  void operator()(const ast::BinaryExpression& e) {
    run(*e.left);
    run(*e.right);
  }

  void operator()(const ast::ArrayExpression& e) {
    for (const auto& el : e.elements) {
      run(*el);
    }
  }

  void operator()(const ast::FuncExpression& e) {
    check_call(e);
    // Recurse regardless of whether this call checked out, so one unknown name
    // does not hide the rest.
    if (const auto callee = e.function->visit(CalleeVisitor{}); callee.receiver) {
      run(*callee.receiver);
    } else if (!callee.name) {
      run(*e.function);
    }
    for (const auto& arg : e.args) {
      // For `f(x=y)` the interesting part is y; x is a parameter name, not a
      // subexpression to check.
      if (keyword_name(*arg) != nullptr) {
        arg->visit([&](const auto& node) {
          using T = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<T, ast::BinaryExpression>) {
            run(*node.right);
          }
        });
      } else {
        run(*arg);
      }
    }
  }

private:
  void check_call(const ast::FuncExpression& e) {
    const auto callee = e.function->visit(CalleeVisitor{});
    if (callee.name == nullptr) {
      errors_.push_back("call target is not a function name: " +
                        ast::to_string(*e.function));
      return;
    }
    const std::string& name = *callee.name;

    const auto* spec = reg_.find(name);
    if (spec == nullptr) {
      std::string msg = "unknown function '" + name + "'";
      const auto known = reg_.names();
      if (known.empty()) {
        msg += "; no functions are registered";
      } else {
        msg += "; available: ";
        for (std::size_t i = 0; i < known.size(); ++i) {
          msg += (i == 0 ? "" : ", ") + known[i];
        }
      }
      errors_.push_back(std::move(msg));
      return;
    }

    int positional = 0;
    std::set<std::string> seen_keywords;
    for (const auto& arg : e.args) {
      const auto* kw = keyword_name(*arg);
      if (kw == nullptr) {
        ++positional;
        continue;
      }
      const auto known = std::find_if(
          spec->keywords.begin(), spec->keywords.end(),
          [&](const ParamSpec& p) { return p.name == *kw; });
      if (known == spec->keywords.end()) {
        std::string msg = "'" + name + "' has no argument '" + *kw + "'";
        if (!spec->keywords.empty()) {
          msg += "; accepts: ";
          bool first = true;
          for (const auto& p : spec->keywords) {
            msg += (first ? "" : ", ") + p.name;
            first = false;
          }
        }
        errors_.push_back(std::move(msg));
      } else if (!seen_keywords.insert(*kw).second) {
        errors_.push_back("'" + name + "' got argument '" + *kw +
                          "' more than once");
      }
    }

    if (positional < spec->min_positional ||
        positional > spec->max_positional) {
      errors_.push_back("'" + name + "' takes " +
                        arity_to_string(*spec) + ", got " +
                        std::to_string(positional));
    }

    for (const auto& p : spec->keywords) {
      if (p.required && seen_keywords.count(p.name) == 0) {
        errors_.push_back("'" + name + "' requires argument '" + p.name + "'");
      }
    }
  }

  static std::string arity_to_string(const FunctionSpec& spec) {
    if (spec.min_positional == spec.max_positional) {
      return std::to_string(spec.min_positional) + " positional argument(s)";
    }
    return std::to_string(spec.min_positional) + " to " +
           std::to_string(spec.max_positional) + " positional argument(s)";
  }

  const FunctionRegistry& reg_;
  std::vector<std::string> errors_;
};

} // namespace

void validate_calls(const ast::Expression& root, const FunctionRegistry& reg) {
  Validator v{reg};
  v.run(root);
  auto errors = v.take_errors();
  if (!errors.empty()) {
    throw ValidationError(errors);
  }
}

} // namespace dexpr
