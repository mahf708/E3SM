#include <iostream>
#include <string_view>

#include <dexpr/ast.hpp>
#include <dexpr/lexer.hpp>
#include <dexpr/parser.hpp>
#include <dexpr/supported_functions.hpp>

namespace {

void print_functions() {
    // The builtin seed only. A component registers its own vocabulary on top of
    // this at init, so what a given model accepts is a superset of this list.
    std::cout << "Builtin functions\n\n";

    for (const auto& entry : dexpr::builtin_functions()) {
        std::cout << "  " << entry.second << '\n';
    }
}

// Parse an expression and check its calls against the builtin vocabulary.
// A component's own functions are not known here, so this is a check of the
// grammar plus the builtins; a component checks against its own registry by
// calling validate_calls() with it.
int check_expression(std::string_view input) {
    dexpr::ast::ExprPtr expr;
    try {
        dexpr::parser::Parser parser{dexpr::Lexer{std::string{input}}};
        expr = parser.parse();
    } catch (const std::exception& e) {
        std::cerr << "does not parse:\n" << e.what() << '\n';
        return 1;
    }

    std::cout << "parsed as: " << dexpr::ast::to_string(*expr) << '\n';

    try {
        dexpr::validate_calls(*expr, dexpr::builtin_functions());
    } catch (const dexpr::ValidationError& e) {
        std::cerr << e.what();
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}

// Prove the builtin vocabulary describes itself correctly. A component runs the
// same check over its own registry after adding a function.
int check_registry() {
    try {
        dexpr::validate_registry(dexpr::builtin_functions());
    } catch (const dexpr::ValidationError& e) {
        std::cerr << e.what();
        return 1;
    }
    std::cout << "builtin functions: "
              << dexpr::builtin_functions().names().size() << " ok\n";
    return 0;
}

void print_help() {
    std::cout <<
R"(Usage:
    dexpr functions          list the builtin functions
    dexpr check <expr>       parse <expr> and check its calls
    dexpr check-registry     check every builtin function against its example
    dexpr help
)";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 1) {
        print_help();
        return 0;
    }

    std::string_view command{argv[1]};

    if (command == "functions") {
        print_functions();
        return 0;
    }

    if (command == "check") {
        if (argc < 3) {
            std::cerr << "check needs an expression\n";
            return 1;
        }
        return check_expression(argv[2]);
    }

    if (command == "check-registry") {
        return check_registry();
    }

    if (command == "help") {
        print_help();
        return 0;
    }

    std::cerr << "Unknown command: " << command << '\n';
    return 1;
}
