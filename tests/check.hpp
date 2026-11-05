/*
Shared check machinery for every test in tests/.

  CHECK(cond, context)          count a check; print expression, line and context on failure
  EXPECT_THROW_AS(Type, expr)   the expression must throw exactly Type (or a type derived from it)
  test::run(fn, name)           run one test function; an unexpected exception is a failure
  test::finish(suite)           print the summary; return 0 (pass) or 1 (fail) for main

Pure C++: usable from .cpp tests built with g++ and .cu tests built with nvcc.
*/

#pragma once

#include <cstdio>
#include <exception>
#include <string>

namespace test
{

// Counters live inside functions: a header cannot define a plain global variable without
// risking "multiple definition" if two files ever include it. A function-local static is
// created once and shared by every caller.
inline int& checks()
{
    static int n = 0;
    return n;
}

inline int& failures()
{
    static int n = 0;
    return n;
}

inline void check_impl(bool ok, const char* expr, int line, const std::string& context)
{
    ++checks();
    if (!ok)
    {
        ++failures();
        std::printf("FAIL line %d: %s  [%s]\n", line, expr, context.c_str());
    }
}

template <typename E, typename F>
void expect_throw_as(F&& f, const char* what, int line)
{
    ++checks();
    try
    {
        f();
    }
    catch (const E&)
    {
        return; // pass
    }
    catch (const std::exception& e)
    {
        ++failures();
        std::printf("FAIL line %d: %s threw the wrong exception type: %s\n", line, what, e.what());
        return;
    }
    catch (...)
    {
        ++failures();
        std::printf("FAIL line %d: %s threw a non-std exception\n", line, what);
        return;
    }
    ++failures();
    std::printf("FAIL line %d: %s did not throw\n", line, what);
}

inline void run(void (*fn)(), const char* name)
{
    try
    {
        fn();
    }
    catch (const std::exception& e)
    {
        ++failures();
        std::printf("FAIL: %s aborted with an unexpected exception: %s\n", name, e.what());
    }
}

inline int finish(const char* suite)
{
    std::printf("%s: %d failures in %d checks\n", suite, failures(), checks());
    return failures() == 0 ? 0 : 1;
}

} // namespace test

#define CHECK(cond, context) ::test::check_impl((cond), #cond, __LINE__, (context))
#define EXPECT_THROW_AS(Type, expr) ::test::expect_throw_as<Type>([&] { (void)(expr); }, #expr, __LINE__)