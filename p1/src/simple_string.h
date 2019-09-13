//
// Created by bpswe on 9/24/2018.
//

#pragma once

#include <iostream>
#include <cstdint>

using u32 = uint32_t;

class simple_string {
public:
    simple_string();
    explicit simple_string(const char* str);
    simple_string(const simple_string& cpy);
    simple_string(simple_string&& mv) noexcept;
    ~simple_string();
    simple_string& operator=(const simple_string&);
    simple_string& operator=(simple_string&&) noexcept;
    const char* c_str() const;

    static void initialize_counts();
    static void print_counts();

private:
    char* st;

    static u32 default_count;
    static u32 create_count;
    static u32 copy_count;
    static u32 assign_count;
    static u32 destruct_count;
    static u32 move_count;
    static u32 move_assign;

};

std::ostream& operator<<(std::ostream&, const simple_string& st);