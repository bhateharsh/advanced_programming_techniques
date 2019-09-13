#include <cstring>
#include <simple_string.h>
u32 simple_string::default_count = 0;
u32 simple_string::create_count = 0;
u32 simple_string::copy_count = 0;
u32 simple_string::assign_count = 0;
u32 simple_string::destruct_count = 0;
u32 simple_string::move_count = 0;
u32 simple_string::move_assign = 0;
simple_string::simple_string() : st(nullptr) {
    default_count++;
}
simple_string::simple_string(const char* s) {
    st = (char*)malloc(strlen(s) + 1);
    strcpy(st, s);
    create_count++;
}
simple_string::simple_string(simple_string&& mv) noexcept {
    st = mv.st;
    mv.st = nullptr;
    move_count++;
}
simple_string::simple_string(const simple_string& s) : st(nullptr) {
    if(s.c_str() != nullptr) {
        st = (char*)malloc(strlen(s.c_str()) + 1);
        strcpy(st, s.c_str());
    }
    copy_count++;
}
simple_string::~simple_string() {
    if(st != nullptr) {
        free(st);
        st = nullptr;
    }
    destruct_count++;
}
simple_string& simple_string::operator=(const simple_string& rhs) {
    if(&rhs != this) {
        free(st);
        st = (char*)malloc(strlen(rhs.c_str()) + 1);
        strcpy(st, rhs.c_str());
    }
    assign_count++;
    return *this;
}
simple_string& simple_string::operator=(simple_string&& rhs) noexcept {
    if(&rhs != this) {
        if(st != nullptr) {
            free(st);
        }
        st = rhs.st;
        rhs.st = nullptr;
    }
    move_assign++;
}
const char* simple_string::c_str() const {
    return st;
}
void simple_string::initialize_counts() {
    default_count = 0;
    create_count = 0;
    copy_count = 0;
    assign_count = 0;
    destruct_count = 0;
    move_count = 0;
    move_assign = 0;
}
void simple_string::print_counts() {
    std::cout << "Default: " << default_count << "\n";
    std::cout << "Create: " << create_count << "\n";
    std::cout << "Copy: " << copy_count << "\n";
    std::cout << "Assign: " << assign_count << "\n";
    std::cout << "Destruct: " << destruct_count << "\n";
    std::cout << "Move Construct: " << move_count << "\n";
    std::cout << "Move Assign: " << move_assign << "\n";
}
std::ostream& operator<<(std::ostream& os, const simple_string& st) {
    os << st.c_str();
    return os;
}