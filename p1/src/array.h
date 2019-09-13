#pragma once

#include <iostream>
#include <string>
//students should not change this

using size_t = std::size_t;

template<typename T> class array_iterator;

template<typename T>
class array {
public:
    //default constructor
    array(){
        m_elements = nullptr;
        m_size = 0;
        m_reserved_size = 0;
    }

    //initialize array with elements in initializer
    array(std::initializer_list<T> l){  
        m_size = l.size();
        m_reserved_size = l.size();
        m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
        for (size_t i=0; i< m_size; i++){
            new(&m_elements[i]) (T)(*(l.begin()+i));
        }
    }

    //copy constructor
    array(const array& cpy):m_size(cpy.m_size), m_reserved_size(cpy.m_reserved_size), m_elements(cpy.m_elements){
        m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
        for (size_t i = 0; i < m_size; i ++){
            new(&m_elements[i]) (T)(cpy.m_elements[i]);
        }        
    }

    //move constructor
    array(array&& cpy):m_size(cpy.m_size), m_reserved_size(cpy.m_reserved_size), m_elements(cpy.m_elements){
        cpy.m_elements = nullptr;    
    }

    // //construct with initial "reserved" size
    array(size_t reserve_size){
        m_size = 0;
        m_reserved_size = reserve_size;
        m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
    }

    // //construct with n copies of t
    array(size_t n, const T& t){
        m_size = n;
        m_reserved_size = n;
        m_elements = (T*)malloc(sizeof(T)*n);
        for (size_t i = 0; i < m_size; i ++){
            new(&m_elements[i]) (T)(t);
        }
    }

    //destructor
    ~array(){
        if (m_elements != nullptr){
            for (size_t i = 0; i < m_size; i++)
	        {
	            m_elements[i].~T(); 
	        }
            free(m_elements);
        }
    }
    //display function
    void display(){
        std::cout<<"Printing the array"<<std::endl;
        std::cout<<"m_size = "<<m_size<<std::endl;
        for (size_t i = 0; i < m_size; i ++){
            std::cout<<"m_elements["<<i<<"] = "<<m_elements[i]<<std::endl;
        }
    }
    //ensure enough memory for n elements
    void reserve(size_t n){
        if (m_elements == nullptr){
        m_size = 0;
        m_reserved_size = n;
        m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
        }
        else{
            m_reserved_size += n;
            T* new_m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
            for (size_t i = 0; i < m_size; i ++){
                new(&new_m_elements[i]) (T)(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            free(m_elements);
            m_elements = new_m_elements;
            new_m_elements = nullptr;
        }
    }

    //add to end of vector
    void push_back(const T& element){
        if (m_size == m_reserved_size){
            T* new_m_elements = nullptr;
            m_reserved_size ++;
            new_m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
            for (size_t i=0; i < m_size; i ++){
                new(&new_m_elements[i]) (T)(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            free(m_elements);
            m_elements = new_m_elements;
            new_m_elements = nullptr;
        }
        m_size ++;
        new(&m_elements[m_size-1]) (T)(element);
    }

    //add to front of vector
    void push_front(const T& element){
        if (m_size == m_reserved_size){
            m_reserved_size ++;
            T* new_m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
            for (size_t i=0; i < m_size; i ++){
                new(&new_m_elements[i+1]) (T)(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            free(m_elements);
            m_elements = new_m_elements;
            new_m_elements = nullptr;
        }
        else{
            for (size_t i=0; i < m_size; i ++){
                m_elements[i+1] = m_elements[i];
            }
        }
        m_size ++;
        
        new(&m_elements[0]) (T)(element);
    }

    //remove last element
    void pop_back(){
        m_elements[m_size-1].~T();
        m_size --;
    }
    
    //remove first element
    void pop_front(){
        for (size_t i = 1; i < m_size; i ++){
            m_elements[i-1] = std::move(m_elements[i]);
        }
        m_elements[m_size-1].~T();
        m_size --;
    }

    //return reference to first element
    T& front() const{
        return m_elements[0];
    }

    //return reference to last element
    T& back() const{
        return m_elements[m_size-1];
    }

    //return reference to specified element
    const T& operator[](size_t idx) const{
        return m_elements[idx];
    }

    //return reference to specified element
    T& operator[](size_t idx){
        return m_elements[idx];
    }

    //return number of elements
    size_t length() const{
        return m_size;
    }

    //returns true if empty
    bool empty() const{
        if (m_size == 0 || m_elements == nullptr){
            return true;
        }
        else{
            return false;
        }
    }

    //remove all elements
    void clear(){
        if (m_elements != nullptr){
            for (int i = 0; i < m_size; i ++){
                m_elements[i].~T();
            }
            free(m_elements);
        }
        m_size = 0;
        m_reserved_size = 0;
    }

    //obtain iterator to first element
    array_iterator<T> begin() const{
        return (array_iterator<T>(m_elements));    
    }

    //obtain iterator to one beyond element*v*v
    array_iterator<T> end() const{
        return (array_iterator<T>(m_elements + m_size));
    }

    //remove specified element
    void erase(const array_iterator<T>& itr){
        size_t index = 0;
        for (size_t i = 0; i < m_size; i++){
    	    if (&m_elements[i] == itr.m_current){
    		    index = i;
    		    break;
    	    }
        }
        for (size_t i = index; i < m_size-1; i++){
    	    m_elements[i] = std::move(m_elements[i+1]);
        }
        m_size--;
        m_elements[m_size].~T();        
    }

    //insert element right before itr
    void insert(const T& val, const array_iterator<T>& itr){
        size_t index = 0;
        for (size_t i = 0; i < m_size; i++){
            if (&m_elements[i] == itr.m_current){
                index = i;
                break;
            }
        }
        if (m_size == m_reserved_size){
            std::cout<<"IF";
            T* new_m_elements = nullptr;
            m_reserved_size ++;
            m_size++;
            new_m_elements = (T*)malloc(sizeof(T)*m_reserved_size);
            for (size_t i=0; i < m_size; i ++){
                if (i < index){
                    new(&new_m_elements[i]) (T)(std::move(m_elements[i]));
                    m_elements[i].~T();
                }
                else if (i == index){
                    new(&new_m_elements[i]) (T)(val);
                }
                else if (i > index){
                    new(&new_m_elements[i]) (T)(std::move(m_elements[i-1]));
                    m_elements[i-1].~T();
                }   
            }
            free(m_elements);
            m_elements = new_m_elements;
            new_m_elements = nullptr;
        }
        else{
            m_size ++;;
            for (size_t i=(m_size-1); i > index; i--){
                m_elements[i] = std::move(m_elements[i-1]);
            }
            m_elements[index].~T();
            new(&m_elements[index]) (T)(val);
        }
    }

private:
    T* m_elements;              //points to actual elements
    size_t m_size;              //number of elements
    size_t m_reserved_size;     //number of reserved elements
};

template<typename T>
class array_iterator {
public:
    array_iterator(){
        m_current = nullptr;
    }
    array_iterator(T* init){
        m_current = init;
    }
    array_iterator(const array_iterator& temp){
        m_current = temp.m_current;
    }
    T& operator*() const{
        return *m_current;
    }
    array_iterator operator++(int __unused){
        array_iterator<T> result;
        result.m_current = m_current;
        m_current++;
        return result;    
    }

    array_iterator operator++(){
        m_current = m_current + 1;
        return *this;
    }

    array_iterator operator--(int __unused){
        array_iterator<T> res(*this);
        m_current = m_current - 1;
        return res;
    }
    
    array_iterator operator--  (){
        m_current = m_current - 1;
        return *this;
    }

    bool operator != (const array_iterator& comp) const{
        return (m_current != comp.m_current);
    }
    bool operator == (const array_iterator& comp) const{
        return (m_current == comp.m_current);
    }

private:
    T* m_current;
    // I want my array class to be able to access m_current even though it's private
    // 'friend' allows this to happen
    friend class array<T>;
};