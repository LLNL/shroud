// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################

class Class1
{
public:
    int m_flag;
    Class1()         : m_flag(0)    {};
    Class1(int flag) : m_flag(flag) {};
};

int * ReturnIntPtrRaw();
int * ReturnIntPtrScalar();
int * ReturnIntPtrPointer();

int * ReturnIntPtrDimRaw(int *len);
int * ReturnIntPtrDimPointer(int *len);
int * ReturnIntPtrDimAlloc(int *len);
int * ReturnIntPtrDimDefault(int *len);

int * ReturnIntPtrDimRawNew(int *len);
int * ReturnIntPtrDimPointerNew(int *len);
int * ReturnIntPtrDimAllocNew(int *len);
int * ReturnIntPtrDimDefaultNew(int *len);

void IntPtrDimRaw(int **array, int *len);
void IntPtrDimPointer(int **array, int *len);
void IntPtrDimAlloc(int **array, int *len);
void IntPtrDimDefault(int **array, int *len);

void IntPtrDimRawNew(int **array, int *len);
void IntPtrDimPointerNew(int **array, int *len);
void IntPtrDimAllocNew(int **array, int *len);
void IntPtrDimDefaultNew(int **array, int *len);

void createClassStatic(int flag);
Class1 * getClassStatic();
Class1 * getClassNew(int flag);

//----------------------------------------------------------------------
// example lifted from pybindgen PyBindGen-0.18.0/tests/foo.h

#include <string>

// Deprecation warnings look ugly and confusing; better to just
// disable them and change this macro when we want to specifically
// test them.
#define ENABLE_DEPRECATIONS 0

#ifndef DEPRECATED
# if ENABLE_DEPRECATIONS && __GNUC__ > 2
#  define DEPRECATED  __attribute__((deprecated))
# else
#  define DEPRECATED
# endif
#endif



class Foo
// -#- automatic_type_narrowing=True -#-
{
    std::string m_datum;
    bool m_initialized;
public:
    static int instance_count;

    Foo () : m_datum (""), m_initialized (false)
        { Foo::instance_count++; }

#if 1
    Foo (int xpto)  DEPRECATED : m_initialized (false) { xpto++; }

    Foo (std::string const &datum) : m_datum (datum), m_initialized (false)
        { Foo::instance_count++; }
    const std::string get_datum () const { return m_datum; }

    std::string get_datum_deprecated () const DEPRECATED { return m_datum; }

    Foo (Foo const & other) : m_datum (other.get_datum ()), m_initialized (false)
        { Foo::instance_count++; }

    void initialize () { m_initialized = true; }
    bool is_initialized () const { return m_initialized; }

    virtual ~Foo() { Foo::instance_count--; }

    static int add_sub (int a, int b=3, bool subtract=false)
        {
            return a+b;
        }
#endif
};




class Zbr
// -#- incref_method=Ref; decref_method=Unref; peekref_method=GetReferenceCount -#-
{
    int m_refcount;
    std::string m_datum;
public:
    Zbr () : m_refcount (1), m_datum ("")
        { Zbr::instance_count++; }
    Zbr (std::string datum) :  m_refcount (1), m_datum (datum)
        { Zbr::instance_count++; }

    std::string get_datum () const { return m_datum; }

    Zbr (Zbr const & other) :
        m_refcount (1), m_datum (other.get_datum ())
        {Zbr::instance_count++;}

    void Ref () {
        // std::cerr << "Ref Zbr " << this << " from " << m_refcount << std::endl;
        ++m_refcount;
    }
    void Unref () {
        // std::cerr << "Unref Zbr " << this << " from " << m_refcount << std::endl;
        if (--m_refcount == 0)
            delete this;
    }
    int GetReferenceCount () const { return m_refcount; }

    virtual int get_int (int x) {
        return x;
    }

    static int instance_count;

    virtual ~Zbr () {
        --Zbr::instance_count;
    }

    // -#- @foobaz(transfer_ownership=true, direction=out) -#-
    int get_value (int* foobaz) { *foobaz = 123; return -1; }
};

// -#- @zbr(transfer_ownership=true) -#-
void store_zbr (Zbr *zbr);
int invoke_zbr (int x);
void delete_stored_zbr (void);

class SomeObject
{
public:
    std::string m_prefix;

    enum {
        TYPE_FOO,
        TYPE_BAR,
    } type;

    static int instance_count;


#if 0
    // A nested class
    class NestedClass
    // -#- automatic_type_narrowing=True -#-
    {
        std::string m_datum;
    public:
        static int instance_count;

        NestedClass () : m_datum ("")
            { Foo::instance_count++; }
        NestedClass (std::string datum) : m_datum (datum)
            { Foo::instance_count++; }
        std::string get_datum () const { return m_datum; }

        NestedClass (NestedClass const & other) : m_datum (other.get_datum ())
            { Foo::instance_count++; }

        virtual ~NestedClass() { NestedClass::instance_count--; }
    };

    // A nested enum
    enum NestedEnum {
        FOO_TYPE_AAA,
        FOO_TYPE_BBB,
        FOO_TYPE_CCC,
    };


    // An anonymous nested enum
    enum  {
        CONSTANT_A,
        CONSTANT_B,
        CONSTANT_C
    };


#endif
private:
    Foo m_foo_value;
    Foo *m_foo_ptr;
    Foo *m_foo_shared_ptr;
#if 0
    Zbr *m_zbr;
    Zbr *m_internal_zbr;

    PyObject *m_pyobject;
    Foobar *m_foobar;

    SomeObject ();
#endif
public:
#if 0

    static std::string staticData;

    virtual ~SomeObject ();
    SomeObject (const SomeObject &other);
    SomeObject (std::string const prefix);
    SomeObject (int prefix_len);

    // -#- @message(direction=inout) -#-
    int add_prefix (std::string& message) {
        message = m_prefix + message;
        return message.size ();
    }

    // -#- @message(direction=inout) -#-
    int operator() (std::string& message) {
        message = m_prefix + message;
        return message.size ();
    }

    // --------  Virtual methods ----------
    virtual std::string get_prefix () const {
        return m_prefix;
    }

    std::string call_get_prefix () const {
        return get_prefix();
    }

    virtual std::string get_prefix_with_foo_value (Foo foo) const {
        return m_prefix + foo.get_datum();
    }

    // -#- @foo(direction=inout) -#-
    virtual std::string get_prefix_with_foo_ref (const Foo &foo) const {
        return m_prefix + foo.get_datum ();
    }

    virtual std::string get_prefix_with_foo_ptr (const Foo *foo) const {
        return m_prefix + foo->get_datum ();
    }


    // A couple of overloaded virtual methods
    virtual std::string get_something () const {
        return "something";
    }
    virtual std::string get_something (int x) const {
        std::stringstream out;
        out << x;
        return out.str ();
    }

    // -#- @pyobject(transfer_ownership=false) -#-
    virtual void set_pyobject (PyObject *pyobject) {
        if (m_pyobject) {
            Py_DECREF(m_pyobject);
        }
        Py_INCREF(pyobject);
        m_pyobject = pyobject;
    }

    // -#- @return(caller_owns_return=true) -#-
    virtual PyObject* get_pyobject (void) {
        if (m_pyobject) {
            Py_INCREF(m_pyobject);
            return m_pyobject;
        } else {
            return NULL;
        }
    }

    // pass by value, direction=in
    void set_foo_value (Foo foo) {
        m_foo_value = foo;
    }

    // pass by reference, direction=in
    void set_foo_by_ref (const Foo& foo) {
        m_foo_value = foo;
    }

    // pass by reference, direction=out
    // -#- @foo(direction=out) -#-
    void get_foo_by_ref (Foo& foo) {
        foo = m_foo_value;
    }
#endif

    // -#- @foo(transfer_ownership=true) -#-
    void set_foo_ptr (Foo *foo) {
        if (m_foo_ptr)
            delete m_foo_ptr;
        m_foo_ptr = foo;
    }

    // -#- @foo(transfer_ownership=false) -#-
    void set_foo_shared_ptr (Foo *foo) {
        m_foo_shared_ptr = foo;
    }

    // return value
    Foo get_foo_value () {
        return m_foo_value;
    }

    // -#- @return(caller_owns_return=false) -#-
    const Foo * get_foo_shared_ptr () {
        return m_foo_shared_ptr;
    }

    // -#- @return(caller_owns_return=true) -#-
    Foo * get_foo_ptr () {
        Foo *foo = m_foo_ptr;
        m_foo_ptr = NULL;
        return foo;
    }

#if 0
    // -#- @return(caller_owns_return=true) -#-
    Zbr* get_zbr () {
        if (m_zbr)
        {
            m_zbr->Ref ();
            return m_zbr;
        } else
            return NULL;
    }

    // -#- @return(caller_owns_return=true) -#-
    Zbr* get_internal_zbr () {
        m_internal_zbr->Ref ();
        return m_internal_zbr;
    }

    // return reference counted object, caller does not own return
    // -#- @return(caller_owns_return=false) -#-
    Zbr* peek_zbr () { return m_zbr; }

    // pass reference counted object, transfer ownership
    // -#- @zbr(transfer_ownership=true) -#-
    void set_zbr_transfer (Zbr *zbr) {
        if (m_zbr)
            m_zbr->Unref ();
        m_zbr = zbr;
    }

    // pass reference counted object, does not transfer ownership
    // -#- @zbr(transfer_ownership=false) -#-
    void set_zbr_shared (Zbr *zbr) {
        if (m_zbr)
            m_zbr->Unref ();
        zbr->Ref ();
        m_zbr = zbr;
    }


    // return reference counted object, caller does not own return
    PointerHolder<Zbr> get_zbr_pholder () {
        PointerHolder<Zbr> foo = { m_zbr };
        m_zbr->Ref ();
        return foo;
    }

    // pass reference counted object, transfer ownership
    void set_zbr_pholder (PointerHolder<Zbr> zbr) {
        if (m_zbr)
            m_zbr->Unref ();
        m_zbr = zbr.thePointer;
        m_zbr->Ref ();
    }

    int get_int (const char *from_string);
    int get_int (double from_float);

    // custodian/ward tests

    // -#- @return(custodian=0, reference_existing_object=true) -#-
    Foobar* get_foobar_with_self_as_custodian () {
        if (m_foobar == NULL) {
            m_foobar = new Foobar;
        }
        return m_foobar;
    }
    // -#- @return(custodian=1, reference_existing_object=true); @other(transfer_ownership=false) -#-
    Foobar* get_foobar_with_other_as_custodian (SomeObject *other) {
        return other->get_foobar_with_self_as_custodian ();
    }
    // -#- @foobar(custodian=0, transfer_ownership=True) -#-
    void set_foobar_with_self_as_custodian (Foobar *foobar) {
        delete m_foobar;
        m_foobar = foobar;
    }

    virtual const char* method_returning_cstring() const { return "foobar"; }

protected:
    std::string protected_method_that_is_not_virtual (std::string arg) const;
#endif
};

//----------------------------------------------------------------------
