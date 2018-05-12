
#include <string>



//----------------------------------------------------------------------
// example lifted from pybindgen PyBindGen-0.18.0/tests/foo.h

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
//----------------------------------------------------------------------
