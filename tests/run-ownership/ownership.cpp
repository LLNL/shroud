
#include "ownership.hpp"

//----------------------------------------------------------------------
// example lifted from pybindgen PyBindGen-0.18.0/tests/foo.cc

static Zbr *g_zbr = NULL;
int Zbr::instance_count = 0;

void store_zbr (Zbr *zbr)
{
    if (g_zbr)
        g_zbr->Unref ();
    // steal the reference
    g_zbr = zbr;
}

int invoke_zbr (int x)
{
    return g_zbr->get_int (x);
}

void delete_stored_zbr (void)
{
    if (g_zbr)
        g_zbr->Unref ();
    g_zbr = NULL;
}
//----------------------------------------------------------------------
