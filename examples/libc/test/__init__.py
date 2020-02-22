import unittest
import test

def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test)
    return suite
