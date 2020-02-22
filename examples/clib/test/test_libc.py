import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

import libc

class String(unittest.TestCase):
    def Xtest_strchr(self):
        self.assertEqual(libc.strchr("abcde", 65), 5)

    def test_strlen(self):
        self.assertEqual(libc.strlen("abcde"), 5)
