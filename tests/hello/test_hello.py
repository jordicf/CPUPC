# (c) Marçal Comajoan Cara 2025
# For the CPUPC Project.
# Licensed under the MIT License (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import unittest

from tools.hello.hello import hello


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello(), "Hello!")
        self.assertEqual(hello("Marçal"), "Hello Marçal!")


if __name__ == "__main__":
    unittest.main()
