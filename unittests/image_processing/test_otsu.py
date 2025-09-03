"""
test_otsu
=========

Tests the methods and classes contained in the otsu submodule of GIANT.
"""

from unittest import TestCase
import numpy as np
from giant.image_processing import otsu
import cv2


class TestOtsu(TestCase):

    def test_otsu(self):

        image = np.zeros((200, 200), dtype=np.uint8)

        s1 = image.copy().astype(bool)
        s2 = image.copy().astype(bool)
        s3 = image.copy().astype(bool)

        s1[100:150, 100:150] = True
        image[s1] = 50

        s2[50:100, 50:100] = True
        image[s2] = 75

        s3[10:60, 100:150] = True
        image[s3] = 200

        image = cv2.GaussianBlur(image, (5, 5), 1)

        thresh, labeled = otsu(image, 4)

        self.assertLess(thresh[0], 50)
        self.assertGreater(thresh[1], 50)
        self.assertLess(thresh[1], 75)
        self.assertGreater(thresh[2], 75)
        self.assertGreater((labeled[s1] == 1).sum(), s1.sum()*3/4)
        self.assertGreater((labeled[s2] == 2).sum(), s2.sum()*3/4)
        self.assertGreater((labeled[s3] == 3).sum(), s3.sum()*3/4)
        
        

if __name__ == '__main__':
    import unittest
    unittest.main()
