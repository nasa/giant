"""
test_correlators
================

Tests the methods and classes contained in the correlators submodule of GIANT.
"""

from unittest import TestCase
import numpy as np
from giant.image_processing import correlators


class TestCv2Correlator(TestCase):
    def test_cv2correlator(self):
        # TODO: test a couple other coefficients as well
        img = np.random.randn(30, 30)
        temp = img[20:27, 15:27]

        cor_surf = correlators.cv2_correlator_2d(img, temp)

        temp_middle = np.floor(np.array(temp.shape) / 2)

        temp_point = np.array([0, 0])  # look for the upper left corner

        img_loc = np.unravel_index(cor_surf.argmax(), cor_surf.shape) - temp_middle + temp_point

        np.testing.assert_array_equal([20, 15], img_loc)

        self.assertAlmostEqual(cor_surf.max(), 1, places=4)


class TestScipyCorrelator(TestCase):
    def test_scipy_correlator(self):
        # TODO: test a couple other coefficients as well
        img = np.random.randn(30, 30)
        temp = img[20:27, 15:27]

        cor_surf = correlators.scipy_correlator_2d(img, temp)

        temp_middle = np.floor(np.array(temp.shape) / 2)

        temp_point = np.array([0, 0])  # look for the upper left corner

        img_loc = np.unravel_index(cor_surf.argmax(), cor_surf.shape) - temp_middle + temp_point

        np.testing.assert_array_equal([20, 15], img_loc)

        self.assertAlmostEqual(cor_surf.max(), 1, places=4)
        
class TestFftCorrelator1d(TestCase):
    def test_fft_correlator_1d(self):
        # Create a random 1D signal (like a scan line)
        signal_length = 100
        line = np.random.randn(signal_length)
        
        # Extract a template from a known location in the line
        template_start = 30
        template_end = 51
        template = line[template_start:template_end]
        
        # Prepare input arrays - each should be 2D with lines as rows
        extracted_lines = line.reshape(1, -1)  # Single line as a row
        predicted_lines = template.reshape(1, -1)  # Single template as a row
        
        # Run the FFT correlator
        cor_result = correlators.fft_correlator_1d(extracted_lines, predicted_lines)
        
        # Find the location of maximum correlation
        max_corr_idx = np.argmax(cor_result)
        
        # Calculate expected location
        # The correlation result should have the template centered at the original location
        template_center = len(template) // 2
        expected_location = template_start + template_center
        
        # Verify the detected location matches the expected location
        self.assertEqual(max_corr_idx, expected_location)
        
        # Verify that the maximum correlation is close to 1 (perfect match)
        self.assertAlmostEqual(cor_result.max(), 1.0, places=4)
    
    def test_fft_correlator_1d_multiple_lines(self):
        # Test with multiple lines
        num_lines = 5
        signal_length = 80
        
        # Create multiple random lines
        lines = np.random.randn(num_lines, signal_length)
        
        # Extract templates from known locations in each line
        template_starts = [10, 15, 20, 25, 30]
        template_length = 15
        templates = np.zeros((num_lines, template_length))
        
        for i, start in enumerate(template_starts):
            templates[i] = lines[i, start:start + template_length]
        
        # Run the FFT correlator
        cor_results = correlators.fft_correlator_1d(lines, templates)
        
        # Verify results for each line
        for i, start in enumerate(template_starts):
            max_corr_idx = np.argmax(cor_results[i])
            
            # Calculate expected location
            template_center = template_length // 2
            expected_location = start + template_center
            
            # Verify the detected location matches the expected location
            self.assertEqual(max_corr_idx, expected_location)
            
            # Verify high correlation coefficient
            self.assertGreater(cor_results[i].max(), 0.9)
    
    def test_fft_correlator_1d_zero_template(self):
        # Test edge case with zero template
        line = np.random.randn(50)
        zero_template = np.zeros(10)
        
        extracted_lines = line.reshape(1, -1)
        predicted_lines = zero_template.reshape(1, -1)
        
        # Run the correlator
        cor_result = correlators.fft_correlator_1d(extracted_lines, predicted_lines)
        
        # All correlation values should be zero or very small
        self.assertTrue(np.all(np.abs(cor_result) < 1e-10))

       
        
if __name__ == '__main__':
    import unittest
    unittest.main()
