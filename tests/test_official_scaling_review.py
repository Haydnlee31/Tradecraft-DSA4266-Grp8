"""Report all observations; sample SD uses n-1, not population variance."""
import unittest
from src.eval.official_scaling_review import describe


class ScalingReviewTests(unittest.TestCase):
    def test_summary_retains_all_values_and_sample_sd(self):
        result = describe([1., 2., 3.])
        self.assertEqual(result, {'values': [1., 2., 3.], 'mean': 2.,
                                  'sample_sd': 1., 'min': 1., 'max': 3.})


if __name__ == '__main__':
    unittest.main()
