import numpy as np
import traceback
import json


# Test object to be used for other homeworks
class Test(object):
    def __init__(self):
        self.scores = {}

    def assertions(self, user_vals, expected_vals, test_type, test_name):
        if test_type == "type":
            try:
                assert type(user_vals) == type(expected_vals)
            except Exception as e:
                print("Type error, your type doesnt match the expected type.")
                print("Wrong type for %s" % test_name)
                print("Your type:   ", type(user_vals))
                print("Expected type:", type(expected_vals))
                return False
        elif test_type == "shape":
            try:
                assert user_vals.shape == expected_vals.shape
            except Exception as e:
                print("Shape error, your shapes doesnt match the expected shape.")
                print("Wrong shape for %s" % test_name)
                print("Your shape:    ", user_vals.shape)
                print("Expected shape:", expected_vals.shape)
                return False
        elif test_type == "closeness":
            try:
                assert np.allclose(user_vals, expected_vals, atol=1e-5)
            except Exception as e:
                print("Closeness error, your values dont match the expected values.")
                print("Wrong values for %s" % test_name)
                print("Your values:    ", user_vals)
                print("Expected values:", expected_vals)
                return False
        return True

    def print_failure(self, cur_test):
        print("*" * 50)
        print("The local autograder failed %s." % cur_test)
        print("*" * 50)
        print(" ")

    def print_name(self, cur_question):
        print("-" * 20)
        print(cur_question)

    def print_outcome(self, short, outcome):
        print(short + ": ", "PASS" if outcome else "*** FAIL ***")
        print("-" * 20)
        print()

    def get_test_scores(self):
        return sum(self.scores.values())

    def run_tests(self, section_title, tests, test_score):
        test_name = section_title.split(" - ")[1]
        any_passed = False  # Track if any test passes

        # Try running each test and collect outcomes
        for test in tests:
            try:
                self.print_name(section_title)
                test_outcome = test()
                self.print_outcome(test_name, test_outcome)
            except Exception:
                traceback.print_exc()
                test_outcome = False

            # Check if the current test passed
            if test_outcome == True:
                any_passed = True
                self.scores[test_name] = test_score
                break
            else:
                # Print failure message for the individual test
                self.print_failure(test_name)

        # After running all tests, check if none passed
        if not any_passed:
            self.scores[test_name] = 0
        return True

    def run_tests_toy(self, section_title, test, test_score):
        test_name = section_title.split(' - ')[1]
        try:
            self.print_name(section_title)
            test_outcome = test()
            self.print_outcome(test_name, test_outcome)
        except Exception:
            traceback.print_exc()
            test_outcome = False

        if test_outcome != True:
            self.print_failure(test_name)
            if test_outcome == False:
                self.scores[test_name] = 0
            else:
                self.scores[test_name] = test_outcome[1]
            return False
        self.scores[test_name] = test_score
        return True
