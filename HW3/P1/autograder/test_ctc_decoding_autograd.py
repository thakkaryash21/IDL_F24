import numpy as np
import sys, os
import pickle
from test import Test
sys.path.append("./")
from CTC.CTCDecoding import GreedySearchDecoder, BeamSearchDecoder

# DO NOT CHANGE -->
isTesting = True
EPS = 1e-20

# -->


class SearchTest(Test):
    def __init__(self):
        pass

    def test_greedy_search(self):
        SEED = 11785
        np.random.seed(11785)
        y_rands = np.random.uniform(EPS, 1.0, (4, 10, 1))
        y_sum = np.sum(y_rands, axis=0)
        y_probs = y_rands / y_sum
        SymbolSets = ["a", "b", "c"]

        expected_results = np.load(
            os.path.join("autograder",  "data", "greedy_search.npy"),
            allow_pickle=True,
        )
        ref_best_path, ref_score = expected_results

        decoder = GreedySearchDecoder(SymbolSets)
        user_best_path, user_score = decoder.decode(y_probs)

        if isTesting:
            try:
                assert user_best_path == ref_best_path
            except Exception as e:
                print("Best path does not match")
                print("Your best path:    \t", user_best_path)
                print("Expected best path:\t", ref_best_path)
                return False

            try:
                assert user_score == float(ref_score)
            except Exception as e:
                print("Best Score does not match")
                print("Your score:    \t", user_score)
                print("Expected score:\t", ref_score)
                return False

        # Use to save test data for next semester
        if not isTesting:
            results = [user_best_path, user_score]
            np.save(os.path.join('autograder', 
                             'data', 'greedy_search.npy'), results, allow_pickle=True)

        return True

    def test_beam_search_i(self, SEED, y_size, syms, bw, BestPath_ref, MergedPathScores_ref):
        np.random.seed(SEED)
        y_rands = np.random.uniform(EPS, 1.0, y_size)
        y_sum = np.sum(y_rands, axis=0)
        y_probs = y_rands / y_sum

        SymbolSets = syms
        BeamWidth = bw

        decoder = BeamSearchDecoder(SymbolSets, BeamWidth)
        BestPath, MergedPathScores = decoder.decode(y_probs)

        if isTesting:
            try:
                assert BestPath == BestPath_ref
            except Exception as e:
                print("BestPath does not match!")
                print("Your best path:    \t", BestPath)
                print("Expected best path:\t", BestPath_ref)
                return False

            try:
                assert len(MergedPathScores.keys()) == len(MergedPathScores)
            except Exception as e:
                print("Total number of merged paths returned does not match")
                print(
                    "Number of merged path score keys: ",
                    "len(MergedPathScores.keys()) = ",
                    len(MergedPathScores.keys()),
                )
                print(
                    "Number of merged path scores:",
                    "len(MergedPathScores)= ",
                    len(MergedPathScores),
                )
                return False

            no_path = False
            values_close = True

            for key in MergedPathScores_ref.keys():
                if key not in MergedPathScores.keys():
                    no_path = True
                    print("%s path not found in reference dictionary" % (key))
                    return False
                else:
                    if not self.assertions(
                        MergedPathScores_ref[key],
                        MergedPathScores[key],
                        "closeness",
                        "beam search",
                    ):
                        values_close = False
                        print("score for %s path not close to reference score" % (key))
                        return False
            return True
        else:
            return BestPath, MergedPathScores

    def test_beam_search(self):
        expected_results = np.load(
            os.path.join("autograder", "data", "beam_search.npy"),
            allow_pickle=True,
        )

        # Initials
        ysizes = [(4, 10, 1), (5, 20, 1), (6, 20, 1)]
        symbol_sets = [["a", "b", "c"], ["a", "b", "c", "d"], ["a", "b", "c", "d", "e"]]
        beam_widths = [2, 3, 3]

        n = 3
        results = []
        for i in range(n):
            BestPathRef, MergedPathScoresRef = expected_results[i]
            y_size, syms, bw = ysizes[i], symbol_sets[i], beam_widths[i]
            result = self.test_beam_search_i(
                i, y_size, syms, bw, BestPathRef, MergedPathScoresRef
            )
            if isTesting:
                if result != True:
                    print("Failed Beam Search Test: %d / %d" % (i + 1, n))
                    return False
                else:
                    print("Passed Beam Search Test: %d / %d" % (i + 1, n))
            else:
                results.append(result)

        # Use to save test data for next semester
        if not isTesting:
            np.save(os.path.join('autograder',
                             'data', 'beam_search.npy'), results, allow_pickle=True)
        return True

    def run_test(self):
        # Test Greedy Search
        self.print_name("Section 5.1 - Greedy Search")
        greedy_outcome = self.test_greedy_search()
        self.print_outcome("Greedy Search", greedy_outcome)
        if greedy_outcome == False:
            self.print_failure("Greedy Search")
            return False

        # Test Beam Search
        self.print_name("Section 5.2 - Beam Search")
        beam_outcome = self.test_beam_search()
        self.print_outcome("Beam Search", beam_outcome)
        if beam_outcome == False:
            self.print_failure("Beam Search")
            return False

        return True
