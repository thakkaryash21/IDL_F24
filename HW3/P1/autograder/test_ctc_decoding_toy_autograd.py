import numpy as np
import sys, os
import pickle
from test import Test
sys.path.append("./")
from CTC.CTCDecoding import BeamSearchDecoder

class BeamSearchToyTest(Test):
	def __init__(self):
		pass

	def test_beam_search_toy(self):

		# Initials
		ysizes = [(3, 3, 1)]
		symbol_sets = [["A", "B"]]
		beam_widths = [3]

		i = 0

		y_probs = np.array([[
			[0.49, 0.03, 0.47],
			[0.38, 0.44, 0.18],
			[0.02, 0.40, 0.58],
		]])
		y_probs = y_probs.T

		SymbolSets = symbol_sets[i]
		BeamWidth = beam_widths[i]

		decoder = BeamSearchDecoder(SymbolSets, BeamWidth)
		BestPath, MergedPathScores = decoder.decode(y_probs)

		try:
			assert BestPath == "A"
		except AssertionError:
			print(f"Incorrect Best Path\nExpected:A\nPredicted:\t{BestPath}")
			return False
		else:
			print(f"Correct Best Path\nExpected:A\nPredicted:\t{BestPath}")


		expected_MergedPathScores = {
			'A':   np.array([0.170575]),
			'AB':  np.array([0.132704]),
			'BAB': np.array([0.119944]),
			'B':   np.array([0.107996]),
			'BA':  np.array([0.086856]),
			'':    np.array([0.003724])
		}
		try:
			assert set(MergedPathScores.keys()) == set(expected_MergedPathScores.keys())
			for key in MergedPathScores.keys():
				assert np.allclose(np.array(MergedPathScores[key]), expected_MergedPathScores[key])
		except AssertionError:
			print(f"Incorrect Merged Path Scores\nExpected: \t{expected_MergedPathScores}\nPredicted: \t{MergedPathScores}")
			return False
		else:
			print(f"Correct Merged Path Scores\nExpected: \t{expected_MergedPathScores}\nPredicted: \t{MergedPathScores}")

		return True
