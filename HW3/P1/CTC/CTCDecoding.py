import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        for i in range(y_probs.shape[1]):
            path_prob *= np.max(y_probs[:, i, 0])
            index = np.argmax(y_probs[:, i, 0])
            if index != 0:
                if blank:
                    decoded_path.append(self.symbol_set[index - 1])
                    blank = 0
                else:
                    if (
                        len(decoded_path) == 0
                        or decoded_path[-1] != self.symbol_set[index - 1]
                    ):
                        decoded_path.append(self.symbol_set[index - 1])
            else:
                blank = 1

        decoded_path = "".join(decoded_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def initializePaths(self, y_probs):
        initialBlankPathScore, initialPathScore = {}, {}
        path = ""

        initialBlankPathScore[path] = y_probs[0]
        initialPathsWithFinalBlank = {path}

        initialPathsWithFinalSymbol = set()

        for i in range(len(self.symbol_set)):
            path = self.symbol_set[i]
            initialPathScore[path] = y_probs[i + 1]
            initialPathsWithFinalSymbol.add(path)

        return (
            initialPathsWithFinalBlank,
            initialPathsWithFinalSymbol,
            initialBlankPathScore,
            initialPathScore,
        )

    def prune(
        self, pathsWithTerminalBlank, pathsWithTerminalSymbol, blankPathScore, pathScore
    ):
        prunedBlankPathScore, prunedPathScore = {}, {}
        prunedPathsWithTerminalBlank, prunedPathsWithTerminalSymbol = set(), set()
        scoreList = []

        for path in pathsWithTerminalBlank:
            scoreList.append(blankPathScore[path])
        for path in pathsWithTerminalSymbol:
            scoreList.append(pathScore[path])

        scoreList.sort(reverse=True)
        cutoff = (
            scoreList[self.beam_width]
            if (self.beam_width < len(scoreList))
            else scoreList[-1]
        )

        for path in pathsWithTerminalBlank:
            if blankPathScore[path] > cutoff:
                prunedPathsWithTerminalBlank.add(path)
                prunedBlankPathScore[path] = blankPathScore[path]

        for path in pathsWithTerminalSymbol:
            if pathScore[path] > cutoff:
                prunedPathsWithTerminalSymbol.add(path)
                prunedPathScore[path] = pathScore[path]

        return (
            prunedPathsWithTerminalBlank,
            prunedPathsWithTerminalSymbol,
            prunedBlankPathScore,
            prunedPathScore,
        )

    def extend_with_blank(
        self,
        pathsWithTerminalBlank,
        pathsWithTerminalSymbols,
        y_probs,
        blankPathScore,
        pathScore,
    ):
        updatedPathsWithTerminalBlank = set()
        updatedBlankPathScore = {}

        for path in pathsWithTerminalBlank:
            updatedPathsWithTerminalBlank.add(path)
            updatedBlankPathScore[path] = blankPathScore[path] * y_probs[0]

        for path in pathsWithTerminalSymbols:
            if path in updatedPathsWithTerminalBlank:
                updatedBlankPathScore[path] += pathScore[path] * y_probs[0]
            else:
                updatedPathsWithTerminalBlank.add(path)
                updatedBlankPathScore[path] = pathScore[path] * y_probs[0]

        return updatedPathsWithTerminalBlank, updatedBlankPathScore

    def extend_with_symbol(
        self,
        pathsWithTerminalBlank,
        pathsWithTerminalSymbol,
        y_probs,
        blankPathScore,
        pathScore,
    ):
        updatedPathsWithTerminalSymbol = set()
        updatedPathScore = {}

        for path in pathsWithTerminalBlank:
            for i in range(len(self.symbol_set)):
                newPath = path + self.symbol_set[i]
                updatedPathsWithTerminalSymbol.add(newPath)
                updatedPathScore[newPath] = blankPathScore[path] * y_probs[i + 1]

        for path in pathsWithTerminalSymbol:
            for i in range(len(self.symbol_set)):
                newPath = (
                    path
                    if (self.symbol_set[i] == path[-1])
                    else path + self.symbol_set[i]
                )
                if newPath in updatedPathsWithTerminalSymbol:
                    updatedPathScore[newPath] = (
                        updatedPathScore[newPath] + pathScore[path] * y_probs[i + 1]
                    )
                else:
                    updatedPathsWithTerminalSymbol.add(newPath)
                    updatedPathScore[newPath] = pathScore[path] * y_probs[i + 1]

        return updatedPathsWithTerminalSymbol, updatedPathScore

    def mergeIdenticalPaths(
        self, pathsWithTerminalBlank, pathsWithTerminalSymbol, blankPathScore, pathScore
    ):
        mergedPaths = pathsWithTerminalSymbol
        finalPathScore = pathScore

        for p in pathsWithTerminalBlank:
            if p in mergedPaths:
                finalPathScore[p] += blankPathScore[p]
            else:
                mergedPaths.add(p)
                finalPathScore[p] = blankPathScore[p]

        return mergedPaths, finalPathScore

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        bestPath = None
        pathScore = {}
        blankPathScore = {}

        num_symbols, seq_len, batch_size = y_probs.shape

        (
            newPathsWithTerminalBlank,
            newPathsWithTerminalSymbol,
            newBlankPathScore,
            newPathScore,
        ) = self.initializePaths(y_probs[:, 0, :])

        for t in range(1, seq_len):
            (
                pathsWithTerminalBlank,
                pathsWithTerminalSymbol,
                blankPathScore,
                pathScore,
            ) = self.prune(
                newPathsWithTerminalBlank,
                newPathsWithTerminalSymbol,
                newBlankPathScore,
                newPathScore,
            )

            newPathsWithTerminalBlank, newBlankPathScore = self.extend_with_blank(
                pathsWithTerminalBlank,
                pathsWithTerminalSymbol,
                y_probs[:, t, :],
                blankPathScore,
                pathScore,
            )

            newPathsWithTerminalSymbol, newPathScore = self.extend_with_symbol(
                pathsWithTerminalBlank,
                pathsWithTerminalSymbol,
                y_probs[:, t, :],
                blankPathScore,
                pathScore,
            )

        mergedPaths, mergedPathScores = self.mergeIdenticalPaths(
            newPathsWithTerminalBlank,
            newPathsWithTerminalSymbol,
            newBlankPathScore,
            newPathScore,
        )

        bestPath = max(mergedPathScores, key=mergedPathScores.get)

        return bestPath, mergedPathScores
