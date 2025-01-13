class Solution:

    def findAllConcatenatedWordsInADict(self, words):
        words = set(words)
        print(words)

        def canconcatenate(word, dpmemo={}):
            if word in dpmemo:
                return dpmemo[word]

            part1, part2 = False, False
            dpmemo[word] = False

            for j in range(1, len(word)):
                if word[:j] in words or canconcatenate(word[:j], dpmemo):
                    part1 = True
                if word[j:] in words or canconcatenate(word[j:], dpmemo):
                    part2 = True
                if part1 and part2:
                    dpmemo[word] = True
                    break
                else:
                    dpmemo[word] = False
            return dpmemo[word]

        result = []
        for word in words:
            if canconcatenate(word):
                result.append(word)
        print(result)

        # result = list(filter(concatenated, words))
        return result


if __name__ == '__main__':
    solu = Solution()
    result = solu.findAllConcatenatedWordsInADict(["dog", "cat", "dogcatdog", "catdog"])
    print(result)
