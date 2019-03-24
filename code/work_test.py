from TextSegmntator import TextSegmentator

txts = TextSegmentator(5, 'sentences', "glove.6B.100d.txt")
txts.process("test1.txt")
