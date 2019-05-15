from TextSegmentator import TextSegmentator
import argparse

def main(args):
    txts = TextSegmentator(args.amount, args.level, args.embeddings, args.type)
    txts.process(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation of plain text into coherent paragraphs based on word embeddings.')
    parser.add_argument('input', type=str,
                    help='input file with plain text')
    parser.add_argument('amount', type=int,
                    help='Desired quantity of paragraphs')
    parser.add_argument('level', type=str,
                    help='level of splitting (words or sentences)')
    parser.add_argument('embeddings', type=str,
                    help='path to glove embeddings')
    parser.add_argument('type', type=str,
                    help='type of embeddings(currently is fasttext and glove only)')
    args = parser.parse_args()
    main(args)
