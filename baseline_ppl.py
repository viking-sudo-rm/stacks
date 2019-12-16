import numpy as np

from src.data.dyck import DyckReader

_OPENING = "(["

reader = DyckReader()
path = "/home/willm/data/dyck2/valid.txt"

word_ppls = []
for line in reader.read(path):
    ppls = []
    depth = 0
    for word in line["source"]:
        if word.text in _OPENING:
            depth += 1
        else:
            depth -= 1

        # If nothing is on the stack after this word, then we only have 2 options.
        ppl = 2. if depth == 0 else 3.
        ppls.append(ppl)

    # Update the mean perplexity.
    word_ppl = np.mean(ppls)
    word_ppls.append(word_ppl)

print("Mean PPL/word", np.mean(word_ppls))