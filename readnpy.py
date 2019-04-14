import numpy as np
import sys

if __name__ == "__main__":
    entity_emb = np.load(sys.argv[1])
    print(entity_emb[0])
