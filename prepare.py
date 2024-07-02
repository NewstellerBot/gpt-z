import numpy as np
import os

train_size = 0.9


if __name__ == "__main__":

    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    with open(input_file_path, "r") as f:
        data = f.read()

    vocab = sorted(list(set(data)))
    vocab_size = len(vocab)

    # Simple char level tokenizer
    ch2ix = {ch: ix for ix, ch in enumerate(vocab)}
    ix2ch = {ix: ch for ix, ch in enumerate(vocab)}

    encode = lambda s: [ch2ix[ch] for ch in s]
    decode = lambda t: "".join([ix2ch[ix] for ix in t])

    data = np.array(encode(data))

    n = len(data)
    div = int(n * train_size)

    train = data[:div]
    val = data[div:]

    train.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
