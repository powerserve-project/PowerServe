import argparse


try:
    import numpy as np

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

except ImportError:

    def dot_product(a, b):
        return sum(x * y for x, y in zip(a, b))

    def norm(a):
        return sum(x * x for x in a) ** 0.5

    def cos_sim(a, b):
        return dot_product(a, b) / (norm(a) * norm(b))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=str)
    parser.add_argument("--f2", type=str)
    args = parser.parse_args()

    file1 = args.f1
    file2 = args.f2

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    a = [float(line.replace("\n", "")) for line in lines1 if line.replace("\n", "").strip()]
    b = [float(line.replace("\n", "")) for line in lines2 if line.replace("\n", "").strip()]

    assert len(a) == len(b), "two file's length must be equal!"
    print(cos_sim(a, b))
