from preprocess import readFiles
from index import index
from randr import rank

def main():
    path = './coll'
    pre = readFiles(path)
    next = index(pre)
    print(next)


if __name__ == "__main__":
    main()

