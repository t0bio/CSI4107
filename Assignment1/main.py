from preprocess import readFiles
from index import index

def main():
    path = './coll'
    pre = readFiles(path)
    index(pre)

if __name__ == "__main__":
    main()

