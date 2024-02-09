from preprocess import readFiles
from index import index

def main():
    path = './test.txt'
    pre = readFiles(path)
    print(pre)

if __name__ == "__main__":
    main()

