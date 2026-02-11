from data_loader.file_loader import FileLoader
from data_loader.data_preprocessor import Preprocessor

def main():
    fullpath = '/Users/dz/Documents/MyDataForPLM'
    loader = FileLoader(fullpath)
    file_contents = loader.load_files()
    preprocessed_contents = Preprocessor().preprocess(file_contents)

    print(preprocessed_contents)

if __name__ == "__main__":
    main()
