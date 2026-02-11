from data_loader.file_loader import FileLoader

def main():
    fullpath = '/Users/dz/Documents/MyDataForPLM'
    loader = FileLoader(fullpath)
    file_contents = loader.load_files()
    print(file_contents)

if __name__ == "__main__":
    main()
