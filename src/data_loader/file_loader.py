from pathlib import Path

class FileLoader:
    def __init__(self, fullpath: str) -> None:
        self.fullpath = fullpath

    def load_files(self) -> dict[str, str]:
        path = Path(self.fullpath)
        files = [file for file in path.iterdir() if file.is_file() and not file.name.startswith('.')]
        
        file_contents: dict[str, str] = {}
        for file in files:
            with open(file, 'r') as f:
                file_contents[file.name] = f.read()

        return file_contents