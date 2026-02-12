from data_loader.file_loader import FileLoader
from data_loader.data_preprocessor import Preprocessor
from data_loader.data_tokenizer import ChunkingConfig, TokenChunker

def main():
    fullpath = '{{fullpath}}'  # Replace with your actual path  
    loader = FileLoader(fullpath)
    file_contents = loader.load_files()
    preprocessed_contents = Preprocessor().preprocess(file_contents)

    cfg = ChunkingConfig(
        tokenizer_name="bert-base-uncased",
        max_tokens=120,
        overlap_tokens=20,
        boundary_aware=True,
    )

    chunker = TokenChunker(cfg)
    all_chunks: list[str] = []
    for content in preprocessed_contents:
        chunks = chunker.chunk_text(content)
        for chunk in chunks:
            all_chunks.append(chunk.text)
   
    print(all_chunks)

if __name__ == "__main__":
    main()
