from data_loader.file_loader import FileLoader
from data_loader.data_preprocessor import Preprocessor
from data_loader.data_tokenizer import ChunkingConfig, TokenChunker
from model.training import Training, TrainingConfig


def main() -> None:
    fullpath: str = "/Users/dz/Documents/MyDataForPLM"
    loader: FileLoader = FileLoader(fullpath)
    file_contents: dict[str, str] = loader.load_files()
    preprocessed_contents: list[str] = Preprocessor().preprocess(file_contents)

    cfg: ChunkingConfig = ChunkingConfig(
        tokenizer_name="bert-base-uncased",
        max_tokens=120,
        overlap_tokens=20,
        boundary_aware=True,
    )

    chunker: TokenChunker = TokenChunker(cfg)
    all_chunks: list[str] = []
    for content in preprocessed_contents:
        chunks = chunker.chunk_text(content)
        for chunk in chunks:
            all_chunks.append(chunk.text)

    # ---- Train the model ---------------------------------------------------
    training_config: TrainingConfig = TrainingConfig(
        tokenizer_name="bert-base-uncased",
        max_seq_length=120,      # match chunking max_tokens
        n_positions=128,         # must be >= max_seq_length
        epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        output_dir="./output",
    )

    trainer: Training = Training(all_chunks, config=training_config)
    trainer.train()
    trainer.evaluate()
    trainer.save()


if __name__ == "__main__":
    main()
