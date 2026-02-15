from data_loader.file_loader import FileLoader
from data_loader.data_preprocessor import Preprocessor
from data_loader.data_tokenizer import ChunkingConfig, TokenChunker
from model.chat import Chat, ChatConfig
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
        epochs=10,
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        output_dir="./output",
    )

    trainer: Training = Training(all_chunks, config=training_config)
    trainer.train()
    trainer.evaluate()
    saved_path = trainer.save()

    # ---- Interactive chat ---------------------------------------------------
    chat: Chat = Chat(
        model_dir=str(saved_path),
        config=ChatConfig(
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
        ),
    )
    chat.start()

if __name__ == "__main__":
    main()
