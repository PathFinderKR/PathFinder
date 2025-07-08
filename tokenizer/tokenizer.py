import pandas as pd
from konlpy.tag import Mecab
from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from src.config import TokenizerConfig, DatasetConfig


def main():
    # Configurations
    tokenizer_config = TokenizerConfig()
    dataset_config = DatasetConfig()

    # Load training data
    train_df = pd.read_csv(dataset_config.train_path, encoding=dataset_config.encoding)
    train_df = train_df[train_df["generated"] == 0].copy()

    # 형태소 분석기
    mecab = Mecab()
    def mecab_texts(texts):
        for text in texts:
            tokens = mecab.morphs(text)
            yield " ".join(tokens)

    # Tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token=tokenizer_config.special_tokens["unk_token"]))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Train
    trainer = trainers.WordPieceTrainer(
        vocab_size=tokenizer_config.vocab_size,
        special_tokens=list(tokenizer_config.special_tokens.values())
    )
    tokenizer.train_from_iterator(
        mecab_texts(train_df["full_text"].tolist()),
        trainer=trainer
    )
    tokenizer.save(tokenizer_config.tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_config.tokenizer_path}")

    # Load and test the tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_config.tokenizer_path)
    tokenizer.add_special_tokens(tokenizer_config.special_tokens)
    print(tokenizer)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(tokenizer.tokenize("수난곡(受難曲)은 배우의 연기 없이 무대에 올려지는 성악을 주로 한 종합 예술이다. "
                             "이러한 의미에서 오라토리오와 유사하지만, 성경의 사복음서를 기반으로 한 예수 그리스도의 생애를 주로 다루고 있다는 점에서 차이가 있다. "
                             "또한 이는 주로 독일 계열 작곡가들에게 쓰인 개념이다. "
                             "수난 또는 수난곡을 뜻하는 영어 'Passion'은 2세기에 나타난 라틴어 'passio'에서 유래하며, 예수의 생애와 고난이란 의미를 담고 있다."))

if __name__ == "__main__":
    main()