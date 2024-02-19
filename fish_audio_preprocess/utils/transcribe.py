from pathlib import Path

from tqdm import tqdm

PROMPT = {
    "zh": "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。",
    "en": "In the realm of advanced technology, the evolution of artificial intelligence stands as a monumental achievement.",
    "jp": "先進技術の領域において、人工知能の進化は画期的な成果として立っています。常に機械ができることの限界を押し広げているこのダイナミックな分野は、急速な成長と革新を見せています。複雑なデータパターンの解読から自動運転車の操縦まで、AIの応用は広範囲に及びます。",
}


def batch_transcribe(files: list[Path], model_size: str, lang: str, pos: int):
    import whisper

    model = whisper.load_model(model_size)
    results = {}
    for file in tqdm(files, position=pos):
        if lang in PROMPT:
            result = model.transcribe(file, language=lang, initial_prompt=PROMPT[lang])
        else:
            result = model.transcribe(file, language=lang)
        results[str(file)] = result["text"]
    return results
