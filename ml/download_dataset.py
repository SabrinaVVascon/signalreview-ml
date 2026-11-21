"""Baixa o CSV original B2W-Reviews01 (~47 MB) da fonte publica.

Dataset: https://github.com/americanas-tech/b2w-reviews01
Licenca do dataset: CC BY-NC-SA 4.0 (uso nao comercial, atribuicao obrigatoria).
"""
import argparse
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"


def download(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    req = Request(SOURCE, headers={"User-Agent": "SignalReviewML/0.1"})
    print("Baixando dataset B2W (~47 MB). Fonte:", SOURCE)
    with urlopen(req, timeout=90) as response, output.open("wb") as file:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            file.write(block)
    print(f"Concluido: {output} ({output.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/B2W-Reviews01.csv"))
    args = parser.parse_args()
    download(args.output)
