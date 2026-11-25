# GintiEval

GintiEval is a benchmark for testing whether large language models can read, write and reason with **number words in Indian languages**. *Ginti* is the Hindi word for counting.

It has 6,307 items in Hindi, Bengali, Marathi, Gujarati, Tamil and Telugu (native script), with romanized Hindi and English as controls.

| Task | Items | What the model has to do |
|---|---|---|
| Words to Digits (W2D) | 2,691 | read a number written in words and give the digits |
| Digits to Words (D2W) | 1,794 | write a number in words |
| Arithmetic (Arith) | 900 | add or subtract two numbers given in words |
| Fraction Words (Frac) | 702 | resolve quarter and half words such as *sawa*, *dedh*, *dhai*, *paune*, *saadhe* |
| System Conversion (Conv) | 220 | move between lakh/crore and million/billion, and regroup digits |

## Layout

```
data/gintieval.jsonl      the benchmark (id, task, lang, stratum, prompt, answer, answer_type, meta)
src/lexicon.py            number-word verbalizers and the fraction-word lexicon
src/build.py              builds the dataset (fixed seed)
src/run.py                runs a model (local MLX model or any OpenAI-compatible endpoint)
src/score.py              answer extraction, strict and lenient D2W matching, error taxonomy
src/analyze.py            scores all runs and writes results/summary.json and the result tables
src/claude_sample.py      scores the 319-item sample answered by the proprietary models
results/<model>.jsonl     raw model outputs, one line per item
results/claude_sample/    the 319-item sample: questions, answers per model, scores
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuilding the data

```
cd src
python build.py
```

The gold answers are produced by rule-based verbalizers: AI4Bharat `indic-numtowords` is the primary reference, with `indic-num2words` and `num2words` as secondary references for accepted spellings. The romanized Hindi forms for 0 to 99 and the fraction-word phrases are hand-curated (see `src/lexicon.py`).

## Running a model

Local model (Apple silicon, MLX):

```
python run.py --backend mlx --model mlx-community/gemma-3-12b-it-qat-4bit --name gemma-3-12b
```

Any OpenAI-compatible endpoint:

```
export GINTI_API_KEY=...
python run.py --backend api --base-url https://openrouter.ai/api/v1 --model <model-id> --name <short-name>
```

Runs are resumable; outputs go to `results/<name>.jsonl`. Then:

```
python analyze.py
```

## Results

Accuracy (%) of six open-weight models (4-bit quantized, zero-shot, greedy decoding) on the full benchmark:

| Model | W2D | D2W | Arith | Frac | Conv | Avg |
|---|---|---|---|---|---|---|
| Llama 3.2 3B | 4.9 | 2.6 | 1.2 | 1.0 | 10.9 | 4.1 |
| Gemma 3 4B | 28.1 | 7.1 | 8.3 | 7.6 | 15.9 | 13.4 |
| Qwen2.5 7B | 8.5 | 2.3 | 0.8 | 4.6 | 39.5 | 11.2 |
| Llama 3.1 8B | 10.9 | 7.6 | 4.7 | 4.1 | 19.1 | 9.3 |
| Aya Expanse 8B | 8.1 | 2.7 | 1.5 | 5.2 | 13.2 | 6.1 |
| Gemma 3 12B | 52.3 | 18.7 | 19.8 | 25.7 | 40.9 | 31.5 |

W2D, D2W and Arith are averaged over the six Indian languages in native script; D2W uses the lenient metric.

Four proprietary models on a stratified 319-item sample (`results/claude_sample/`):

| Model | W2D | D2W | Arith | Frac | Conv | Avg |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 92.6 | 90.0 | 88.9 | 97.5 | 86.7 | 91.1 |
| Claude Sonnet 5.5 | 97.8 | 93.3 | 100.0 | 100.0 | 100.0 | 98.2 |
| Claude Opus 5.5 | 100.0 | 96.7 | 100.0 | 100.0 | 100.0 | 99.3 |
| Claude Fable 5.2 | 100.0 | 96.7 | 100.0 | 100.0 | 100.0 | 99.3 |

These were queried through the Claude Code agent interface (no code execution, reasoning allowed, no control over decoding), so the two tables are not a controlled comparison.

## Known limitations

- The reference forms have not been audited by native speakers of all six languages.
- Many number words have several accepted spellings, so D2W scores are lower bounds.
- The romanized Hindi forms follow one common spelling among several.

## Licence

Code: MIT. Data and model outputs: CC BY 4.0.
