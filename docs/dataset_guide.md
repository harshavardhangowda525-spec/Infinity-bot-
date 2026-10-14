# Dataset guide (open-model fine-tuning)

## Folders

| Folder | Used for |
|---|---|
| `datasets/chat/*.jsonl` | Training, chat domain |
| `datasets/coding/*.jsonl` | Training, coding domain |
| `datasets/validation/*.jsonl` | Validation loss during training (never trained on) |
| `training/evaluation/test_cases.jsonl` | Final held-out test set for `evaluate.py` (never trained on, separate from validation) |
| generated at prepare time | Tool-use transcripts: expert bug fixes replayed through the real INFINITY runtime. Only runs judged `completed_verified` are kept |
| `training/data/` (generated, git-ignored) | `train.jsonl`, `val.jsonl`, `report.json`, `report.md` |

## Format (one JSON object per line)

```json
{"id": "chat-greet-1", "domain": "chat", "license": "Project-Original",
 "messages": [{"role": "user", "content": "hi"},
              {"role": "assistant", "content": "Hi! What can I help you with?"}]}
```

- **`domain`**: `chat`, `coding` or `tool`. **`license`** is required: record where the
  example may legally come from.
- **Roles:** `system` (first only), `user`, `assistant`, `tool`.
  - The last message must come from the assistant.
  - An assistant message must follow a user or tool message.
- **`loss_on`**: `all_assistant` (the default) trains on every assistant turn; `last_assistant` trains on the last one only.
- **System prompt:** chat and coding examples without one get the bot's real Normal-chat
  system prompt automatically, so training matches what the model sees in the app.
- **Chat template:** the base model's own template is used (`tokenizer.apply_chat_template`).
  - Tool observations become user turns prefixed with `[tool result]`. Inference does the same.
  - Loss is computed on assistant tokens only.

## Checks run by `prepare_data.py`

- **Rejected:** malformed JSON, missing or invalid fields, empty messages, corrupted text
  (U+FFFD or NUL), anything that looks like a secret, wrong role order. Each rejection is
  listed with its file and line.
- **Duplicates:** removed after whitespace and case normalisation.
- **Leaks:** a training example whose first user message also appears in validation or in
  the test set is **removed from training**.
- **Too long:** examples longer than `data.max_seq_len` are **dropped and counted**, never
  silently truncated.
- **Benchmark tasks:** the live-repair tasks used by `infinity bench` are excluded from tool validation data.

## The included examples

- The `seed.jsonl` files hold **10 chat, 6 coding and 4 validation examples**. They show
  the format. They are far too few to teach a model much.
- Most of the default training signal comes from the generated tool-use transcripts (120 tasks).
- A small dataset will **not** produce large gains. Measure with `evaluate.py` before and after.

## Adding data: rules

- Write your own examples, or use datasets whose licence allows training, and record the licence in `license`.
- **Do not paste replies from commercial AI services** (ChatGPT, Claude, Gemini and so on).
  Their terms generally forbid using outputs to train other models.
- Do not include private conversations, passwords, API keys or personal data.
  The secret filter catches common key formats, not everything.
- Never put test-set prompts into training. The leak check removes exact matches; paraphrases are your responsibility.
