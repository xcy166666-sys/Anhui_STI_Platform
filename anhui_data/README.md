# Anhui Project Data

The complete project dataset is intentionally excluded from this public repository because source records may contain project owners, contacts, and email addresses.

Place the controlled local dataset at:

```text
anhui_data/cleaned/project_vectors_source.jsonl
```

Each line must be a JSON object with this minimum structure:

```json
{
  "project_id": "project-unique-id",
  "content": "text used for retrieval",
  "metadata": {
    "project_name": "project name",
    "track": "industry track",
    "technology": "technology direction",
    "stage": "project stage",
    "summary": "project summary"
  }
}
```

Do not commit the restored dataset or `chat_state.json`.
