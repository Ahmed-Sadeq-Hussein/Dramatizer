# Load test trial

Against https://127.0.0.1:5005; each person renders every 30-60 s and asks every 120 s; 2 minutes per run.

| People | Renders/min | Render p50 / p95 (s) | Waiting p50 / p95 | Working p50 / p95 | 429 / 503 / timeouts / expired | Deepest queue | Chatbot p50 / p95 (s) | GPU memory max |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.5 | 9.4 / 14.61 | 0.5 / 1.0 | 8.9 / 13.7 | 0 / 0 / 0 / 0 | 1 | 0.03 / 0.04 | 9781.0 MB |
