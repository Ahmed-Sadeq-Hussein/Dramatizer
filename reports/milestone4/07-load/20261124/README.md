# Load test 20261124

Against https://127.0.0.1:5005; each person renders every 30-60 s and asks every 120 s; 10 minutes per run.

| People | Renders/min | Render p50 / p95 (s) | Waiting p50 / p95 | Working p50 / p95 | 429 / 503 / timeouts / expired | Deepest queue | Chatbot p50 / p95 (s) | GPU memory max |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.3 | 9.42 / 18.1 | 0.52 / 0.9 | 9.3 / 18.0 | 0 / 0 / 0 / 0 | 1 | 26.89 / 50.0 | 9796.0 MB |
| 5 | 4.1 | 84.32 / 95.09 | 65.94 / 78.79 | 17.3 / 22.3 | 0 / 0 / 0 / 0 | 4 | 90.09 / 90.15 | 9669.0 MB |
| 20 | 4.6 | 205.53 / 248.69 | 188.63 / 231.56 | 17.35 / 23.68 | 0 / 107 / 0 / 0 | 11 | 90.1 / 110.16 | 9599.0 MB |

**Capacity:** the site serves **2 people rendering at once** with renders done in about 18 s (p95). Little's law with 30 % headroom, from the most the GPU finished when it was saturated: 4.6 renders a minute (20 people), one every 13.0 s. A render alone takes 9.3 s of GPU work; with people rendering and asking at once it took 17.35 s (p50), because CosyVoice shares the GPU with the chatbot. A queue of 2 holds what 2 people have waiting at once.
