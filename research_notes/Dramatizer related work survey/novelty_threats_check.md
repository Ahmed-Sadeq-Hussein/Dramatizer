# Novelty threats to Dramatizer: primary-source check (as of 2026-10-27)

Dramatizer's core claims used for verdicts: (C1) search over the *input markup / text prompt* of a frozen zero-shot voice-cloning TTS; (C2) evolutionary / quality-diversity (MAP-Elites) search; (C3) automatic scoring (guards x Wundt novelty) plus a human pick from a diverse shortlist; (C4) cross-voice/cross-line transfer of found deliveries through relative-position genes and nearest-speaker seeding.

## Q1. arXiv 2609.24163 "Listen, Critique, and Refine"

### Takeaway
This is a trained LALM that drafts speech, critiques it in text and resynthesizes once (two hops), trained with RL. There is no population, no markup search, no human pick and no cross-voice transfer. **Threat to novelty: no.** At most it is a related "iterative refinement at inference" baseline.

### Cited Findings
- Title "Listen, Critique, and Refine: RL-Based Self-Refinement for Instruction-Following Speech Synthesis". Authors: Chee-En Yu, Yi-Cheng Lin, Sung-Feng Huang, Yun-Shao Tsai, Ho-Lam Chung, Xuanjun Chen, Hung-yi Lee. arXiv v1 dated 21 Sep 2026, eess.AS / cs.SD. No venue is listed. — [arXiv abs](https://arxiv.org/abs/2609.24163)
- Mechanism (paraphrased from the abstract): a single LALM produces draft speech as audio tokens, writes a text critique of its own acoustic realization, then produces refined speech conditioned on the draft and the critique. It is trained with RL and reports +7.15% relative on InstructTTSEval. — [arXiv abs](https://arxiv.org/abs/2609.24163)
- The abstract does not state the RL reward. — [arXiv abs](https://arxiv.org/abs/2609.24163)

### Inferences
- What is iterated is the model's own audio tokens and a free-text critique. Nothing is iterated over input markup, and the "search" is learned rather than evolutionary. The differences from C1 to C4 are clear.

### Gaps
- I did not open the full HTML or PDF, so the reward definition is unverified.

## Q2. arXiv 2608.28970 LoopTTS, "Diagnose, Then Refine"

### Takeaway
An AudioLLM judge diagnoses prosodic defects (stress, pauses, flat intonation) and writes structured instructions. A trained instruction-following Refiner TTS then resynthesizes from the initial audio. It is a closed-loop correction method, not a search. **Threat to novelty: partial, in framing only.** It is the closest "automatic critic proposes stress/pause edits" work, but it uses a trained refiner, has no population, no QD, no human shortlist and no cross-voice transfer.

### Cited Findings
- "Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction" (LoopTTS). Authors: Zeyang Song, Tianchi Liu, Tianrui Wang, Chenglin Xu, Steven Y. Guo, Haizhou Li. arXiv v1 dated 29 Aug 2026. Comment: "Accepted by EMNLP 2026 main conference". — [arXiv abs](https://arxiv.org/abs/2608.28970)
- Pipeline (paraphrased): base TTS, then an AudioLLM Judge that diagnoses salient prosodic problems and emits structured refinement instructions, then a Refiner (an instruction-following TTS) that re-synthesizes conditioned on the initial audio, the target text and the instruction. The Refiner is trained on Refiner-DB (42K examples with word-level prosodic weak supervision). Evaluation is by humans, against open-loop regeneration. — [arXiv abs](https://arxiv.org/abs/2608.28970)

### Inferences
- The edits are word-level stress and pause instructions, which overlap with the *kind* of edits in Dramatizer's genome. But they come from one judge in a single diagnose-and-correct pass, not from search, and they are aimed at fixing defects rather than exploring diverse deliveries. Dramatizer should cite it as the closest "critic-guided prosody editing" work and contrast single-pass correction with population-based exploration.

### Gaps
- The abstract does not say whether the loop repeats for multiple rounds. I did not read the full paper.

## Q3. LAION "MOSS-TTS emotion evolution"

### Takeaway
This is the strongest prior art. It is a genetic algorithm over TTS *prompt* content (voice description, script delivery cue, and in the edge-case variant **inline vocal-burst tags**) plus LoRA strength and sampling parameters, with automatic EmoNet fitness and a WER guard. It is published as LAION project web pages with no date, no authors, no paper and no peer review. **Threat to novelty: partial, and substantial for C1 and C2.** It already does "evolutionary search over TTS prompt and inline paralinguistic tags with automatic emotion fitness and a WER guard". It does **not** do QD/MAP-Elites (it uses a plain elitist GA maximising a single target), a Wundt novelty term, a human shortlist pick, relative-position genes, or cross-voice transfer. It also modifies the model (LoRA λ), not only the input.

### Cited Findings
- Page 1, "Evolving maximum emotional expression — all 40 LoRAs". Genome: "the GENERAL voice description, the SCRIPT delivery cue, the carrier sentence, the LoRA strength λ" plus audio temperature, top_p and top_k (λ seeded in 0.75–1.5, mutated within 0.4–2.0). Fitness is "the mean target-emotion strength over those 8 candidates", where each candidate scores EmoNet(target) − 0.25·max(0, WER−0.6). Population 16, 10 generations, elitism keeps the top 6. Offspring are 60% crossover of two elites plus mutation and 40% mutation of one elite. Mutations: regenerate description 35%, swap carrier 20%, λ jitter 23%, temperature 12%, top_k 10%. Base model laion/moss-tts-local-transformer-4.55b-voice-acting-v2 with TTS-AGI/moss-emotion-loras-v3 (rank 32). Results: LoRA adds about +0.295 target-emotion strength on average over the best prompt alone. Highest is Emotional Numbness at 0.60; hardest are Jealousy/Envy and Pride at 0.09. No date, authors, GitHub or paper link, and no peer-review statement on the page. — [LAION project page](https://projects.laion.ai/moss-emotion-evolution-all/)
- Page 2, "Evolving extreme genuine vocalizations with emotion LoRAs". Genome: "the GENERAL voice description, the SCRIPT cue, the inline vocal-burst tags" (e.g. `<screams>`, `<laughs softly>`, `<sobbing>`) plus carrier line, λ and sampling. Fitness is "a multi-objective score": EmoNet target + vocal-burst blend + genuineness + DSP proxies (RMS, median pitch, jitter, spectral flatness, 3–9 Hz tremor). No weights are given. Population 24, 6 generations, top-8 elitism, best-of-4 sampling per genome. Mutations include "add/remove a vocal-burst tag" at 20%. Six edge cases with an average LoRA lift of +0.147. Stronger λ is reported to "over-drive into gibberish". No date or authors. Tag positions within the script are not specified. — [LAION project page](https://projects.laion.ai/moss-emotion-edgecase-evolution/)

### Inferences
- Novelty claims like "first evolutionary search over TTS markup" are **no longer safe**. Dramatizer must cite these pages and narrow its claim to: (a) QD/MAP-Elites illumination over prosody descriptors with a novelty (Wundt) term instead of single-objective maximisation; (b) markup-only control of a frozen zero-shot cloning model (no LoRA); (c) position-aware, transferable genes; (d) a human-in-the-loop final pick from a diverse archive; (e) cross-voice seeding by speaker-embedding similarity.
- These are grey literature (non-peer-reviewed web reports), but they are public prior art for novelty purposes.

### Gaps
- The publication date and authorship are not on either page. I could not find a GitHub repo or blog post linking to them. Archived snapshots (Wayback) would be needed to date them.

## Q4. arXiv 2608.00998: interactive GA for personalised emotional TTS (Interspeech 2026)

### Takeaway
This is an interactive GA in which the listener selects preferred samples. The search space is a **2-D arousal–valence coordinate** fed to a Grad-TTS emotion controller, not markup. **Threat to novelty: partial, for C3 only.** It is prior art for "evolutionary search with a human selecting from candidates in TTS", but the genome (A–V point), the model (a trained, non-cloning Grad-TTS) and the goal (personalised perception mapping) all differ. There is no automatic scoring, no QD and no cross-voice transfer.

### Cited Findings
- "Beyond One-Size-Fits-All: Personalized and Culturally Adaptive Emotional TTS via Interactive Optimization of Individual Emotion Perception Spaces". Authors: Wangzixi Zhou, Bagus Tris Atmaja, Sakriani Sakti. Submitted 2 Aug 2026. Comment: "Accepted for publication at INTERSPEECH 2026". — [arXiv abs](https://arxiv.org/abs/2608.00998)
- The individual is "an A–V coordinate e=[a,v]∈ℝ²". The system "generates a population of candidate A–V coordinates", and the listener picks the best-matching sample(s), which become parents. Operators: multi-parent arithmetic crossover, and uniform mutation U(−M_g, M_g) with M_1=0.20, γ=0.90, M_min=0.05. Converges "within three rounds". Backbone is Grad-TTS with FastSpeech2-style pitch/energy predictors and HiFi-GAN. Population size N is not reported. — [arXiv HTML](https://arxiv.org/html/2608.00998)

### Inferences
- Cite it as the most recent peer-reviewed interactive-evolution TTS work, and as evidence that IGA-for-TTS is an established idea. Dramatizer's difference is that humans only pick from an automatically pre-filtered QD archive rather than drive every generation.

### Gaps
- None material.

## Q5. Step-Audio-EditX (Nov 2025)

### Takeaway
Its "iterative editing" re-applies the editing model to the previous output audio for N=3 rounds. There is no candidate population, tag search or automatic selection at inference. The Gemini judge is evaluation only. **Threat to novelty: no.** It is an alternative control mechanism (audio-domain editing with tags) and possibly a useful backend or baseline.

### Cited Findings
- "Step-Audio-EditX Technical Report", StepFun (corresponding author yanchao@stepfun.com). arXiv 2511.03601, v1 dated 5 Nov 2025 and v2 dated 19 Nov 2025. — [arXiv HTML v2](https://arxiv.org/html/2511.03601v2)
- "The process begins with a zero-shot clone as the initial audio". "The output of the N-th round is denoted as iteration_N" (N=3). Evaluation: "All evaluation audio is generated via zero-shot voice cloning and subsequently scored using the Gemini-2.5-Pro" model, with paralinguistics rated on a 1–3 scale. A margin-scoring model (1–10) is used only for training-data construction. — [arXiv HTML v2](https://arxiv.org/html/2511.03601v2)

### Inferences
- Its scoring components (an LLM judge and a margin scorer) could inspire Dramatizer's fitness function, but they do not overlap with search over markup.

### Gaps
- I did not confirm whether the same instruction is reused each round. The paper implies it is.

## Q6. Targeted venue sweep (GECCO, EvoMUSART, SSW, Interspeech, ICASSP, QD paper list)

### Takeaway
In the sources I could open, I found no evolutionary, QD or novelty-search work applied to TTS prompts or prosody markup. The sweep is **incomplete**: the ACM GECCO tables of contents, the Springer EvoMUSART 2026 TOC (behind a login redirect), SSW 2025 and ICASSP 2026 were not browsed directly. Searches returned nothing relevant from them. One relevant adjacent item surfaced: a human-in-the-loop prosody-transfer paper (Interspeech 2024).

### Cited Findings
- QD paper list: the first ~100k characters (entries for 2025, 2024 and part of 2023, of 262 total) contain no title mentioning speech, voice, TTS, audio, sound, prosody, music or vocal. The newest entries are from 2025, so the list appears not to be maintained into 2026. — [quality-diversity.github.io/papers](https://quality-diversity.github.io/papers)
- A general web search for MAP-Elites/QD plus TTS surfaced no QD-for-TTS paper. QDAIF and LLM-as-QD-generator papers are text-domain. — [QDAIF arXiv](https://arxiv.org/abs/2310.13032)
- EvoMUSART 2026 (Toulouse, 8–10 Apr 2026, LNCS, 13 full + 15 short papers) lists "voices" in scope, but I could not open the TOC (Springer redirect). EvoMUSART 2027's deadline is 1 Nov 2026. — [Springer book page (not opened)](https://link.springer.com/book/10.1007/978-3-032-24350-8); [EvoStar 2026](https://www.evostar.org/2026/evomusart/)
- The GECCO 2025 search returned only general conference pages. No speech/TTS paper surfaced. — [GECCO '25 proceedings](https://dl.acm.org/doi/proceedings/10.1145/3712256) (TOC not browsed)
- The Interspeech 2026 search found no evolutionary prompt-search paper. Adjacent items were HPRO (reward optimisation for emotional TTS, arXiv 2606.28249) and a WildSpoof 2026 entry using LLM scoring to *select* reference prompts (arXiv 2605.23859). Neither was opened. — [search results only; unverified]
- Adjacent, opened: Maurya & Sigurgeirsson, "A Human-in-the-Loop Approach to Improving Cross-Text Prosody Transfer", arXiv 2406.06601 (6 Jun 2024), Interspeech 2024. Humans adjust "salient correlates of prosody" of a reference-based rendition, and adjusted versions were preferred 57.8% of the time. The paper notes that closeness in the latent reference space is not a reliable prosodic-similarity metric for cross-text transfer. — [arXiv abs](https://arxiv.org/abs/2406.06601)

### Inferences
- Maurya & Sigurgeirsson supports Dramatizer's human-pick stage and its cross-line transfer motivation. It is also a caution: transfer via reference latents is weak, which favours Dramatizer's markup-level transfer. Threat to novelty: no, but it should be cited.
- Overall, the peer-reviewed literature still appears to lack QD/MAP-Elites over TTS input markup. The LAION GA pages are the main exception for plain GA over prompts and tags.

### Gaps
- GECCO 2024–2026, EvoMUSART 2024–2026, SSW 2025 and ICASSP 2026 TOCs were not browsed title by title. The remaining ~375k characters of the QD list (pre-2023) were not read. HPRO and the WildSpoof paper were not opened.

## Q7. Foundational citation verification

### Takeaway
All IDs and authors check out on arXiv. QDAIF's ICLR 2024 venue was not confirmed from the arXiv record.

### Cited Findings
- arXiv 1504.04909, "Illuminating search spaces by mapping elites", Jean-Baptiste Mouret & Jeff Clune, 20 Apr 2015 (comment "Early draft"). The correct title is not "MAP-Elites". — [arXiv API](https://export.arxiv.org/api/query?id_list=1504.04909)
- CMA-ME: "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space", Fontaine, Togelius, Nikolaidis, Hoover, arXiv 1912.02400. Comment "Accepted to GECCO 2020". DOI 10.1145/3377930.3390232. — [arXiv API](https://export.arxiv.org/api/query?id_list=1912.02400)
- QDAIF: "Quality-Diversity through AI Feedback", Bradley, Dai, Teufel, Zhang, Oostermeijer, Bellagente, Clune, Stanley, Schott, Lehman, arXiv 2310.13032 (19 Oct 2023). The arXiv record has no venue. ICLR 2024 is **unverified** here. — [arXiv API](https://export.arxiv.org/api/query?id_list=2310.13032)
- arXiv 1803.09047, "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron", Skerry-Ryan et al., 24 Mar 2018. — [arXiv API](https://export.arxiv.org/api/query?id_list=1803.09047)
- arXiv 1803.09017, "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis", Yuxuan Wang et al., 23 Mar 2018. — [arXiv API](https://export.arxiv.org/api/query?id_list=1803.09017)
- Sigurgeirsson & King, "Do Prosody Transfer Models Transfer Prosody?", arXiv 2303.04289, "Accepted at ICASSP 2023". They find that models trained with a different but prosodically related reference perform "significantly worse", which suggests the models learn utterance-level, speaker- and text-tied representations rather than transferable prosody. — [arXiv abs](https://arxiv.org/abs/2303.04289)

### Inferences
- Use the exact MAP-Elites title in the bibliography. The Sigurgeirsson & King finding directly motivates Dramatizer's choice to transfer markup rather than reference embeddings.

### Gaps
- The QDAIF ICLR 2024 acceptance needs confirmation via OpenReview.
