# Automatic generation of expressive markup / SSML / prosody annotations for TTS (prior work for Dramatizer)

Scope note: research done 2026-10-27 through web search plus arXiv abstract and HTML fetches (about 18 tool calls). Items marked "(snippet-only)" were seen only in search results or abstracts, not read in full. 2025-2026 items are flagged with dates. Closeness ratings are judged against Dramatizer, which runs an evolutionary search over position-relative paralinguistic markup genomes for CosyVoice 2/3. Candidates are scored automatically, a human picks from a diverse shortlist, and the winning genome seeds the search for the next voice.

## Q1. LLM-generated SSML / prosody markup / emphasis for TTS

### Takeaway
A small but growing body of work has LLMs write SSML, emphasis positions or relative prosody changes in a single pass. The strongest empirical finding for Dramatizer comes from Ould Ouali et al. (ICNLSP 2025). They benchmarked seven open LLMs and found that all of them "consistently under-generate tags", producing structurally incomplete SSML. A fine-tuned cascade beat prompting-only LLMs. All of this work predicts the markup once; none of it searches over markup.

### Cited Findings
- **Improving French Synthetic Speech Quality via SSML Prosody Control.** Ould Ouali, Sani, Bueno, Dauvet, Horstmann, Moulines. ICNLSP 2025, arXiv:2508.17494 (24 Aug 2025). Code: https://github.com/hi-paris/Prosody-Control-French-TTS.
  - **Predicts:** break positions (QwenA), then numeric pitch, rate, volume and break duration for each prosody element (QwenB). Both are Qwen2.5-7B models fine-tuned with QLoRA, rendered with Azure fr-FR-HenriNeural.
  - **Evaluation:** break F1 99.2%; MAE on pitch, rate and volume 25-40% lower than prompting-only LLMs and a BiLSTM; MOS rose from 3.20 to 3.87 (p<0.005) with n=18 listeners, and 15/18 listeners preferred the SSML version.
  - **Source:** [arXiv abs](https://arxiv.org/abs/2508.17494)
- **Same paper, LLM baselines.** Seven open models were benchmarked through Ollama: Mistral 7B, Qwen2.5 7B, Llama 3 8B, Granite 3.3 8B, Qwen3 8B, DeepSeek-R1 32B and Qwen3 32B.
  - The authors write that "All models consistently under-generate tags" and that LLM methods "often produce incomplete tags, invalid syntax, or imprecise prosodic control".
  - Few-shot prompting gave fewer break tags but more prosody tags. Llama 3's prosody tagging was "almost collapsing" under few-shot prompting.
  - This is direct evidence that LLM-written markup underuses tags.
  - **Closeness: HIGH.** It is the clearest baseline result of the "LLM writes markup" kind. It differs from Dramatizer in four ways: one-shot prediction, supervision from human-read reference prosody, Azure SSML rather than zero-shot voice-cloning event tokens, and no search or listener loop.
  - **Source:** [arXiv HTML](https://arxiv.org/html/2508.17494)
- **Using a Large Language Model to Control Speaking Style for Expressive TTS.** Sigurgeirsson and King (authors recalled from memory, not re-verified). SSW 2023.
  - **Predicts:** InstructGPT is prompted to propose relative changes to F0, energy and duration for FastSpeech 2, at both utterance level and word level, relative to the TTS model's own default predictions.
  - **Evaluation:** not fetched (snippet-only).
  - **Closeness: MEDIUM-HIGH.** This is an early "LLM as prosody annotator" work. It is one-shot, it controls acoustic features rather than text markup, and it does not use voice cloning.
  - **Source:** [ISCA SSW 2023 PDF](https://www.isca-archive.org/ssw_2023/sigurgeirsson23_ssw.pdf)
- **EME-TTS: Unlocking the Emphasis and Emotion Link in Speech Synthesis.** Interspeech 2025, arXiv:2507.12015.
  - **Predicts:** at inference an LLM predicts emphasis positions from the emotion label and the text. The authors argue LLMs infer emphasis better than earlier semantic or grammar-based predictors.
  - **Evaluation:** reported to give more emotionally expressive speech, especially when context is present. Exact metrics were not fetched (snippet-only).
  - **Closeness: MEDIUM.** LLM emphasis placement could be an "LLM baseline" gene source. It is one-shot and tied to a custom emphasis-aware TTS.
  - **Sources:** [ISCA PDF](https://www.isca-archive.org/interspeech_2025/li25i_interspeech.pdf); [arXiv](https://arxiv.org/html/2507.12015)
- **StressTransfer: Stress-Aware Speech-to-Speech Translation with Emphasis Preservation.** arXiv:2510.13194 (Oct 2025).
  - **Predicts:** an LLM adds explicit emphasis markers to the translated text. CosyVoice2 renders them.
  - **Evaluation:** emphasis is evaluated with a "Sentence Stress Reasoning Accuracy" metric (snippet-only).
  - **Closeness: MEDIUM-HIGH.** It uses the same TTS family (CosyVoice2) and inline emphasis markup written by an LLM, but it is one-shot and serves translation.
  - **Source:** [arXiv](https://arxiv.org/html/2510.13194)
- **Prosody Analysis of Audiobooks.** Pethe, Pham, Childress, Yin, Skiena. IEEE ICSC 2025, arXiv:2310.06930 (v1 Oct 2023, v3 Jan 2025).
  - **Predicts:** pitch, volume and rate from narrative text, using a language model trained on 93 aligned book-audiobook pairs.
  - **Evaluation:** predicted pitch correlated with human readings better than a commercial TTS for 22/24 books, and volume for 23/24.
  - **SSML listening test:** according to the search-result summary, a majority of annotators preferred the SSML-enhanced version in 12 of 24 sample pairs. The abstract does not describe the SSML conversion.
  - **Closeness: MEDIUM.** It is one-shot prediction from text and is evaluated against human audiobook narrators.
  - **Source:** [arXiv](https://arxiv.org/abs/2310.06930)
- **Hobbyist and open-source tools.**
  - "Text-To-SSML-Generator", a Gemini-based utility, converts plain text into SSML with inferred emotion and prosody. Its author calls it vibe-coded, so treat it as a demo. [GitHub](https://github.com/danielrosehill/Text-To-SSML-Generator)
  - "cue" sits between an LLM and a TTS. It injects [Breathing], [Laughter] and [Sigh] tags using rules: "haha" becomes a laugh tag, and ellipses get a breath. It caps tags at 2 per 100 words. [GitHub](https://github.com/solace-embermind/cue)
  - **Closeness: MEDIUM.** These are practical "LLM or rules write the tags" baselines with no evaluation.
- **EmphTTS** (arXiv:2609.27599, Sep 2026) uses GRPO to tune a duration predictor for word-level emphasis control.
  - **EMORL-TTS** (arXiv:2510.05758, Oct 2025) uses RL for fine-grained emotion control in LLM-based TTS. It notes that explicit discrete prosody tokens need "extensive annotation" and are hard to scale.
  - Both are snippet-only. Neither predicts text markup; both optimise the model rather than the input. **Closeness: LOW-MEDIUM.**
  - **Sources:** [EmphTTS](https://arxiv.org/pdf/2609.27599); [EMORL-TTS](https://arxiv.org/html/2510.05758v2)

### Inferences
- Under-generation is documented for small and medium open LLMs (7B-32B). No equivalent benchmark was found for frontier models such as GPT-4o/5, Claude or Gemini. Dramatizer should therefore run its own LLM baseline rather than assume the finding carries over.
- Dramatizer's search can explore tag density and placement directly. That directly addresses the documented under-generation failure mode of one-shot LLM markup.

### Gaps
- No paper found that benchmarks LLM-written markup specifically for CosyVoice 2/3's inline token set ([breath], [laughter], <strong>, instruct text).
- No large-scale "SSML generation" benchmark beyond the French study was found.

## Q2. Classic emphasis / prominence / phrase-break prediction from text

### Takeaway
There is a mature non-LLM literature: BERT-based word prominence on the Helsinki Prosody Corpus, and BERT or BiLSTM phrase-break and pause prediction. All of it is supervised, one-shot sequence labelling evaluated by accuracy or F1, plus sometimes MOS.

### Cited Findings
- **Predicting Prosodic Prominence from Text with Pre-trained Contextualized Word Representations.** Talman, Suni, Celikkanat, Kakouros, Tiedemann, Vainio. NoDaLiDa 2019 (venue from memory), arXiv:1908.02262.
  - Introduces the Helsinki Prosody Corpus: automatic prominence labels over the LibriTTS clean subsets, 262.5 hours from 1230 speakers.
  - BERT beat the other sequence labellers. An aggregator lists accuracy as BERT 83.2%, BiLSTM 82.1%, CRF 81.8% and SVM 80.8% (aggregator figures, unverified against the paper).
  - Code: Helsinki-NLP/prosody.
  - **Closeness: LOW-MEDIUM.** It could seed emphasis-gene positions; it is supervised and has no listening loop.
  - **Sources:** [arXiv](https://arxiv.org/pdf/1908.02262); [leaderboard](https://vercel.hyper.ai/en/sota/tasks/prosody-prediction/benchmark/prosody-prediction-on-helsinki-prosody-corpus)
- **What does BERT learn about prosody?** Kakouros and O'Mahony. ICPhS 2023 (venue from memory), arXiv:2304.12706. A layer-weight probing study of prominence. **Closeness: LOW.**
  - **Source:** [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.12706)
- **Phrase-break and pause prediction.**
  - arXiv:2304.04157 compares a BLSTM with a fine-tuned BERT for external phrase-break prediction in end-to-end TTS, and reports better listener comprehension. [arXiv](https://arxiv.org/html/2304.04157v3)
  - Futamata et al., arXiv:2104.12395 (Interspeech 2021), combine BERT and BiLSTM features for Japanese. Prosody naturalness MOS was 4.39, versus 4.37 for ground-truth breaks. [arXiv](https://arxiv.org/abs/2104.12395v1)
  - "Duration-aware pause insertion using pre-trained language model for multi-speaker TTS" predicts respiratory and punctuation pauses by duration class. [DeepAI](https://www.deepai.org/publication/duration-aware-pause-insertion-using-pre-trained-language-model-for-multi-speaker-text-to-speech)
  - PauseSpeech (Korea University, arXiv:2306.07489) uses a BERT context encoder and a pause predictor. [arXiv](https://arxiv.org/html/2306.07489v1)
  - "Speaker-Conditioned Phrase Break Prediction ... with Phoneme-Level PLM" is arXiv:2509.00675 (Sep 2025). Its speaker conditioning is loosely analogous to Dramatizer's per-voice adaptation. [arXiv](https://arxiv.org/html/2509.00675v1)
  - **Closeness, all: LOW.**
- **Search-result summary of a wavelet-label prominence study.** BERT was fine-tuned on 3-level wavelet-based prominence labels (F1 0.588), enabling controllable FastSpeech 2. Exact paper not identified (snippet-only). [search summary context: ISCA/arXiv results above]

### Inferences
- These predictors could provide informed priors or initial populations for emphasis and pause genes, instead of uniform random positions.

### Gaps
- None of the classic work evaluates against zero-shot voice-cloning models or uses listener-in-the-loop selection.

## Q3. Audiobook / podcast / drama / dialogue pipelines that assign emotion or style per line; captioning-then-TTS

### Takeaway
Recent pipelines (2025-2026) use an LLM, usually GPT-4o-class, to write a per-sentence natural-language delivery instruction or persona, which an instruction-following TTS renders. Podcast systems mostly have the LLM write spontaneous scripts with fillers, not markup. Dataset work uses LLMs to write style captions for training prompt-TTS models.

### Cited Findings
- **MultiActor-Audiobook: Zero-Shot Audiobook Generation with Faces and Voices of Multiple Speakers.** Park, Joo, Jung. arXiv:2505.13082 (May 2025).
  - Two steps: Multimodal Speaker Persona Generation, which builds voices from face images and captions via FleSpeech, and LLM-based Script Instruction Generation.
  - In the instruction step, GPT-4o receives the sentence, its context and the character persona, and writes one concise instruction per sentence covering emotion, tone and pitch, accounting for emotional transitions between sentences.
  - **Evaluation:** human and MLLM evaluation; MOS competitive with commercial audiobook products; a 0.225 average gain in automated assessment (from an aggregator summary). The authors admit lower quality than some commercial systems because of the FleSpeech backbone.
  - **Closeness: HIGH.** It is the closest "LLM writes per-line instructions" baseline. It differs from Dramatizer in being one-shot, using natural-language instructions only (no event tokens or position genes), and having no search or iteration.
  - **Sources:** [arXiv abs](https://arxiv.org/abs/2505.13082); [HTML](https://arxiv.org/html/2505.13082)
- **Audiobook-CC: Controllable Long-context Speech Generation for Multicast Audiobook.** Min Liu, JingJing Yin, Xiang Zhang et al. arXiv:2509.17516 (Sep 2025), listed as Interspeech 2026.
  - Pipeline: chapter segmentation, then text and character-persona analysis, then voice assignment, then synthesis. It adds a context mechanism, style/prompt disentanglement and self-distillation.
  - Demo at everest-ai.github.io. Evaluation details were not fetched.
  - **Closeness: MEDIUM.** The control is in the model, not a markup search.
  - **Sources:** [arXiv](https://arxiv.org/html/2509.17516v1); [papers.cool](https://papers.cool/venue/liu26p@interspeech_2026@ISCA)
- **MoonCast: High-Quality Zero-Shot Podcast Generation.** Ju, Yang et al. arXiv:2503.14345 (Mar 2025), OpenReview.
  - An LLM writes spontaneous scripts (fillers, turn-taking). A long-context two-speaker model renders them: about 2.5B parameters, 40K-token context, about 500k hours of training data.
  - **Closeness: LOW-MEDIUM.** The LLM shapes the text, including disfluencies, but not prosody tags.
  - **Sources:** [arXiv](https://arxiv.org/pdf/2503.14345); [OpenReview](https://openreview.net/pdf/2efcdcc28ba78697d4a58d816011484b0f441f8b.pdf)
- **CoVoMix, CoVoMix2 and related dialogue models.**
  - CoVoMix does zero-shot multi-speaker, multi-round dialogue generation and was trained on Fisher.
  - CoVoMix2 (arXiv:2506.00885, Jun 2025) is a fully non-autoregressive flow-matching follow-up trained on podcasts, audiobooks and simulated overlap.
  - Other 2025 dialogue/podcast TTS:
    - SoulX-Podcast, arXiv:2510.23541. [arXiv](https://arxiv.org/pdf/2510.23541)
    - DialoSpeech, arXiv:2510.08373. [arXiv](https://arxiv.org/html/2510.08373v1)
    - FireRedTTS-2, arXiv:2509.02020. [arXiv](https://arxiv.org/pdf/2509.02020)
  - Whether these accept per-line paralinguistic tags was not verified. **Closeness: LOW.**
  - **Sources:** [CoVoMix2](https://www.alphaxiv.org/overview/2506.00885); [MoonCast paper on CoVoMix](https://arxiv.org/pdf/2503.14345v1)
- **Computational Narrative Understanding for Expressive Text-to-Speech (LibriQuote).** Michel, Epure, Cerisara. arXiv:2509.04072 (Sep 2025, rev. Apr 2026), Findings of ACL 2026.
  - About 5.3K hours of audiobook character quotes, with contextual pseudo-labels of speech verbs and adverbs ("whispered softly").
  - Fine-tuning a flow-matching TTS on it improves expressivity and intelligibility. The paper also provides an expressivity benchmark.
  - **Closeness: MEDIUM.** It is a useful expressivity benchmark and a source of context-derived delivery cues.
  - **Source:** [arXiv](https://arxiv.org/abs/2509.04072)
- **SceneTTS-Bench: A Benchmark for Scene-Level TTS in Drama Dubbing.** arXiv:2609.26255 (Sep 2026, snippet-only). **Closeness: MEDIUM** as an evaluation resource. [arXiv](https://arxiv.org/pdf/2609.26255)
- **Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability.** arXiv:2601.22661 (Jan 2026, snippet-only). Proposes an automatic expressiveness reward for role-play TTS. This is potentially useful as an extra Dramatizer fitness term. [arXiv](https://arxiv.org/pdf/2601.22661)
- **Step-Audio-EditX Technical Report.** arXiv:2511.03601 (Nov 2025). An editable expressive TTS/LLM, seen only in search results. [arXiv](https://arxiv.org/pdf/2511.03601)
- **Captioning-then-TTS datasets.**
  - LibriTTS-P (arXiv:2406.07969) uses templates plus Llama2 augmentation to write style prompts. [arXiv](https://arxiv.org/html/2406.07969v1)
  - ParaSpeechCaps (arXiv:2503.04713, Mar 2025) uses Gemini to verify situational tags and fine-tunes Parler-TTS. [arXiv](https://arxiv.org/html/2503.04713v2)
  - SpeechCraft (arXiv:2408.13608) has about 2,000 hours and more than 2M clips captioned by expert classifiers and captioning models. [arXiv](https://export.arxiv.org/pdf/2408.13608)
  - TextrolSpeech (arXiv:2308.14430) has 330 hours and 236,220 style descriptions. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.14430)
  - **Closeness: LOW.** These are training data for instruction-TTS, not inference-time markup.
- **Non-verbal tag pipelines.**
  - NVSpeech (arXiv:2508.04195, Aug 2025) builds 48,430 utterances, a paralinguistic-aware ASR that writes inline tags, and a TTS that renders vocalizations at token positions. [arXiv](https://arxiv.org/pdf/2508.04195)
  - arXiv:2508.05385 extends the F5-TTS vocabulary with [laughing] and [breath] tags. [arXiv](https://arxiv.org/pdf/2508.05385)
  - arXiv:2609.14231 (Sep 2026) reports that systems still struggle to place a requested vocalization convincingly at a specified position. [arXiv](https://arxiv.org/pdf/2609.14231)
  - Related 2026 resources: NVV-SuperBench (arXiv:2604.16211) and NVV-Locator (arXiv:2609.09940). [SuperBench](https://arxiv.org/pdf/2604.16211); [Locator](https://arxiv.org/pdf/2609.09940)
  - **Closeness: MEDIUM.** They validate the [breath]/[laughter] tag paradigm and its placement unreliability, which is a motivation for Dramatizer's validity scoring.

### Inferences
- The dominant 2025-2026 recipe is "LLM writes one instruction per line, then instruct-TTS renders it". MultiActor-Audiobook is the canonical example and is what a reviewer will expect Dramatizer to beat.
- The 2026 finding that tag placement is unreliable supports scoring many candidates and selecting among them, rather than trusting one prediction.

### Gaps
- No paper was found that uses CosyVoice 3's instruct and inline-tag interface for audiobook or drama with per-line LLM tags and reports a tag-usage analysis.

## Q4. LLM or agent critique-and-revise loops for TTS output

### Takeaway
By 2026 there are closed-loop systems in which an audio LLM listens, critiques and drives re-synthesis: LoopTTS (EMNLP 2026) and AgentSteerTTS (2026). Their loops edit instructions or latents with an automatic judge. None is human-in-the-loop, none is population-based, and none transfers solutions across voices.

### Cited Findings
- **Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction (LoopTTS).** Song, Liu, Wang, Xu, Guo, Haizhou Li. arXiv:2608.28970 (29 Aug 2026), EMNLP 2026 main.
  - Filter-Judge-Refiner design. WER and UTMOS filter out catastrophic outputs. An AudioLLM judge detects word-level problems (misplaced stress, unnatural pauses, flat intonation) and writes structured refine instructions. An instruction-following Refiner TTS re-synthesizes, conditioned on the draft, text and instruction. It is trained on Refiner-DB (42K examples).
  - **Evaluation:** human evaluation on low-quality utterances; beats raw output and open-loop regeneration. Numbers and iteration count were not in the abstract.
  - **Closeness: HIGH.** It is a closed loop with an automatic critic. It differs from Dramatizer in using a single trajectory, a trained refiner model rather than markup on a frozen TTS, no human selection and no cross-voice transfer.
  - **Source:** [arXiv](https://arxiv.org/abs/2608.28970)
- **AgentSteerTTS: A Multi-Agent Closed-Loop Framework for Composite-Instruction Text-to-Speech.** arXiv:2605.17583 (May 2026, snippet-only).
  - A slow path uses a multimodal LLM (Qwen3 or Gemini mentioned) to listen and write a natural-language critique. A fast path does latent gradient correction of emotion intensity.
  - **Closeness: HIGH-MEDIUM.** An LLM critiques and revises instructions. It is not evolutionary, has no human shortlist, and the fast path edits latents rather than markup.
  - **Source:** [arXiv](https://arxiv.org/html/2605.17583)
- **Self-Refine (Madaan et al., 2023)** is the generic generate-feedback-refine protocol these build on, cited in the search summary. **SI-Agent** (arXiv:2507.03223, Jul 2025) iteratively revises system instructions with instructor, follower and feedback agents. It is text-only. **Closeness: LOW.** [SI-Agent](https://arxiv.org/pdf/2507.03223)

### Inferences
- An "LLM critic plus LLM reviser of markup" loop on a frozen CosyVoice is a natural second baseline beyond one-shot LLM markup. LoopTTS and AgentSteerTTS are the closest prior art for that idea.

### Gaps
- No work was found where an LLM revises SSML or inline-tag markup itself, as opposed to natural-language instructions, based on listening to the TTS output.

## Q5. How expressiveness is evaluated; evidence of tag underuse or misuse

### Takeaway
Evaluation is mostly MOS, A/B preference with small listener panels (often 18-30), emotion-recognition or MLLM judging, and correlation with human-narrator prosody. Recent work adds LALM-based rewards. Tag underuse by LLMs is explicitly documented in one paper (Ould Ouali et al. 2025). Tag-placement unreliability on the TTS side is documented in 2026 nonverbal-vocalization work.

### Cited Findings
- **MOS and preference.** French SSML study: MOS 3.20 to 3.87, 15/18 listeners preferred SSML. [arXiv](https://arxiv.org/abs/2508.17494)
- **Correlation with human narrators, plus preference.** Prosody Analysis of Audiobooks: 22/24 books for pitch, 23/24 for volume. [arXiv](https://arxiv.org/abs/2310.06930)
- **Human and MLLM judges.** MultiActor-Audiobook uses MOS against commercial products. [arXiv](https://arxiv.org/abs/2505.13082)
- **Task-specific accuracy.** StressTransfer uses Sentence Stress Reasoning Accuracy. [arXiv](https://arxiv.org/html/2510.13194) Phrase-break work uses F1 and naturalness MOS. [arXiv](https://arxiv.org/abs/2104.12395v1)
- **Automatic filters and gates.** LoopTTS uses WER and UTMOS as gates before the AudioLLM judge, which is similar to Dramatizer's WER and validity gating. [arXiv](https://arxiv.org/abs/2608.28970)
- **LALM-based expressiveness reward.** Mean Continuation Log-Probability (arXiv:2601.22661). [arXiv](https://arxiv.org/pdf/2601.22661)
- **Underuse and misuse evidence.**
  - "All models consistently under-generate tags"; Llama 3 few-shot prosody tagging was "almost collapsing". [arXiv HTML](https://arxiv.org/html/2508.17494)
  - TTS systems struggle to realise vocalizations at requested positions. [arXiv 2609.14231](https://arxiv.org/pdf/2609.14231)

### Inferences
- None of the surveyed works uses a novelty or diversity criterion such as a Wundt curve, or presents a diverse shortlist to a human. Dramatizer's evaluation design appears novel relative to this literature.

### Gaps
- Exact listener counts and metrics for MultiActor-Audiobook, Audiobook-CC and AgentSteerTTS were not verified from the full texts.

## Q6. Evolutionary or black-box search over markup (instead of one-shot prediction)

### Takeaway
No work was found that runs an evolutionary or black-box optimiser over SSML, paralinguistic tags or TTS text-prompt markup. The nearest neighbours fall into three groups: (a) an Interactive Genetic Algorithm over emotion arousal-valence spaces for personalised emotional TTS (Interspeech 2026); (b) evolutionary or grammar-guided prompt optimisation for LLMs and image models; (c) closed-loop critique systems (Q4).

### Cited Findings
- **Beyond One-Size-Fits-All: Personalized and Culturally Adaptive Emotional TTS via Interactive Optimization of Individual Emotion Perception Spaces.** Zhou, Atmaja, Sakti. arXiv:2608.00998 (2 Aug 2026), Interspeech 2026.
  - Uses an Interactive Genetic Algorithm with listener feedback to personalise arousal-valence emotion representations. Participants came from Japan, China and Indonesia.
  - Personalised spaces gave more perceptually aligned emotion than averaged values. No numbers were in the abstract.
  - **Closeness: HIGH (methodologically).** It is the only evolutionary, human-feedback TTS optimisation found. It searches a continuous emotion-control space, not text markup, and it does not transfer genomes across voices or lines.
  - **Source:** [arXiv](https://arxiv.org/abs/2608.00998)
- **Grammar-Guided Evolutionary Search for Discrete Prompt Optimisation** (arXiv:2507.10326, Jul 2025). The grammar keeps edits syntactically valid, analogous to keeping markup valid. Not speech. **Closeness: MEDIUM** as a technique reference. [arXiv](https://arxiv.org/pdf/2507.10326)
- **PromptEvolver** (arXiv:2604.06061, Apr 2026). A GA with a VLM as the crossover and mutation operator, using black-box query access to the target generator. Image domain. **Closeness: MEDIUM.** [arXiv](https://arxiv.org/html/2604.06061v1)
- **A Toolbox for Improving Evolutionary Prompt Search** (arXiv:2511.05120, LUHME 2025). GA beats differential evolution on most text tasks. **Closeness: LOW.** [arXiv](https://arxiv.org/html/2511.05120v1)
- **Not found:** an explicit search for evolutionary, genetic or black-box optimisation of SSML or expressive TTS markup returned no relevant work. [search results](https://arxiv.org/html/2507.10326v1)

### Inferences
- Dramatizer's combination appears unclaimed in the literature found: population-based search over position-relative markup genes on a frozen zero-shot voice-cloning TTS, automatic multi-objective scoring with novelty, a human-picked shortlist, and seeding across voices.
- The closest pieces to cite and contrast are:
  - the Interactive GA (evolutionary search with human feedback, but continuous emotion space);
  - LoopTTS and AgentSteerTTS (closed loop, but single-trajectory and LLM-judged);
  - Ould Ouali et al. (the evidence that one-shot LLM markup under-generates).

### Gaps
- The searches cannot rule out unindexed workshop papers or industry systems. Coverage is limited to about 18 searches and fetches.
- The full method of the Interactive GA paper (genome encoding, population size) was not read.
