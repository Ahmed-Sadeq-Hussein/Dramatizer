# Automatic and Human Evaluation of Expressive and Cloned Speech, and the Pitfalls of Optimising Against Automatic Speech Metrics

Research date: 2026-10-27. Items dated 2025-2026 are flagged **[2025]** / **[2026]**. Several 2026 arXiv IDs (26xx.xxxxx) were found through search and checked only at abstract level, as noted per item. "Dramatizer use" tags: **REWARD** (fine as a search reward or guard), **HELD-OUT** (use only as an independent judge for reporting), **AVOID** (do not rely on it, or use only as a diagnostic).

---

## Q1. MOS predictors (UTMOS/UTMOSv2, DNSMOS, NISQA, SQUIM, VoiceMOS findings)

### Takeaway
Learned MOS predictors are accurate on in-domain, seen systems, but they get worse on unseen systems, unseen speakers and new domains (every VoiceMOS edition reports this). New evidence [2026] also suggests they penalise informal or expressive speaking styles. Dramatizer's SQUIM "perceptual quality" term estimates signal-level enhancement metrics, not TTS naturalness. It is a reasonable artefact/noise guard but a weak proxy for expressive quality. UTMOSv2 or a different predictor should be used only as a held-out judge.

### Cited Findings
- **VoiceMOS Challenge 2023** (Cooper, Huang, et al., ASRU 2023, arXiv 2310.02640) framed the challenge as *zero-shot* subjective quality prediction across several domains. The series moved to zero-shot because predictors trained on one listening test transfer poorly to others. — [arXiv 2310.02640](https://arxiv.org/pdf/2310.02640); companion lessons paper: "Zero-shot Out-of-domain is No Joke" — [ResearchGate](https://www.researchgate.net/publication/383654330_Zero-shot_Out-of-domain_is_No_Joke_Lessons_Learned_in_the_VoiceMOS_2023_MOS_Prediction_Challenge)
- **VoiceMOS Challenge 2024: "Beyond Speech Quality Prediction"** (Huang et al., SLT 2024). The tracks were (1) high-quality TTS samples, (2) singing/voice conversion and (3) semi-supervised quality prediction for noisy, clean and enhanced speech. Eight teams took part. The top systems reached correlations of about 0.939 and 0.979, but **unseen speakers and systems were harder to predict**. A gap remained between supervised and zero-shot settings, and most teams could not perform consistently across tracks with a single model. Retrieval-based kNN heads (RAMP) and non-SSL features (spectrograms, pitch histograms) helped. — [Paper PDF](https://homepage.citi.sinica.edu.tw/papers/yu.tsao/10634-F.pdf)
- **UTMOSv2** (UTokyo-SaruLab) is the official implementation of the "T05" VoiceMOS 2024 system. It combines SSL features with spectrogram features (an image-classifier backbone) and ranked first on 7 of 16 metrics in Track 1. A package-metadata site lists the licence as MIT; I did not confirm this from the LICENSE file. — [T05 paper](https://www.researchgate.net/publication/384075048_The_T05_System_for_The_VoiceMOS_Challenge_2024_Transfer_Learning_from_Deep_Image_Classifier_to_Naturalness_MOS_Prediction_of_High-Quality_Synthetic_Speech); [licence metadata](https://science.ecosyste.ms/projects/55946)
- **UTMOS22-strong** weights are reported as MIT by a third-party conversion page; I did not confirm this upstream. The PyPI package `utmos` is a wrapper "not affiliated with original authors". — [HF vokra/utmos22-strong](https://huggingface.co/vokra/utmos22-strong); [PyPI utmos](https://pypi.org/project/utmos/1.1.9/)
- **[2026] VoiceMOS Challenge 2026: "Evaluating Speech Enhancement, Emotional TTS and Accented TTS Systems"** (Huang, Wang, Sach, Gao, Sanders, Cooper, Toda; arXiv 2609.13792, 12 Sep 2026). Track 2 covers **naturalness and emotion-related prediction for emotional TTS**. Track 3 covers speaker and accent similarity for codec-based TTS. There were 18 teams, and most beat the baselines. I read only the abstract, so per-track results on expressive generalisation are unverified. — [arXiv 2609.13792](https://arxiv.org/abs/2609.13792)
- **[2026] Conversational Speech Naturalness Predictor** (arXiv 2603.01467): **NISQA and UTMOSv2 correlated *negatively* with human naturalness ratings for conversational speech.** The predictors gave higher scores to formal-sounding speech and lower scores to "chit-chatty" speech that humans preferred. — [arXiv 2603.01467](https://arxiv.org/html/2603.01467v1)
- **[2026] "Beyond Naturalness: Probing Automated TTS Evaluators on Linguistically Grounded Dimensions"** (Bamgbose et al., EMNLP 2026 main, arXiv 2608.09930). The authors build 860 utterances annotated by linguists on 10 perceptual dimensions and test 4 MOS predictors and 4 audio-LLM judges. MOS predictors are mostly sensitive to acoustic and signal-level properties. Audio-LLM judges are prompt-dependent and favour word-level errors. "Neither class reliably captures a breadth of linguistically structured, human-perceptible speech errors." The data and code are released. — [arXiv 2608.09930](https://arxiv.org/abs/2608.09930)
- **DNSMOS** (Microsoft, P.835 SIG/BAK/OVRL) was built for noise suppression. The DNSMOS Pro paper (Interspeech 2024) reports that the original DNSMOS has a negative bias relative to its training data. A NORESQA-MOS paper reports that DNSMOS has high bias and NISQA has higher variance across datasets. — [DNSMOS Pro, ISCA](https://www.isca-archive.org/interspeech_2024/cumlin24_interspeech.pdf); [NORESQA-MOS arXiv 2206.12285](https://arxiv.org/pdf/2206.12285)
- **NISQA-TTS** has been used as the automatic quality metric in emotional TTS (EmoSpeech, SSW 2023). That paper did not check whether NISQA is biased on emotional speech. — [EmoSpeech, ISCA SSW 2023](https://www.isca-archive.org/ssw_2023/diatlova23_ssw.pdf)
- **TorchAudio-SQUIM** (Kumar et al., ICASSP 2023, arXiv 2304.01448) provides reference-free estimators. `SQUIM_OBJECTIVE` predicts PESQ, STOI and SI-SDR. `SQUIM_SUBJECTIVE` predicts MOS using **non-matching references** (NORESQA-MOS), so it needs a clean reference clip from any speaker. It runs at 16 kHz only. In the tutorial, estimates closely track the true metrics at 20 dB SNR (estimated vs true STOI 0.961/0.967, PESQ 2.78/2.80) and diverge more at -5 dB SNR (estimated vs true SI-SDR -6.25/-5.02). — [arXiv 2304.01448](https://arxiv.org/pdf/2304.01448); [tutorial](https://pytorch.org/audio/2.1/tutorials/squim_tutorial.html)
- **[2025]** "Good practices for evaluation of synthesized speech" (Cooper, Le Maguer, Klabbers, Yamagishi; arXiv 2503.03250v2, Oct 2025) advises naming and justifying any MOS predictor, labelling its output "predicted MOS", and noting that predictors may generalise poorly outside their training domain. It also says objective metrics complement listening tests and do not replace them. — [arXiv 2503.03250](https://arxiv.org/html/2503.03250v2)
- **[2025] TTSDS2** (Minixhofer et al., SSW 2025) proposes robust distributional objective evaluation for human-quality synthetic speech. I did not read the details; it is listed as an alternative held-out judge. — [ISCA SSW 2025](https://www.isca-archive.org/ssw_2025/minixhofer25_ssw.pdf)

### Inferences
- SQUIM-objective (PESQ/STOI/SI-SDR) measures **degradation relative to clean speech**, not naturalness or expressiveness. Breathy, whispered, shouted or creaky deliveries probably look like "noise" to it, so as a positive reward it may push the search toward neutral delivery. Recommendation: **use SQUIM-objective as a floor/guard (threshold), not as a maximised reward term**.
- **Held-out judges:** report UTMOSv2 (MIT, VoiceMOS 2024 winner) and, if possible, a predictor from a different family (DNSMOS P.835 or NISQA). Because none of them is validated for emotional speech, report them as "artefact checks", not "expressive quality". When the VoiceMOS 2026 Track 2 systems become available, they are the closest fit for emotional-TTS naturalness.
- Given the [2026] conversational-naturalness result, expect MOS predictors to **penalise** the dramatised renders relative to neutral ones. A drop in predicted MOS is not proof of worse quality, so human tests must arbitrate.

### Gaps
- I found no controlled study that quantifies UTMOS, DNSMOS, NISQA or SQUIM bias against acted emotional speech specifically (closest: the conversational-naturalness paper). The VoiceMOS 2026 emotional-TTS results were not read beyond the abstract.
- I could not fetch the VoiceMOS 2022 paper (Huang et al., Interspeech 2022) or the original UTMOS paper (Saeki et al., Interspeech 2022) in this session. Both are well known but should be cited from their ISCA pages after checking.
- No 2025 VoiceMOS edition results page was found.
- DNSMOS and NISQA licences were not verified.

---

## Q2. Speaker similarity (Resemblyzer/GE2E, ECAPA-TDNN, WavLM-SV; biases)

### Takeaway
The de facto TTS standard is cosine similarity from **WavLM-large fine-tuned for speaker verification** (Seed-TTS-eval), with ECAPA-TDNN common in emotional and cross-lingual TTS. Resemblyzer (GE2E) is older and weaker and rarely used in recent papers. Emotional prosody measurably lowers SIM against a neutral reference, so a SIM guard tied to a neutral reference can work directly against expressiveness.

### Cited Findings
- **Seed-TTS** (Anastassiou et al., ByteDance, arXiv 2406.02430, 2024) computes SIM as cosine similarity between embeddings from **WavLM-large fine-tuned on speaker verification**, comparing the generated speech with the reference prompt. — [arXiv 2406.02430](https://arxiv.org/pdf/2406.02430)
- Later systems copy this protocol (FireRedTTS3, SemaVoice [2026], GLM-TTS [2025]). The k2-fsa `TTS_eval_models` repository bundles the eval models. — [FireRedTTS3](https://arxiv.org/html/2608.17492v2); [SemaVoice](https://arxiv.org/pdf/2605.16964); [GLM-TTS](https://arxiv.org/pdf/2512.14291); [HF k2-fsa/TTS_eval_models](https://huggingface.co/k2-fsa/TTS_eval_models)
- **Emotion lowers SIM.** A [2026] emotional TTS paper (CoCoEmo, arXiv 2602.03420, as surfaced by search) notes that SIM against a neutral reference "can decrease when emotional prosody moves away from the neutral reference and therefore conflates intended emotional variation with speaker-identity drift". It adds a same-emotion-reference variant, **SIM-EXP**. — [arXiv 2602.03420](https://arxiv.org/pdf/2602.03420)
- A Radboud MSc thesis (Mikhailovskii, 2024) on CREMA-D/RAVDESS reports degraded speaker verification under emotional variability for a WavLM + ECAPA-TDNN system. LDA, PLDA and contrastive tweaks helped little; fine-tuning with modified Barlow Twins plus augmentation helped. This is a thesis, not peer-reviewed. — [Thesis PDF](https://www.cs.ru.nl/masters-theses/2024/D_Mikhailovskii___Speaker_Similarity_for_Emotional_Speech.pdf)
- Emotion-aware speaker verification that fuses speaker and emotion cosine scores (Sensors 2025) — [PMC12431519](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431519/)
- A cross-lingual emotional TTS paper using ECAPA-TDNN reports reference bounds: same-speaker ground-truth cosine about 0.80 (upper bound) and source vs target speaker about 0.18 (lower bound). This is useful for calibrating what a SIM drop means. — [DiCLET-TTS arXiv 2309.00883](https://arxiv.org/pdf/2309.00883)
- A study of ECAPA-TDNN vs x-vector speaker representations in zero-shot multi-speaker TTS [2025] — [arXiv 2506.20190](https://arxiv.org/pdf/2506.20190)
- **[2026] VoiceMOS 2026 Track 3** asks predictors to estimate *human* speaker similarity ratings for codec TTS. This implies that embedding cosine and perceived similarity are considered different enough to need their own predictor. — [arXiv 2609.13792](https://arxiv.org/abs/2609.13792)

### Inferences
- **Dramatizer's Resemblyzer guard:** keep it as a guard (REWARD-side), but (a) set the threshold *relative* to the SIM that a neutral render of the same line gets, rather than as an absolute value, and (b) where possible use a same-emotion reference (the SIM-EXP idea).
- **Held-out judge:** report WavLM-large-SV SIM (Seed-TTS-eval protocol) and/or ECAPA-TDNN. These are different architectures and training data from GE2E/Resemblyzer, which avoids circularity and makes the numbers comparable with the literature. Report SIM against both the original reference and a neutral render.
- Optimising a single SV embedding can be exploited (adversarial-style); see Q7. Checking with a second, independent SV model is the cheap mitigation.

### Gaps
- I found no paper documenting Resemblyzer/GE2E accuracy relative to WavLM-SV on TTS outputs. I did not verify the Resemblyzer licence (believed Apache-2.0; unconfirmed).
- I found no controlled per-emotion-pair quantification of the ECAPA cosine drop.
- Channel and recording-condition bias in SV embeddings is well known, but I did not retrieve a source for it in this session.

---

## Q3. WER for TTS (Whisper-large-v3 / Paraformer; normalisation, hallucination, expressive delivery)

### Takeaway
Seed-TTS-eval uses Whisper-large-v3 (English) and Paraformer-zh (Mandarin). WER values depend strongly on text normalisation and ASR choice, and Whisper hallucinates on silence and trailing audio. For dramatised speech, long pauses, laughter, breaths and non-lexical vocalisations can inflate WER, or be "transcribed away", in ways unrelated to intelligibility.

### Cited Findings
- Seed-TTS: "we employ Whisper-large-v3 and Paraformer-zh as the ASR engines for English and Mandarin." — [arXiv 2406.02430](https://arxiv.org/pdf/2406.02430). The k2-fsa eval toolkit does the same. — [HF k2-fsa/TTS_eval_models](https://huggingface.co/k2-fsa/TTS_eval_models). Some [2026] papers use Whisper-large-v3-turbo instead, and Seed-TTS-eval English subset sizes are reported inconsistently (1,008 vs 1,088). — [Gradium benchmark](https://gradium.ai/content/on-device-tts-benchmark-2026); [EmphTTS arXiv 2609.27599](https://arxiv.org/pdf/2609.27599)
- The Whisper paper (Radford et al., 2022) introduced its own text normaliser because naive WER penalises "innocuous differences in transcript style". It also warns that tuning the normaliser risks overfitting. — [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
- Normalisation pitfalls in multilingual ASR evaluation (2024) — [arXiv 2409.02449](https://arxiv.org/pdf/2409.02449)
- Inworld's open-tts-eval: two evaluations on the same data can report different WER depending on ASR settings, normalisation, voices and synthesis parameters. Whisper often mishears names and accents, which inflates WER. — [GitHub README](https://github.com/inworld-ai/open-tts-eval/blob/main/README.md)
- Whisper hallucinates text in silent regions and after the end of input. Some work reports a separate hallucination rate on non-speech inputs and applies the Whisper English normaliser before counting. Cross-checking with an architecturally unrelated ASR avoids Whisper-specific bias. — [AURA arXiv 2609.23979](https://arxiv.org/pdf/2609.23979); [WhisperVC arXiv 2511.01056](https://arxiv.org/pdf/2511.01056) (as summarised by search; detail not independently read)
- **[2026]** Park & Li note that perceptual rewards can be gamed through "transcript drift" and keep a **CER zone constraint** (a band, not a minimisation) during GRPO. — [arXiv 2608.31035](https://arxiv.org/abs/2608.31035)

### Inferences
- **Dramatizer guard:** treat WER as a *thresholded* guard (for example, reject if WER exceeds the neutral render's WER by more than δ), not a minimised term. Pushing WER toward 0 favours over-articulated, flat delivery. Run Whisper with a fixed normaliser and temperature 0, and strip or tag stage directions and nonverbal tokens from the reference before scoring.
- **Held-out judge:** report WER with a *different* ASR family (Paraformer-en, a Conformer/Parakeet/CTC model, or a different Whisper size), plus hallucination checks (output on silence-padded audio). This avoids rewarding audio that happens to decode well under one Whisper checkpoint.
- Long dramatic pauses raise the risk of Whisper hallucination in silent regions, which interacts with Dramatizer's "pause share" descriptor. Trim or VAD-segment audio before ASR.

### Gaps
- I found no paper measuring how emotional intensity or nonverbal vocalisations (laughs, sobs) change Whisper WER on TTS output specifically. EmergentTTS-Eval's paralinguistics category is the nearest proxy (Q5).

---

## Q4. Expressiveness / emotion metrics (emotion2vec, SER accuracy, AutoPCP, SeamlessExpressive, EmoBox)

### Takeaway
Current practice is: (a) **emotion2vec embedding cosine similarity** between generated and reference speech ("Emo SIM"), (b) emotion-classifier accuracy on the intended label, and (c) **AutoPCP** utterance-level prosody similarity (from Meta's SeamlessExpressive work). All three are *reference-similarity* metrics. They reward copying the reference's emotion or prosody, which is not the same as Dramatizer's "different from neutral in the right direction".

### Cited Findings
- **emotion2vec** (Ma et al., ACL Findings 2024; self-supervised speech emotion representation) — [ResearchGate](https://www.researchgate.net/publication/384208856_emotion2vec_Self-Supervised_Pre-Training_for_Speech_Emotion_Representation)
- "Emo SIM" is computed as cosine similarity of emotion2vec-base embeddings between generated and reference speech, averaged over the test set. — [Emo-BPO arXiv 2610.12958](https://arxiv.org/html/2610.12958) **[2026]**
- **Laugh Now Cry Later / EmoCtrl-TTS** (Microsoft, arXiv 2407.12229, 2024) uses **AutoPCP_multilingual_v2** ("an utterance-level estimator that quantifies prosody similarity between two speech samples") and a time-varying Emo SIM (frame-wise cosine after length interpolation). Baseline comparison: SeamlessExpressive scored AutoPCP 2.31 / Emo SIM 0.587, vs EmoCtrl-TTS at 3.38 / 0.848 (Table 7). — [arXiv 2407.12229](https://arxiv.org/html/2407.12229v2)
- Emo-DPO (2024) uses emotion-preference optimisation and evaluates with emotion classifiers. — [arXiv 2409.10157](https://arxiv.org/pdf/2409.10157)
- **[2026] VoiceMOS 2026 Track 2** targets prediction of human "emotion-related" ratings for emotional TTS. This is the first challenge-validated route to predicting perceived emotion, not just classifier output. — [arXiv 2609.13792](https://arxiv.org/abs/2609.13792)
- **[2026]** Park & Li: single-reward GRPO with a learned "arousal" predictor mostly improves that predictor's own score, and "subjective predictors are not interchangeable quality surrogates". — [arXiv 2608.31035](https://arxiv.org/abs/2608.31035)

### Inferences
- **Held-out judges for Dramatizer:** (1) emotion2vec(-plus) predicted label vs the scene's intended emotion, plus the shift in emotion2vec embedding from the neutral render. (2) AutoPCP between the dramatised render and a *human actor's* reading of the same line, if any exist. (3) arousal/valence/dominance from an SER regressor (for example, a wav2vec2/WavLM MSP-Podcast model; not verified in this session). None of these overlaps with Dramatizer's six DSP descriptors, so they are proper non-circular judges.
- Do **not** reuse the six DSP prosody descriptors as reporting metrics for "expressiveness". They are the optimised quantities, so reporting them is circular. Report them only as manipulation checks (did the search move what it was meant to move).
- Classifier-based emotion accuracy can be gamed by exaggeration, since caricatured emotion classifies more easily. Pair it with human naturalness/appropriateness ratings.

### Gaps
- **EmoBox** (an emotion recognition toolkit/benchmark, believed to be Ma et al., Interspeech 2024) did not appear in my searches. Its datasets, splits and licence are unverified.
- I did not fetch the **SeamlessExpressive / Seamless** paper (Meta, 2023), which defines AutoPCP and the expressivity metrics (AutoPCP, rate and pause alignment). Its exact definitions, AutoPCP licence and availability should be checked from the facebookresearch/seamless_communication repo.
- I found no validation of emotion2vec similarity against human perception for TTS output.

---

## Q5. Benchmarks for expressive/controllable TTS and audio-LM-as-judge

### Takeaway
**EmergentTTS-Eval** (NeurIPS 2025 D&B) is the main expressive benchmark that uses an audio-LLM judge (Gemini 2.5 Pro win-rates). **InstructTTSEval** covers instruction/style following. Seed-TTS-eval is the standard for cloning (WER plus SIM). [2026] work shows audio-LLM judges hallucinate errors on emotional or prosodic items and are prompt-sensitive.

### Cited Findings
- **[2025] EmergentTTS-Eval** (Boson AI; arXiv 2505.23009; NeurIPS 2025 Datasets & Benchmarks). It has 1,645 test cases in six categories: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation and questions. Cases are expanded by an LLM from human-written seeds. A Large Audio Language Model judge (main: Gemini 2.5 Pro) produces win-rates against a baseline (gpt-4o-mini-tts), and the authors report correlation with human preference. The authors note that judging emotion and prosody can produce judge hallucinations, such as wrongly flagging pronunciation problems. Code is on GitHub. — [arXiv 2505.23009](https://arxiv.org/pdf/2505.23009); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/04970a25af46918606ba2cf0a3d7905d-Paper-Datasets_and_Benchmarks_Track.pdf); [GitHub](https://github.com/boson-ai/emergenttts-eval-public)
- **[2026]** Fish Audio S2 reports an 81.88% EmergentTTS-Eval win-rate against gpt-4o-mini-tts with a Gemini 2.5 Pro judge. This shows the benchmark is now used in industry reports. — [arXiv 2603.08823](https://arxiv.org/pdf/2603.08823)
- **InstructTTSEval** evaluates style-following across dimensions such as speaking rate, pitch and emotional expressiveness. A [2026] paper argues that neither it nor EmergentTTS-Eval provides attribute-level human ground truth. I did not find InstructTTSEval's original paper. — [arXiv 2608.09930](https://arxiv.org/pdf/2608.09930)
- **[2026] Beyond Naturalness** (EMNLP 2026): audio-LLM judges detect errors selectively, depend on the prompt and do not generalise across dimensions. — [arXiv 2608.09930](https://arxiv.org/abs/2608.09930)
- **[2026] Diagnose, Then Refine** (arXiv 2608.28970) uses an audio-LLM to diagnose and correct TTS errors in a closed loop. This is relevant prior work for Dramatizer's loop, though I read only the search snippet. — [arXiv 2608.28970](https://arxiv.org/pdf/2608.28970)
- **[2025]** A survey of automatic evaluation methods across text, visual and speech generation — [arXiv 2506.10019](https://arxiv.org/pdf/2506.10019)
- **[2026] GSRM: Generative Speech Reward Model for Speech RLHF** (arXiv 2602.13891); seen only as a search title. — [arXiv 2602.13891](https://arxiv.org/pdf/2602.13891)

### Inferences
- **Held-out judge:** an audio-LLM (Gemini 2.5 Pro or similar) asked pairwise "which reading better fits the scene direction?" is the closest automatic proxy to Dramatizer's goal. Randomise the A/B order, run both orders, and treat it as an auxiliary signal checked against humans.
- Run the EmergentTTS-Eval *emotions* and *paralinguistics* subsets as an external benchmark if the Dramatizer pipeline can take arbitrary text. Use Seed-TTS-eval (test-en) to show the cloning guard has not degraded baseline WER/SIM.

### Gaps
- I did not verify the original InstructTTSEval paper (authors, size, judge). I did not search for the EmoVoice evaluation protocol or other 2025-2026 expressive benchmarks beyond those listed.
- I found no study of position bias or self-preference in audio-LLM judges for TTS.

---

## Q6. Human evaluation protocols (MOS, CMOS, MUSHRA, AB, BWS; sample sizes)

### Takeaway
Use **pairwise preference (AB) or CMOS between the neutral render and the Dramatizer render** for the main claim, and MUSHRA-style multi-stimulus tests for comparing several systems. Report listener counts, samples per system, ratings per sample, the exact question and the statistical test. Use substantially more than 20 listeners, and use non-parametric tests.

### Cited Findings
- **Wester, Valentini-Botinhao & Henter, "Are we using enough listeners? No!"** (Interspeech 2015): in more than 60% of Interspeech 2014 synthesis papers, conclusions rested on fewer than 20 listeners. The paper gives a checklist for meaningful subjective evaluation. — [ISCA Archive](https://www.isca-archive.org/interspeech_2015/wester15c_interspeech.html); [PDF](https://people.kth.se/~ghe/pubs/pdf/wester2015using.pdf)
- **Kirkland, Mehta, Lameris, Henter, Székely & Gustafson, "Stuck in the MOS pit"** (SSW 2023): most 2021-22 papers under-report scale labels, increments and instructions. Changing the scale increment or the rating instructions *shifts MOS* for some systems. "Naturalness" vs "quality" wording matters (quality MOS averaged higher). — [ISCA Archive](https://www.isca-archive.org/ssw_2023/kirkland23_ssw.html)
- **[2025] Cooper, Le Maguer, Klabbers & Yamagishi, "Good practices for evaluation of synthesized speech"** (arXiv 2503.03250v2, from Dagstuhl seminar 25032). Its recommendations:
  - Follow a referenced MOS protocol (Blizzard for naturalness, ITU-T P.800 for quality). A single MOS value is not meaningful on its own.
  - AB/ABX and best-worst scaling are valid, and pairwise comparisons reveal finer differences.
  - True MUSHRA needs anchors that TTS lacks.
  - Minimum reporting: number of listeners, samples per system, ratings per sample, statistical test, exact question, scale.
  - Screen crowdworkers and control loudness.
  - Use Wilcoxon or Mann-Whitney for ordinal ratings, and binomial or Clopper-Pearson tests for preferences. Omnibus tests before post-hoc comparisons are preferable to blanket Bonferroni. Claims of equivalence need an equivalence design.

  — [arXiv 2503.03250](https://arxiv.org/html/2503.03250v2)
- **[2024/2025] "Rethinking MUSHRA"** (arXiv 2411.12719) finds that MUSHRA's reliance on matching human reference speech penalises modern TTS systems that can exceed the reference, and proposes modified variants. — [arXiv 2411.12719](https://arxiv.org/html/2411.12719v3)
- **[2025] "Towards Responsible Evaluation for Text-to-Speech"** (arXiv 2510.06927): MOS is not comparable across studies, so re-evaluate old and new systems in the same test. — [arXiv 2510.06927](https://arxiv.org/html/2510.06927v1)
- **[2026]** Park & Li: in multi-rater A/B tests, predictor-reward gains did not transfer evenly to listeners. **Best-of-8 reranking was not clearly worse perceptually than GRPO.** This supports Dramatizer's candidate-search design as a legitimate approach. — [arXiv 2608.31035](https://arxiv.org/abs/2608.31035)
- Reading list maintained by the VoiceMOS organisers — [Speech Synthesis Evaluation Reading List](https://voicemos-challenge-2023.github.io/speech-synthesis-evaluation/index.html)

### Inferences
- Suggested Dramatizer human study:
  1. **AB preference** (neutral vs Dramatizer, and Dramatizer vs a strong baseline such as random-of-N or a WER/SIM-only selection). The question should be scene-anchored: "Which delivery better fits the scene and direction?" Show the text and stage direction.
  2. A separate **naturalness MOS or CMOS** to check for a naturalness cost.
  3. A **speaker-similarity** rating against the reference (same/different scale).

  Test on held-out speakers, randomise order, and include attention checks and duplicate-pair consistency checks. Plan statistical power in advance. "Well above 20 listeners" is the only floor the sources support.
- Best-worst scaling is a good fit for ranking N candidate renders of the same line. It also enables a direct check of whether the automatic ranking agrees with human ranking (rank correlation), which is the key validity test for Dramatizer's composite score.

### Gaps
- I found no concrete source-backed sample-size formula for TTS AB tests in this session (Cooper et al. require "sufficient power" but give no number). I did not retrieve ITU-R BS.1534 (MUSHRA) or ITU-T P.800 directly. CMOS is not covered in the Cooper et al. guide.

---

## Q7. Pitfalls: reward hacking / Goodhart, neutral bias, circularity, leakage

### Takeaway
Direct evidence of reward hacking in TTS is still thin but growing. RL-for-TTS papers routinely add a second, independent evaluator or constraint, and [2026] work shows predictor-reward gains transfer unevenly to humans. The documented risks for Dramatizer are: MOS predictors that prefer formal or neutral speech, SIM that penalises emotion, WER that rewards flat over-articulation, and circularity if any optimised model or a close relative is reused as the judge.

### Cited Findings
- **RL for TTS with automatic rewards** (DRO, Interspeech 2025; diffusion-TTS RL, Interspeech 2025; multi-reward GRPO 2025; FlowTTS-GRPO 2026) uses WER, speaker similarity, UTMOS or DNSMOS as rewards. The diffusion-TTS RL paper used UTMOS as the reward and **a separate MOS network to detect overfitting to the reward model**. — [Chen et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.pdf); [DRO, Gao et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/gao25d_interspeech.pdf); [Multi-Reward GRPO arXiv 2511.21270](https://arxiv.org/html/2511.21270); [FlowTTS-GRPO arXiv 2606.23190](https://arxiv.org/pdf/2606.23190) **[2026]**
- **[2026] Park & Li, "When Does Predictor-Based RL Align with Human Perception?"** (arXiv 2608.31035, submitted to EMNLP 2026). Rewards for style, naturalness, likability and arousal each mostly improve their own metric. Gains transfer unevenly to human A/B listeners. Reward gaps do predict listener choices, but calibration varies by axis. A CER zone constraint guards against transcript drift. Best-of-N is a strong baseline. The authors recommend checking rewards with diagnostics before multi-reward post-training. — [arXiv 2608.31035](https://arxiv.org/abs/2608.31035)
- **Neutral/formal bias of MOS predictors:** NISQA and UTMOSv2 correlate negatively with human naturalness on conversational speech **[2026]**. — [arXiv 2603.01467](https://arxiv.org/html/2603.01467v1)
- **SIM penalises emotion** against neutral references **[2026]**. — [arXiv 2602.03420](https://arxiv.org/pdf/2602.03420)
- **Unseen systems and speakers** degrade predictors (VoiceMOS 2024), so a predictor scored on Dramatizer's own outputs is out-of-domain by construction. — [VoiceMOS 2024](https://homepage.citi.sinica.edu.tw/papers/yu.tsao/10634-F.pdf)
- The Whisper authors warn that tuning a normaliser to a test set overfits; the same applies to tuning Dramatizer thresholds on its evaluation set. — [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
- Both automatic evaluator classes (MOS predictors and audio-LLM judges) miss many human-perceptible errors **[2026]**. — [arXiv 2608.09930](https://arxiv.org/abs/2608.09930)

### Inferences
- **Circularity map for Dramatizer:**

  | Optimised (search side) | Held-out reporting judge |
  |---|---|
  | Whisper WER | Different ASR family (Paraformer-en / Parakeet / Conformer) plus hallucination check |
  | Resemblyzer SIM | WavLM-large-SV (Seed-TTS-eval) and/or ECAPA-TDNN; SIM vs neutral render and vs reference |
  | SQUIM objective | UTMOSv2 + DNSMOS/NISQA (label as "predicted MOS"); do not use SQUIM-subjective as the "independent" judge, because it shares a family and authors with SQUIM-objective |
  | 6 DSP prosody descriptors + Wundt novelty | emotion2vec label/shift, AutoPCP vs actor reference, SER arousal/valence, audio-LLM pairwise judge, humans |

- **Goodhart risks specific to the Wundt/novelty term:** novelty measured on the same six descriptors can be maximised by extreme tempo or pitch-range changes that humans hear as unnatural. The audio-LLM and human naturalness checks should specifically look for "caricature" outputs. Log descriptor values for the selected candidates to detect boundary-seeking.
- **Leakage:** keep speakers, scripts and scenes held out between threshold/weight tuning and reporting. Do not tune the Wundt-curve peak or guard thresholds on the human-test items. Do not choose held-out judges after seeing which one favours Dramatizer; pre-register the judge list.
- **Validity check:** report the rank correlation between Dramatizer's composite score and human preference across candidates of the same line. This is the most direct test of whether optimising the composite is worthwhile.
- Compare against a **random-of-N** and a **guard-only best-of-N** baseline. Park & Li show Best-of-N is competitive, so the claim should be about the *expressive objective*, not about search alone.

### Gaps
- I found no paper that demonstrates adversarial or "hacked" audio maximising UTMOS or SV similarity while sounding bad to humans, beyond the indirect evidence above. I did not search the adversarial-attack literature on speaker verification.
- I found no source on leakage in MOS-predictor training data overlapping with TTS evaluation sets (for example, BVCC systems overlapping with tested models).
