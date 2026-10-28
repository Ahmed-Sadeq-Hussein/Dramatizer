# Cross-speaker prosody/style transfer and few-shot delivery transfer in TTS (prior work for Dramatizer)

Scope note: about 20 search/fetch calls (Oct 2026). Items marked **[unverified-memory]** are well-known papers whose metadata I give from prior knowledge and did not re-fetch in this session. Check them before citing. Everything else was seen in search results or fetched pages. Each "Closeness" line rates the item against Dramatizer, which transfers a symbolic, position-based markup genome across voices and lines for CosyVoice 2/3 and seeds the search from the nearest speaker by embedding cosine similarity.

## Q1. Reference-based prosody transfer (Tacotron PT, GST, fine-grained, text-agnostic vs text-dependent, cross-speaker emotion)

### Takeaway
The field started (2018) with latent reference embeddings: one utterance-level vector (Skerry-Ryan) or a soft mix of learned tokens (GST). It then moved to fine-grained phoneme/word-level latents and to adversarial speaker disentanglement for cross-speaker use (CopyCat, Daft-Exprt, iEmoTTS, DiEmo-TTS). The recurring problems are source-speaker leakage and text dependence. Sigurgeirsson & King (2023) show that "prosody transfer" embeddings are largely utterance-specific (tied to the speaker and text) rather than transferable prosody. That finding is the strongest motivation for an explicit, text-position-based symbolic representation like Dramatizer's.

### Cited Findings
- **Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron.** Skerry-Ryan, Battenberg, Xiao, Wang, Stanton, Shor, Weiss, Clark, Saurous. 2018, ICML 2018. https://arxiv.org/abs/1803.09047 **[unverified-memory]**
  - What is transferred and how: a reference encoder compresses a reference utterance into a fixed-length prosody embedding that conditions Tacotron. Transfer is most faithful when the reference and target text are the same or similar (text-dependent).
  - Evaluation: subjective AB/similarity tests plus objective pitch/timing distance metrics.
  - Closeness: medium. It is the founding "copy delivery from one utterance to another" paper, but uses an opaque latent tied to the reference audio, with no search or speaker-based seeding.
- **Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis (GST).** Y. Wang, Stanton, Y. Zhang, Skerry-Ryan, Battenberg, Shor, Xiao, Ren, Jia, Saurous. 2018, ICML 2018. https://arxiv.org/abs/1803.09017 **[unverified-memory]**
  - What is transferred and how: the reference is encoded as attention weights over a bank of learned "style tokens". Weights can be transferred or set by hand, which makes them text-agnostic and more interpretable than a raw embedding. A search result describes GSTs as "soft, interpretable labels" for style ([CSTR/related search](https://arxiv.org/pdf/2107.12562)).
  - Evaluation: listening tests for style control and transfer.
  - Closeness: medium. The token weights are a low-dimensional global "genome" of style that could in principle be searched, but they are latent and global, not positional markup.
- **Fine-grained robust prosody transfer for single-speaker neural TTS.** Klimkov et al. (Amazon), 2019 (arXiv 1907.02479; Interspeech 2019 venue not confirmed in my search). [arXiv](https://arxiv.org/abs/1907.02479v1), [Amazon Science](https://amazon.science/publications/fine-grained-robust-prosody-transfer-for-single-speaker-neural-text-to-speech)
  - What is transferred and how: phoneme-level prosody latents, made more robust with a VAE. Can transfer without input text by feeding ASR output plus the reference audio.
  - Evaluation: MUSHRA and objective metrics. References from unseen speakers degraded results relative to same-speaker references, with the proposed method degrading less ([arXiv PDF](https://arxiv.org/pdf/1907.02479)).
  - Closeness: low–medium. Fine-grained and aligned to the text like Dramatizer's genes, but needs the same text and a single speaker.
- **CopyCat: Many-to-Many Fine-Grained Prosody Transfer for Neural TTS.** Karlapati, Moinet, Joly, Klimkov, Sáez-Trigueros, Drugman. Interspeech 2020. [ISCA](https://www.isca-archive.org/interspeech_2020/karlapati20_interspeech.html), [PDF](https://cdn.amazon.science/47/7c/157f6ad04b35b767b9b5083a36e8/copycat-many-to-many-fine-grained-prosody-transfer-for-neural-text-to-speech.pdf)
  - What is transferred and how: fine-grained prosody from a source speaker to a target speaker, with a bottleneck designed to cut "source speaker leakage", where the output sounds like the source voice.
  - Evaluation: listening tests for prosody similarity and speaker similarity.
  - Closeness: medium. Same question as Dramatizer (does delivery from voice A work on voice B) but text-dependent and latent.
- **CopyCat2: A Single Model for Multi-Speaker TTS and Many-to-Many Fine-Grained Prosody Transfer.** Karlapati et al. 2022 (Interspeech 2022 per my knowledge; not confirmed). https://arxiv.org/pdf/2206.13443
  - What is transferred and how: word-level prosody representations. Per the search summary, these reduce leakage and improve target-speaker similarity by roughly 31–35%. The authors also note that speaking rate interacts with perceived speaker similarity.
  - Closeness: medium. Word-level granularity is the closest latent analogue to word/position genes. The rate–identity interaction matters for Dramatizer's rate genes.
- **Daft-Exprt: Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis.** Zaïdi, Seuté, van Niekerk, Carbonneau (Ubisoft La Forge). arXiv 2108.02271, Interspeech 2022. [arXiv](https://arxiv.org/pdf/2108.02271), [ISCA](https://www.isca-archive.org/interspeech_2022/zaidi22b_interspeech.html)
  - What is transferred and how: a reference prosody encoder over energy, pitch and spectrogram conditions a FastSpeech 2-style model through FiLM layers. Adversarial speaker classification with gradient reversal reduces leakage. A higher adversarial weight preserves target identity better but transfers less perceived prosody, so there is an explicit identity-versus-transfer trade-off. Designed for inter-text (text-agnostic) transfer.
  - Evaluation: MUSHRA for transfer, MOS for naturalness, a neutral-reference low anchor, and Pearson correlation of pitch contours after trimming and resampling (because the texts differ in length).
  - Closeness: medium–high on evaluation design. Cross-speaker and cross-text, like Dramatizer, and its length-normalised pitch-contour comparison resembles Dramatizer's 0–1 relative positions. The representation is still latent.
- **Do Prosody Transfer Models Transfer Prosody?** Sigurgeirsson & King. ICASSP 2023. [arXiv](https://arxiv.org/abs/2303.04289), [Edinburgh](https://www.research.ed.ac.uk/en/publications/do-prosody-transfer-models-transfer-prosody/)
  - Finding: training with a different but prosodically related reference performs "significantly worse" than same-utterance training. The authors conclude that PT models learn an utterance-level representation highly dependent on the reference speaker and text, not transferable prosody.
  - Closeness: high as motivation. This is the key negative result Dramatizer can position against: it argues for representations that are explicitly decoupled from audio and text, though it does not test symbolic markup.
- **A Human-in-the-Loop Approach to Improving Cross-Text Prosody Transfer.** Sigurgeirsson & King (authors per my knowledge; check). 2024, arXiv 2406.06601. https://arxiv.org/pdf/2406.06601
  - What is transferred and how: humans choose among candidate references or latents for cross-text transfer, with Daft-Exprt as the baseline.
  - Finding: closeness in the latent reference embedding was not a reliable proxy for perceived prosodic similarity in the cross-text case.
  - Closeness: high. It is a search/selection loop over candidate deliveries with human judgement. It differs from Dramatizer in using latent candidates and no speaker-similarity seeding. The finding that embedding distance is a poor proxy is a caution for using any embedding as a transfer predictor.
- **ADEPT: A Dataset for Evaluating Prosody Transfer.** Torresquintero et al. 2021 (Interspeech 2021, venue from memory). https://arxiv.org/pdf/2106.08321
  - Argues the field has no clear definition or measure of successful prosody transfer. Provides a corpus of prosodically varied references limited to variations listeners can reliably distinguish.
  - Closeness: medium as an evaluation resource for whether Dramatizer's transferred deliveries still carry the intended interpretation.
- **Cross-speaker Style Transfer with Prosody Bottleneck in Neural Speech Synthesis.** 2021, arXiv 2107.12562. https://arxiv.org/pdf/2107.12562
  - Multi-speaker Transformer TTS with an explicit prosody bottleneck. Reports better cross-speaker style transfer and prosody control.
  - Closeness: medium.
- **Cross-speaker emotion transfer papers:**
  - Cross-speaker emotion disentangling and transfer for end-to-end speech synthesis (also titled "Controllable cross-speaker emotion transfer"), arXiv 2109.06733. Tacotron2 with an emotion embedding, disentangling modules to keep speaker identity out of it, and scalar strength control. https://arxiv.org/abs/2109.06733 Closeness: low–medium (categorical emotion, latent).
  - Cross-Speaker Emotion Transfer for Low-Resource TTS Using Non-Parallel Voice Conversion with Pitch-Shift Augmentation, arXiv 2204.10020. Uses VC to create emotional data for a neutral target speaker. Subjective naturalness and emotion-similarity gains. https://arxiv.org/pdf/2204.10020 Closeness: low.
  - iEmoTTS (arXiv 2206.14866). Disentangles prosody and timbre, supports zero-shot emotion transfer to unseen speakers, and has intensity control. https://arxiv.org/pdf/2206.14866 Closeness: medium.
  - Cross-speaker Emotion Transfer by Manipulating Speech Style Latents, arXiv 2303.08329. Domain-adversarial training plus cycle consistency. Emotion is moved along an SVM-hyperplane direction in latent space. https://ar5iv.labs.arxiv.org/html/2303.08329 Closeness: medium. It edits a latent direction, which is a continuous counterpart to adding genes.
  - Text-driven Emotional Style Control and Cross-speaker Style Transfer, arXiv 2207.06000 (Interspeech 2022). A text interface for style. https://ar5iv.labs.arxiv.org/html/2207.06000 Closeness: medium (the control is textual, like CosyVoice instructions).
  - DiEmo-TTS (arXiv 2505.19687, 2025). Self-supervised distillation gives speaker-independent emotion embeddings that target speaker leakage. https://arxiv.org/abs/2505.19687v1 Closeness: medium.
- **A Neural TTS System with Parallel Prosody Transfer from Unseen Speakers.** arXiv 2309.11487. Transfers prosody from an unseen speaker's parallel (same-text) recording onto trained voices while keeping target identity, per listening tests. https://arxiv.com/abs/2309.11487 Closeness: medium (parallel text only).

### Inferences
- Across 2018–2025, every latent method fights the same coupling: the embedding carries the source speaker and source text. Dramatizer's symbolic genome is decoupled from source audio by construction, so leakage of source timbre cannot happen. The open question becomes whether the TTS renders the same markup consistently across voices, which no paper above tests.
- Daft-Exprt's trade-off (stronger identity preservation means weaker transfer) suggests a hypothesis for Dramatizer: markup effects may be attenuated on voices whose reference prompt has very different baseline prosody.

### Gaps
- I did not find any paper that evaluates transfer of a symbolic delivery (markup) across speakers using the protocols above (MUSHRA prosody similarity or pitch-contour correlation).
- The exact venue for Klimkov 2019 and CopyCat2 was not confirmed.

## Q2. Zero-shot TTS with separate timbre and prosody/style/emotion prompts (2023–2026)

### Takeaway
Modern zero-shot systems factor the speaker (timbre) prompt from the prosody/style/emotion prompt: NaturalSpeech 3 (factorised codec), Mega-TTS 2 (P-LLM prosody prompts with multi-speaker prosody interpolation), MegaTTS 3 (accent-intensity CFG), Vevo (style vs timbre references), IndexTTS2 (an emotion prompt that can come from a different speaker, plus text-described emotion), and Step-Audio-EditX (RL-based emotion/paralinguistic editing). These systems make cross-speaker delivery transfer a first-class use case, but the "delivery" is always an audio prompt or a latent. Only CosyVoice-style instruct text and inline tags (and Step-Audio-EditX's paralinguistic tags) are symbolic, and Dramatizer builds on exactly those.

### Cited Findings
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models.** Ju, Y. Wang, Shen, Tan, … Zhao (Microsoft et al.). arXiv 2403.03100, 2024. The fetched page lists no venue (ICML 2024 per my memory, unverified). https://arxiv.org/abs/2403.03100
  - Factorised vector quantisation splits speech into content, prosody, timbre and acoustic-detail subspaces. A factorised diffusion model generates each one from its own prompt.
  - Closeness: medium. The prosody prompt is audio and latent, so it is a cross-speaker delivery transfer channel, but not editable markup.
- **Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis** (earlier title "…with Arbitrary Length Speech Prompts"). Jiang et al. ICLR 2024. https://arxiv.org/abs/2307.07218v4, https://iclr.cc/virtual/2024/poster/17876
  - A multi-reference timbre encoder plus a prosody latent language model (P-LLM). Prosody interpolation uses prosody prompts from several speakers while keeping the target timbre.
  - Evaluation: LibriSpeech test-clean SIM and naturalness. Longer prompts help, and it outperforms fine-tuning from 10 s to 5 min of data.
  - Closeness: medium–high. It explicitly mixes prosody from other speakers onto a target voice, which is conceptually close to genome transfer, but the mixing is latent with no speaker-similarity selection.
- **MegaTTS 3: Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis.** Jiang et al. 2025, arXiv 2502.18924. https://arxiv.org/abs/2502.18924
  - Multi-condition classifier-free guidance gives accent-intensity control without paired data. Reports GPE 0.31, VDE 0.29, FFE 0.34 on LibriSpeech test-clean (Table 4).
  - Closeness: low–medium.
- **Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement.** Xueyao Zhang et al. ICLR 2025. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9328208f88ec69420031647e6ff97727-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2502.07243v1)
  - An AR transformer produces content-style tokens guided by a style reference, then flow matching produces acoustics guided by a separate timbre reference. Disentanglement comes from a VQ-VAE over HuBERT with codebook size as the bottleneck. Trained self-supervised on 60K h of audiobooks. Matches or beats prior methods on accent and emotion conversion.
  - Closeness: medium. Style and timbre references from different speakers is cross-speaker delivery transfer, but through audio.
- **IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot TTS.** Siyi Zhou et al. (bilibili). arXiv 2506.21619, v1 June 2025, v2 Sept 2025. https://arxiv.org/abs/2506.21619v2
  - A separate emotion/style prompt "can originate from a different speaker than the timbre prompt". A GRL removes emotion-irrelevant information. A Qwen3-fine-tuned text-to-emotion module enables natural-language emotion instructions. Also offers explicit duration control.
  - Closeness: medium–high. It is the clearest current system for "deliver voice B with voice A's emotion", and its text-instruction path is semi-symbolic. It differs from Dramatizer in having no positional markup and no search.
- **CosyVoice 2** (Du et al., Alibaba, arXiv 2412.10117, Dec 2024; authors and ID from memory plus the HF README).
  - The official HF README shows an instruct mode and inline tags `<laughter></laughter><strong></strong>[laughter][breath]`. The tokenizer source (`cosyvoice/tokenizer/tokenizer.py`) is the authoritative tag list. https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
  - Closeness: high (substrate). This markup vocabulary is Dramatizer's gene alphabet.
- **CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training.** Du, Gao, Y. Wang, et al. 2025, arXiv 2505.17589. https://arxiv.org/abs/2505.17589
  - Its abstract covers a multi-task tokenizer (ASR, SER, audio-event detection, speaker analysis), a differentiable reward model for post-training, 1M hours, and 1.5B parameters. The abstract does not describe instruct tags; check the full paper or repo for its tag set.
  - Closeness: high (substrate).
- **Step-Audio-EditX technical report.** StepFun. arXiv 2511.03601, Nov 2025. https://arxiv.org/abs/2511.03601v1
  - LLM-based audio model trained with RL for iterative editing of emotion, speaking style and paralinguistics (breath, laughter, filled pauses, using NVSpeech-derived tags), with zero-shot TTS. Claims to outperform MiniMax-2.6-hd and Doubao-Seed-TTS-2.0 on emotion editing (vendor claim).
  - Closeness: medium–high. It uses iterative edits with paralinguistic tags, but the editing is one-shot per utterance with no cross-voice reuse.
- **ParaSpeechCaps** (EMNLP 2025, "Scaling Rich Style-Prompted TTS Datasets"). 59 style tags; fine-tuned Parler-TTS. https://underline.io/lecture/129932-scaling-rich-style-prompted-text-to-speech-datasets
  - Closeness: low–medium. Global tag vocabulary, not positional.

### Inferences
- Factorised prompting means a "delivery" can in principle come from any speaker. Dramatizer's contribution is to make that delivery a portable, editable, searchable symbolic object instead of a reference clip.
- IndexTTS2 and Mega-TTS 2 both report that cross-speaker style prompts work, which is prior evidence that delivery is partially speaker-independent at the model level.

### Gaps
- None of these papers report how stable a given textual instruction or tag sequence is across many voices. Per-voice variance of tag effects is unmeasured in the literature I found.
- The full CosyVoice 3 instruct and tag specification was not verified.

## Q3. Symbolic or interpretable prosody representations that transfer across speakers

### Takeaway
Symbolic prosody (ToBI, "ToBI Lite", prominence/emphasis labels, SSML) has a long history as a control input, and recent work derives word-level emphasis pseudo-labels automatically (wavelet prosody toolkit, SSL emphasis classifiers) to drive FastSpeech-style models. However, I found no paper that combines ToBI-style or markup labels with an explicit cross-speaker transfer experiment, so Dramatizer's position here appears novel. The nearest interpretable-and-transferable work is speaker-independent prosodic clustering (phoneme-level discrete prosody classes).

### Cited Findings
- ToBI in TTS: a Romanian RSS-ToBI corpus compared HMM synthesis with and without ToBI labels ([Boroș 2014, CSTR](https://www.cstr.ed.ac.uk/downloads/publications/2014/boros-2014.pdf)). A simplified "ToBI Lite" was used for prosodically enriched unit selection ([Wightman, Speech Prosody 2002](https://mail.sprosig.org/sp2002/pdf/wightman.pdf)). The search found no paper pairing ToBI labels with cross-speaker transfer. Closeness: medium (symbolic, but not tested for transfer).
- **Improved Prosodic Clustering for Multispeaker and Speaker-independent Phoneme-level Prosody Control.** arXiv 2111.10168. https://arxiv.org/pdf/2111.10168 Discrete, speaker-independent phoneme-level prosody clusters for control. Closeness: medium–high (discrete codes meant to be speaker-independent; latent-derived rather than human-readable markup).
- **EE-TTS: Emphatic Expressive TTS with Linguistic Information** (arXiv 2305.12107). Wavelet Prosody Toolkit pseudo-labels give binary per-character emphasis. https://arxiv.org/pdf/2305.12107 Closeness: medium.
- **EME-TTS** (arXiv 2507.12015; Interspeech 2025). An SSL-based EmphaClass classifier gives word-level emphasis pseudo-labels on ESD; emphasis is linked to emotion. https://arxiv.org/pdf/2507.12015 Closeness: medium (emphasis genes are symbolic; no cross-speaker reuse).
- **Seshadri et al. (Apple)**, arXiv 2110.03012. Word-level emphasis in FastSpeech 2 by changing latent feature values for chosen words at inference. https://arxiv.org/pdf/2110.03012 Closeness: medium.
- **Bauer et al., SSW 2025.** Controllable-emphasis TTS trained with word-wise emphasis scores from detectors; compares label sources. https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2025_BauerZMD_EmphasisControl_SSW_ePrint.pdf Closeness: medium.
- **"BERT, can HE predict contrastive focus?"** (arXiv 2207.01718). An LM predicts prominence placement, then TTS realises it. https://ar5iv.labs.arxiv.org/html/2207.01718 Closeness: medium. Predicting where emphasis goes from text alone parallels a position-based gene.
- **Supervised and Unsupervised Approaches for Controlling Narrow Lexical Focus in Seq2Seq Speech Synthesis** (arXiv 2101.09940). https://www.alphaxiv.org/abs/2101.09940 Closeness: medium.
- **Improving French Synthetic Speech Quality via SSML Prosody Control** (arXiv 2508.17494, 2025). Generates SSML prosody markup automatically. Also summarises Stephenson et al. 2022 (BERT predicting 3-level prominence, F1 = 0.588, controlling FastSpeech 2). https://arxiv.org/pdf/2508.17494 Closeness: medium–high (automatic markup for delivery; single voice, no transfer study).
- **Automatic SSML tag assignment patent.** US12198676 ("Assigning SSML tags to an audio corpus"). https://patents.google.com/patent/US12198676 Closeness: low–medium.

### Inferences
- Symbolic labels are speaker-independent by definition (a pitch accent or `<strong>` on word k means the same for any voice), but how they are realised acoustically varies by speaker and model. Dramatizer's experiment directly measures this realisation variance, which the symbolic-prosody literature has not measured cross-speaker for modern zero-shot LLM-TTS.
- Dramatizer's choice of relative position (0–1) instead of word identity makes the representation text-agnostic as well. I found no prior symbolic system that re-anchors markup onto a different sentence by relative position. The closest analogue is Daft-Exprt's trimming and resampling of pitch contours for cross-text comparison.

### Gaps
- No paper found on transferring ToBI/SSML annotations from one speaker's recording to another speaker's synthesis with a perceptual evaluation.
- No paper found on position-normalised (relative-offset) markup re-anchoring across sentences.

## Q4. Expressive S2ST and voice conversion as evidence of prosody transferability

### Takeaway
SeamlessExpressive shows that prosody (rate, pauses, emotion and style via PRETSSEL) can carry across languages and through a unit bottleneck, with measurable but imperfect preservation and quality trade-offs. Emphasis-aware S2ST (StressTransfer, 2025) moves toward symbolic, word-level prosody transfer across texts. VC systems such as Vevo separate style from timbre, reinforcing that delivery can be moved between voices.

### Cited Findings
- **Seamless: Multilingual Expressive and Streaming Speech Translation** (Seamless Communication / Meta, arXiv 2312.05187, Dec 2023). https://arxiv.org/pdf/2312.05187, https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation
  - SeamlessExpressive = Prosody UnitY2 (controls speech rate and pauses) + PRETSSEL (unit-to-speech with cross-lingual expressivity preservation).
  - Evaluation: AutoPCP (a prosody-similarity model), rate as the Spearman correlation of syllable rate, a pause-alignment score (stopes), and human PCP on a 1–4 scale for overall expressive identity, emotion and rhythm (mDRAL, mExpresso). PRETSSEL improved all PCP dimensions but lowered some MOS subscores (clarity, sound quality).
  - Closeness: medium. Cross-text (and cross-language) transfer of rate and pauses, two of Dramatizer's gene types, with metrics usable for Dramatizer (AutoPCP, rate correlation, pause alignment).
- **StressTransfer: Stress-Aware Speech-to-Speech Translation with Emphasis Preservation** (arXiv 2510.13194, Oct 2025). https://arxiv.org/pdf/2510.13194
  - Preserves word-level emphasis across translation. Reports SeamlessExpressive at 35.06 ASR-BLEU / 2.63 UTMOS vs SeamlessM4T-v2-Large at 41.39 / 3.52 on its test set.
  - Closeness: medium–high. Emphasis is mapped onto different text (the translation), which is an explicit, word-aligned delivery transfer.
- **DINO-PRETSSEL** (arXiv 2406.02733). Noise-robust PRETSSEL with a slight drop in prosody preservation on clean input. https://arxiv.org/pdf/2406.02733 Closeness: low.
- **Vevo** (see Q2). Vevo-Style does accent and emotion conversion and Vevo-Timbre does style-preserving VC; the variants are separable. https://huggingface.co/amphion/Vevo Closeness: medium.

### Inferences
- S2ST metrics (AutoPCP, rate Spearman, pause alignment) are ready-made, text-agnostic transfer metrics Dramatizer could reuse to quantify "did the delivery survive the move to a new voice and line".

### Gaps
- I did not verify AutoPCP's publication details separately (it is described inside the Seamless paper).

## Q5. Meta-learning / warm-starting optimisation from similar past problems (key references)

### Takeaway
Dramatizer's nearest-speaker seeding is a direct instance of a classic idea: initialise a search with the best solutions from the most similar previous tasks, where similarity is measured by meta-features. The canonical references are case-injected GAs (Louis & McDonnell 2004), MI-SMBO (Feurer et al. 2015), and WS-CMA-ES (Nomura et al. 2021). WS-CMA-ES explicitly reports that the gain correlates with source–target task similarity, which is exactly Dramatizer's hypothesis.

### Cited Findings
- **Learning with Case-Injected Genetic Algorithms (CIGAR).** Sushil J. Louis & John McDonnell. IEEE Trans. Evolutionary Computation 8(4):316–328, 2004. [DTIC PDF](https://apps.dtic.mil/sti/pdfs/ADA446085.pdf), [UNR PDF](https://www.cse.unr.edu/~sushil/class/gas/papers/cigar.pdf)
  - Keeps a case base of past GA solutions and injects solutions from similar past problems into the initial (and later) populations instead of starting at random. Speeds learning on related problems. Applied in GECCO 2004 strategic gaming and other domains ([GECCO 2004](https://gpbib.pmacs.upenn.edu/gecco2004/prof92.html)).
  - Closeness: high (method). This is essentially Dramatizer's seeding scheme, with speaker-embedding cosine similarity as the problem-similarity metric.
- **Initializing Bayesian Hyperparameter Optimization via Meta-Learning (MI-SMBO).** Feurer, Springenberg, Hutter. AAAI 2015, pp. 1128–1135. https://ojs.aaai.org/index.php/AAAI/article/view/9354
  - Seeds SMBO (Spearmint, SMAC) with the best configurations from the datasets nearest in meta-feature space. Tested on 57 datasets: mild gains for low-dimensional HPO, substantial gains for CASH.
  - Closeness: high (method). Nearest-neighbour warm start by meta-features, the same as nearest speaker by embedding.
- **Warm Starting CMA-ES for Hyperparameter Optimization (WS-CMA-ES).** Nomura, Watanabe, Akimoto, Ozaki, Onishi. AAAI 2021, 35:9188–9196. https://ojs.aaai.org/index.php/AAAI/article/view/17109, https://arxiv.org/abs/2012.06932
  - Initialises the CMA-ES mean and covariance by fitting a promising distribution from source-task trials. Outperforms CMA-ES and BO with or without priors, and performance correlates with task similarity. Implemented in Optuna (`source_trials`).
  - Closeness: high (method). Suggests a refinement for Dramatizer: build a seed distribution from the top-k genomes of the nearest voices, not only the single winner.
- **Scalable Meta-Learning for Bayesian Optimization** (arXiv 1802.02219, 2018). https://arxiv.org/abs/1802.02219v1 Closeness: medium.
- **[unverified-memory]** Gupta, Ong & Feng, "Multifactorial Evolution: Toward Evolutionary Multitasking," IEEE TEVC 2016 (transfer across concurrently optimised tasks). URL not fetched; verify before citing.

### Inferences
- The literature predicts that the benefit of seeding depends on how well the similarity metric tracks solution transferability. Dramatizer should therefore report the correlation between speaker cosine similarity and seed fitness on the target (as WS-CMA-ES does with task similarity) and include random-seed and random-other-speaker-seed baselines.
- Negative transfer (seeding is worse than random) is a known risk when the tasks are dissimilar, so it should be measured.

### Gaps
- I did not fetch the GECCO-specific warm-start or transfer-optimisation surveys. The listed references are the key ones only.

## Q6. Has anyone transferred a discovered markup/delivery between voices using speaker similarity to choose the source?

### Takeaway
**None found.** I found no paper that (a) discovers a markup or paralinguistic-tag delivery by search and then (b) reuses it on another voice, choosing the source voice by speaker-embedding similarity. The closest adjacent work selects training or source speakers by similarity for data or prosody (not reusing a searched symbolic solution), or does iterative paralinguistic editing per utterance with no cross-voice reuse.

### Cited Findings
- **Guennec et al., SSW 2023.** Trained TTS on only the training speakers closest to (or farthest from) each target, out of a pool of 14. Speaker selection measurably changed target similarity, more for speaker-encoding models than for adaptation. https://www.isca-archive.org/ssw_2023/guennec23_ssw.html Closeness: medium. Similarity-based source selection, but for training data, not delivery.
- **"Cloning one's voice using very limited data in the wild"** (Hieratron, arXiv 2110.03347). Chose the source speaker whose prosody was closest to the target's: speaker similarity rose by about 0.05 and quality fell by about 0.18, which is feasible only with enough multi-speaker data. https://arxiv.org/pdf/2110.03347 Closeness: medium–high. Prosody-similarity source selection is conceptually close, but the transfer is latent.
- **Step-Audio-EditX** (arXiv 2511.03601). Iterative tag-based editing of emotion and paralinguistics, with no cross-voice reuse or similarity-based seeding. https://arxiv.org/abs/2511.03601v1
- **Human-in-the-loop cross-text prosody transfer** (arXiv 2406.06601). Selects among candidate deliveries but finds latent distance a poor proxy for perceived similarity. https://arxiv.org/pdf/2406.06601
- Searches for LLM- or optimisation-driven SSML/paralinguistic-tag search with cross-speaker reuse returned only patents, benchmarks and vendor documentation, with no such study ([search results incl. USPTO 12198676](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12198676); [MTalk-Bench](https://arxiv.org/pdf/2508.18240)).

### Inferences
- Dramatizer's combination is not reported in the literature I found: a symbolic, position-anchored delivery genome, evolutionary search on a zero-shot LLM-TTS, and speaker-embedding nearest-neighbour seeding following the CIGAR, MI-SMBO and WS-CMA-ES pattern.
- The nearest framings are "CIGAR / MI-SMBO applied to TTS delivery" (method side) and "symbolic rather than latent prosody transfer, answering Sigurgeirsson & King 2023" (speech side).
- A caveat from Hieratron and the human-in-the-loop work: speaker or embedding similarity may not predict delivery transferability, so the seeding hypothesis can fail and should be tested against random seeding.

### Gaps
- My searches were limited (about 20 calls). A dedicated search of Interspeech/ICASSP 2025–2026 and GECCO 2025–2026 proceedings for "prompt/markup search for TTS" could still turn up a close match. Absence here is "not found", not "proven absent".
- I did not check LLM-agent TTS directing systems (e.g. LLM-planned audiobook narration) for cross-voice reuse of plans.
