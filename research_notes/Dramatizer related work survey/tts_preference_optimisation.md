# Reward- and Preference-Based Optimisation of TTS Models (prior work for Dramatizer)

Scope note: researched 2026-10-27 with web search plus arXiv abstract/HTML fetches. Items marked **[title/snippet only]** were seen only as search-result titles or third-party snippets; their details are unverified. Items marked **[2025]** / **[2026]** carry the arXiv submission month. "Closeness" is judged against Dramatizer: a frozen zero-shot voice-cloning TTS (CosyVoice 2/3), an inference-time evolutionary search over paralinguistic markup in the text prompt, guard rewards (WER, SIM, validity), a Wundt-curve expressiveness/novelty term, and a human pick from a diverse shortlist.

## Q1. CosyVoice 3 DiffRO and related CosyVoice RL work: rewards and gains

### Takeaway
DiffRO (Alibaba Tongyi, 2025) skips the RL loop. It back-propagates a reward from an ASR-like Token2Text model (and optionally SER/MOS/AED heads) through Gumbel-Softmax-sampled speech tokens, with a KL penalty. It cuts WER/CER by roughly 12-35% relative on seed-tts-eval and much more on some low-resource languages. It also slightly lowers speaker similarity, which the authors call "hacking". An emotion variant (DiffRO-EMO) raises emotion accuracy but hurts pronunciation. Later Alibaba/CosyVoice-family work (RRPO, FlowTTS-GRPO) adds emotion rewards and RL on the flow-matching stage.

### Cited Findings
- **CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training.** Du et al. (Alibaba), arXiv 2505.17589, May 2025 **[2025]**. DiffRO optimises the speech tokens rather than the synthesised audio, which avoids running flow-matching and the vocoder in the reward loop. The main reward is the posterior probability of the target text under an ASR-trained Token2Text model. Gumbel-Softmax sampling makes it differentiable, and a token-level KL term (weight β) anchors the model to the reference. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- CosyVoice 3 multi-task reward (MTR): adds SER, MOS-prediction and AED (audio event detection) heads to control attributes. DiffRO-EMO uses the SER reward. No speaker-similarity reward is used. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- Gains (Table 4, SEED-TTS-Eval). CosyVoice3-0.5B: test-zh CER 1.16→0.75, test-en WER 2.02→1.76. CosyVoice3-1.5B: 1.12→0.71 and 2.21→1.45. The paper states RL gives 12-35% relative content-consistency gains and narrows the 0.5B/1.5B gap. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- Multilingual gains (Table 5, 0.5B): Korean WER 12.8→4.02, Italian 6.68→2.94. Cross-lingual ja→ko WER 41.0→16.9. CosyVoice 2 + DiffRO also improves (hard-zh 12.58→10.66). — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- Emotion (Table 9, 1.5B, DiffRO-EMO): "happy" accuracy 0.64→0.98, "angry" 0.48→0.68, "sad" 0.44→0.50. The authors note the emotion gain "can adversely affect pronunciation" but give no WER figure. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- Failure modes:
  - RL slightly reduces speaker similarity on most sets. The paper calls this a persistent "hacking" problem: the model over-focuses on the target reward.
  - Gains are small on hard cases (tongue twisters, rare words): 1.5B hard-zh 9.77→9.06.
  - Instruct-following under DiffRO is not reported.
  - [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- **Differentiable Reward Optimization for LLM based TTS system.** Gao et al., arXiv 2507.05911, Interspeech 2025 **[2025]**. This is the standalone DiffRO paper. It reports SOTA WER on seed-tts-eval and describes the multi-task reward (SER, MOS, AED). — [arXiv](https://arxiv.org/abs/2507.05911v1); [ISCA PDF](https://www.isca-archive.org/interspeech_2025/gao25d_interspeech.pdf)
- **RRPO: Robust Reward Policy Optimization for LLM-based Emotional TTS.** Cong Wang, Changfeng Gao, Yang Xiang, Zhihao Du, Keyu An, Han Zhao, Qian Chen, Xiangang Li, Yingming Gao, Ya Li. arXiv 2512.04552, Dec 2025, ICASSP 2026 **[2025]**.
  - Reward hacking found: a vanilla emotion reward model is exploited by the policy, which generates "acoustic artifacts to achieve spurious rewards" and so degrades perceptual quality.
  - Fix: hybrid regularisation that makes the reward model robust.
  - Results: better emotional expressiveness and naturalness than the baselines.
  - The demo is named "RRPO-CosyVoice", so the base is probably CosyVoice, but the abstract does not state this (uncertain).
  - [arXiv](https://arxiv.org/abs/2512.04552)
- **FlowTTS-GRPO: Online RL with Multi-Objective Reward Optimization for Flow-Matching Based TTS.** Haoxu Wang, Biao Tian, Weiqin Li, Xiang Lv, Han Zhao, Xiangang Li. arXiv 2606.23190, Jun 2026, Interspeech 2026 **[2026]**.
  - Method: GRPO applied directly to the flow-matching stage of CosyVoice 3.0 and F5-TTS, converting ODE trajectories to SDE paths.
  - Rewards: a weighted multi-reward combination, which converges faster than a probabilistic scheme. The individual rewards are not named in the abstract.
  - Results: gains in SIM and perceptual quality.
  - [arXiv](https://arxiv.org/abs/2606.23190)
- I found no evidence that CosyVoice 2 itself (the original report) used DPO/RL post-training. The CosyVoice 3 paper's only own RL is DiffRO. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)

### Inferences
- DiffRO's emotion head shows that a frozen-SER reward can push CosyVoice towards stronger categorical emotion. The pronunciation cost and the SIM loss resemble what Dramatizer's guards (WER, SIM) are designed to catch. A reward-tuned CosyVoice could serve as a drop-in better base for Dramatizer's search.
- Because DiffRO operates on LLM speech tokens, it would change the distribution Dramatizer searches over. Markup sensitivity, such as how strongly [breath] or emphasis tags are realised, could increase or decrease. This has not been studied.

### Gaps
- DiffRO numbers for instruct/markup-following (e.g., `<strong>`, `[laughter]` realisation) are not reported.
- I did not verify the full author list of 2507.05911.

## Q2. DPO / preference alignment and GRPO for TTS (catalogue)

### Takeaway
Since 2024 the field has moved from offline DPO on preference pairs to online GRPO with automatic rewards, which is now standard in LLM-TTS technical reports. The work splits as follows:
- **Offline DPO on pairs:** SpeechAlign, Tian et al., Emo-DPO, Koel-TTS, FPO.
- **Uncertainty-weighted or reward-model-free variants:** UNO, RIO.
- **Diffusion/flow RL:** DLPO, F5R-TTS, FlowTTS-GRPO, GROW.
- **Online GRPO on LLM-TTS:** DMOSpeech 2, Llasa multi-reward GRPO, GLM-TTS.
- **Production REINFORCE:** Seed-TTS.

Nearly all use WER/CER plus SIM, with UTMOS/MOS, SER and LLM- or LALM-judged prosody/style as add-ons.

### Cited Findings
Catalogue (title — authors — year/venue — URL — reward — method — finding — closeness):

- **SpeechAlign: Aligning Speech Generation to Human Preferences** — Zhang et al. — 2024, arXiv 2404.05600 (also on OpenReview; NeurIPS 2024 per my recollection, unverified) — [arXiv](https://arxiv.org/pdf/2404.05600), [OpenReview](https://openreview.net/pdf?id=SKCbZR8Pyd)
  - Reward: none explicit. Ground-truth codec tokens are "preferred" and model outputs "dispreferred".
  - Method: iterative self-improvement with DPO on a codec LM.
  - Finding: the first DPO approach for speech generation, as described by later papers ([FPO paper](https://arxiv.org/html/2502.02950v1); [RIO paper](https://arxiv.org/pdf/2407.02243)).
  - Closeness: low. It is training-time and uses no expressiveness term.
- **Enhancing Zero-shot TTS Synthesis with Human Feedback (UNO, uncertainty-aware optimisation)** — Chen Chen, Yuchen Hu, Wen Wu, Helin Wang, Eng Siong Chng, Chao Zhang — 2024, arXiv 2406.00654 (preprint) — [arXiv](https://arxiv.org/abs/2406.00654)
  - Reward: human subjective annotations, with no reward model or paired preference data. Per secondary sources it uses unpaired preferences and replaces DPO's β with normalised inverse uncertainty ([RIO/FPO citing summaries](https://arxiv.org/html/2502.02950v1)).
  - Method: a sampling-annotating-learning loop.
  - Finding: improves MOS, WER and SIM, and adapts to a desired speaking style in emotional TTS.
  - Closeness: medium. Its sample-human-annotate loop resembles Dramatizer's human-rated shortlist, but UNO updates weights.
- **Robust Zero-Shot TTS Synthesis with Reverse Inference Optimization (RIO)** — Hu et al. (I did not verify the author list) — 2024, arXiv 2407.02243 — [arXiv](https://arxiv.org/pdf/2407.02243)
  - Reward: a pretrained MOS estimator, plus a Bayesian "reverse inference" criterion: whether the output can serve as a prompt to regenerate the original.
  - Method: selects positive and negative exemplars from the model's own outputs for preference learning, with no human annotation.
  - Finding: improved robustness and less training-inference mismatch (from summaries).
  - Closeness: low-medium. The self-consistency verifier idea could serve as a Dramatizer guard.
- **Preference Alignment Improves Language Model-Based TTS** — Tian et al. — 2024, arXiv 2409.12403 (full author list not verified) — [arXiv](https://arxiv.org/html/2409.12403v1)
  - Reward: automatic metrics (WER, SIM, proxy MOS) used to build DPO pairs.
  - Method: an empirical DPO study on a 1.15B LM-TTS trained on 55k hours.
  - Finding: consistent gains in intelligibility, SIM and proxy subjective scores, with SIM and proxy scores surpassing human speech in some tests. Gains generalise to unseen metrics and human ratings.
  - Closeness: low.
- **Emo-DPO: Controllable Emotional Speech Synthesis through DPO** — Xiaoxue Gao, Chen Zhang, Yiming Chen, Huayun Zhang, Nancy F. Chen — 2024, arXiv 2409.10157 (ICASSP 2025 per later citations, unverified) — [arXiv](https://arxiv.org/html/2409.10157v1)
  - Reward: preference pairs of target-emotion speech vs. other-emotion speech.
  - Method: DPO on an LLM-TTS whose speech tokens feed a frozen flow-matching model and a frozen vocoder (CosyVoice-like pipeline).
  - Finding: outperforms baselines on emotional expressiveness.
  - Closeness: medium. It targets emotion on a CosyVoice-like stack, but by tuning weights.
- **Emo-LiPO (listwise preference for fine-grained emotion intensity)** — includes Xiaoxue Gao — IJCAI 2026 **[2026]** **[title/snippet only]** — [IJCAI PDF](https://www.ijcai.org/proceedings/2026/0652.pdf)
- **DLPO: Diffusion Model Loss-Guided RL for Fine-Tuning TTS Diffusion Models** — Jingyi Chen et al. — arXiv 2405.14632 (May 2024). Also "Fine-Tuning TTS Diffusion Models Using RL with Human Feedback", Interspeech 2025, arXiv 2508.03123 **[2025]** — [arXiv](https://arxiv.org/pdf/2405.14632), [Interspeech version](https://arxiv.org/pdf/2508.03123)
  - Reward: UTMOS (a MOS predictor), with the original diffusion loss added as a penalty against drift.
  - Base model: WaveGrad 2.
  - Finding: UTMOS 3.65, NISQA 4.02. Listeners preferred DLPO audio 67% of the time. The paper compares against RWR/DDPO-style baselines, per my understanding (unverified).
  - Closeness: low.
- **Emotion-Aligned Generation in Diffusion TTS via Preference-Guided Optimization** — arXiv 2509.25416 **[2025]** **[title/snippet only]** — [arXiv](https://arxiv.org/html/2509.25416v2)
- **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** — Anastassiou et al. (ByteDance, 46 authors) — 2024, arXiv 2406.02430 — [arXiv HTML](https://arxiv.org/html/2406.02430)
  - Algorithms: compared PPO/REINFORCE (external reward models) against DPO. Both were effective; external rewards give clearer attribute control. The final models use REINFORCE.
  - Rewards: SIM + WER (Seed-TTS_RL-SIM-WER), and SER accuracy (Seed-TTS_RL-SER).
  - Results: EN WER 2.249→1.945; hard-set WER 7.585→6.423; SIM +0.004-0.006; CMOS +0.14.
  - Emotion accuracy: angry 0.46→0.91, happy 0.44→0.80, sad 0.53→0.78, surprise 0.13→0.82.
  - Reported reward hacking: to reduce WER, the model produced "slower, overly clear speech", which hurt naturalness.
  - Closeness: medium. It is the canonical evidence that emotion-classifier rewards work and that WER rewards push towards over-articulated, neutral delivery.
- **Koel-TTS: Enhancing LLM-based Speech Generation with Preference Alignment and Classifier-Free Guidance** — NVIDIA (authors not verified) — arXiv 2502.05236, EMNLP 2025 **[2025]** — [arXiv](https://arxiv.org/html/2502.05236v2), [ACL Anthology](https://preview.aclanthology.org/setup/2025.emnlp-main.1076)
  - Reward: ASR and speaker-verification-derived preferences (per my understanding, unverified).
  - Method: preference alignment combined with CFG.
  - Closeness: low.
- **Fine-grained Preference Optimization Improves Zero-shot TTS (FPO)** — arXiv 2502.02950 **[2025]** — [arXiv](https://arxiv.org/html/2502.02950v1)
  - Method: preference optimisation localised to problematic segments rather than whole utterances (from title and abstract context; details unverified).
  - Closeness: low-medium. Segment-level credit assignment is analogous to Dramatizer's per-span markup genes.
- **Advancing Zero-shot TTS Intelligibility across Diverse Domains via Preference Alignment** — arXiv 2505.04113 per my recollection (unverified) **[2025]** **[title/snippet only]** — [mirror](https://sinoxiv.napstic.cn/article/24637542)
- **F5R-TTS: Improving Flow-Matching based TTS with GRPO** — Sun, Xiao et al. (FrontierLabs) — arXiv 2504.02407 **[2025]** — [arXiv](https://arxiv.org/pdf/2504.02407), [code](https://github.com/FrontierLabs/F5R-TTS)
  - Reward: WER (ASR) + SIM (speaker verification).
  - Method: reformulates the deterministic flow-matching output as a probabilistic (Gaussian) policy so GRPO can be applied.
  - Finding: 29.5% relative WER reduction and 4.6% relative SIM gain in zero-shot cloning.
  - Closeness: low.
- **DMOSpeech 2: RL for Duration Prediction in Metric-Optimized Speech Synthesis** — Yinghao Aaron Li et al. — arXiv 2507.14988, AAAI 2026 **[2025]** — [arXiv](https://arxiv.org/abs/2507.14988), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/40450/44411)
  - Reward: SIM + WER.
  - Method: GRPO on the duration predictor only (group size 16, 1.5k steps).
  - Finding: EN WER 1.752 vs. 1.947 for F5-TTS and 3.750 for DMOSpeech without RL.
  - Closeness: medium. Speaking rate and duration are one of Dramatizer's gene types; DMOSpeech 2 learns a duration policy instead.
- **Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis** — Ye et al. (HKUST et al.) — arXiv 2502.04128 **[2025]**. See Q4 for its test-time verifier search. — [arXiv](https://arxiv.org/html/2502.04128)
- **Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale** — Yicheng Zhong, Peiji Yang, Zhisheng Wang — arXiv 2511.21270, Nov 2025 **[2025]** — [arXiv](https://arxiv.org/abs/2511.21270)
  - Rewards: WER/CER + SIM, a rule-based length penalty, an entropy regulariser for decoding stability, and an LLM-annotated prosody/pause alignment reward (an external reasoning LLM proposes plausible pause structures in context).
  - Base model: Llasa, per a third-party summary.
  - Finding: better prosodic stability, SIM and naturalness. Best zh CER 1.10.
  - Closeness: **high-medium**. Its LLM-proposed pause structures are close to Dramatizer's pause genes, but they are used as a training reward rather than searched at inference time. — [snippet](https://arxiv.org/html/2511.21270)
- **GLM-TTS: Controllable & Emotion-Expressive Zero-shot TTS with Multi-Reward RL** — Zhipu / zai-org — open-sourced 2025-12-11, tech report arXiv 2512.14291 **[2025]** — [GitHub](https://github.com/zai-org/GLM-TTS), [arXiv](https://arxiv.org/abs/2512.14291)
  - Rewards: SIM, CER, emotion and **laughter** rewards.
  - Method: GRPO on the LLM stage, with token-level reward allocation and a distributed reward server.
  - Finding: zh CER 1.03→0.89, SIM 76.1→76.4. Expressiveness gains are claimed qualitatively only. RL weights are listed as "coming soon".
  - Closeness: medium-high. A laughter reward directly parallels Dramatizer's [laughter] genes.
- **GROW: Group-Relative Advantage-Weighted On-Policy RL of Autoregressive-Diffusion TTS** — arXiv 2608.03215 **[2026]** — [arXiv](https://arxiv.org/html/2608.03215)
  - Rewards: intelligibility + SIM.
  - Base model: DiTAR.
  - Finding: average WER 2.016→1.558, SIM 0.676→0.715, UTMOS kept. Rewarding SIM alone also lowered WER (from snippet).
  - Closeness: low.
- **Beyond Fixed Targets: On-Policy One-Shot Speaker Personalization for AR TTS** — arXiv 2610.17838 **[2026]** — [arXiv](https://arxiv.org/html/2610.17838)
  - Compares GRPO with on-policy self-distillation (OPSD).
  - Finding: GRPO is best overall; OPSD adapts faster and needs no external reward.
  - Closeness: low.
- **When Synthetic Speech Is All You Have: Better Call GRPO** — arXiv 2607.08409 **[2026]** **[title/snippet only]** — [arXiv](https://arxiv.org/html/2607.08409)
- **Qwen-Audio-3.0-TTS (multi-stage training paradigm)** — arXiv 2607.23938 **[2026]**. Multi-stage training, presumably including RL; **[title/snippet only]**. — [arXiv](https://arxiv.org/pdf/2607.23938)
- **IndexTTS 2.5 Technical Report** — Index Speech Team (bilibili) — arXiv 2601.03888 **[2026]**. It is listed among RL/post-trained systems in a search summary, but I did not verify its RL details. — [arXiv](https://arxiv.org/pdf/2601.03888)

### Inferences
- The default reward stack is WER/CER + SIM, sometimes with UTMOS. This matches Dramatizer's guard set almost exactly, which supports Dramatizer's choice of guards.
- Expressiveness rewards are newer and less reliable: SER accuracy, emotion intensity, laughter, LLM pause plans, and LALM style likelihood.

### Gaps
- I found no primary evidence of RL/GRPO in the MiniMax-Speech or IndexTTS2 papers (searches returned nothing). Spark-TTS appears only as a base for third-party GRPO emotion work. This should be checked directly.
- I found no paper applying DDPO specifically to TTS. DDPO appears in TTS papers only as a comparison baseline (e.g., DLPO; unverified) and in a voice-conversion paper (DDPO-VC, arXiv 2606.15313, speaker de-identification; [arXiv](https://arxiv.org/pdf/2606.15313)).
- Several 2026 items are known only from titles.

## Q3. RL for emotional / expressive TTS specifically

### Takeaway
Emotion RL now uses multiple rewards:
- emotion category from SER models
- global intensity, e.g., in VAD space
- local emphasis
- laughter
- LALM-judged role-play style consistency

Recurring problems are reward-model exploitation (acoustic artifacts, repetitive tails), pronunciation cost, and coarse utterance-level rewards that cannot supervise intra-utterance emotion trajectories, which is exactly where markup-level control operates.

### Cited Findings
- **EMORL-TTS: RL for Fine-Grained Emotion Control in LLM-based TTS** — Haoxun Li, Yu Liu, Yuqing Sun, Hanlei Shi, Leyuan Qu, Taihao Li — arXiv 2510.05758, Oct 2025, ICASSP 2026 **[2025]** — [arXiv](https://arxiv.org/abs/2510.05758)
  - Rewards: emotion category, global intensity (in VAD space) and local emphasis.
  - Method: SFT, then GRPO.
  - Finding: better emotion accuracy, intensity differentiation and emphasis clarity, with quality only "comparable" to baselines. It also studies how emphasis placement changes perceived intensity.
  - Base model: Spark-TTS with a KL anchor to the SFT policy, per a third-party snippet (uncertain). — [snippet](https://arxiv.org/pdf/2510.05758)
  - Closeness: **high**. Emphasis placement as an intensity lever is a core Dramatizer gene, but EMORL trains it into the model.
- **Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability (MCLP)** — Yong Ren, Jingbei Li, Haiyang Sun et al. — arXiv 2601.22661, Jan 2026, ICML 2026 **[2026]** — [arXiv](https://arxiv.org/abs/2601.22661)
  - Reward: likelihood under a large audio-language model (LALM) of ground-truth continuation tokens, given the generated speech. It is used both as a metric and as an RL reward for style alignment with role-play instructions, and aligns with human judgements of style consistency.
  - Reported hacking (per search snippet): optimising a single style metric caused "severe hacking" with fixed repetitive acoustic patterns at utterance ends and CER above 50%. The fix was a combined reward that includes CER. IndexTTS2 and Spark-TTS are used as baselines. — [snippet](https://arxiv.org/html/2601.22661v2)
  - Closeness: **high**. The style reward is a candidate scorer for Dramatizer, and the hacking finding validates Dramatizer's multiplicative WER guard.
- **RRPO** (see Q1): emotion RM exploited via acoustic artifacts; hybrid-regularised RM fixes it. — [arXiv](https://arxiv.org/abs/2512.04552)
- **Seed-TTS RL-SER** (see Q2): SER-accuracy reward with REINFORCE raises emotion accuracy substantially (surprise 0.13→0.82). — [arXiv HTML](https://arxiv.org/html/2406.02430)
- **DiffRO-EMO** (see Q1): SER reward via differentiable optimisation, with a noted pronunciation cost. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- **GLM-TTS** (see Q2): emotion + laughter rewards in GRPO. — [GitHub](https://github.com/zai-org/GLM-TTS)
- **HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional TTS** — arXiv 2606.28249 **[2026]** **[title/snippet only]** — [arXiv](https://arxiv.org/pdf/2606.28249)
- **EmphTTS: an emphasis-control TTS with RL** — arXiv 2609.27599 **[2026]** **[title/snippet only]**. Closeness: likely high, since emphasis is a Dramatizer gene. — [arXiv](https://arxiv.org/pdf/2609.27599)
- **Post-Training Zero-Shot TTS for Fine-Grained Emotion and Duration Control via Natural Language** — arXiv 2609.11523 **[2026]** **[title/snippet only]** — [alphaXiv](https://www.alphaxiv.org/abs/2609.11523)
- **Sequential Trajectories and Simultaneous Blending: Multi-Emotion Modeling for Instruction-Following TTS** — arXiv 2608.30325 **[2026]**. Its related-work section notes that existing emotion-RL methods "largely formulate rewards over global features", which gives limited supervision for multi-stage emotion trajectories or blended emotions (via search snippet). — [arXiv](https://arxiv.org/pdf/2608.30325)
- **RLAIF-SPA** (RL from AI feedback for expressive speech) — referenced in a search summary only **[title/snippet only, no URL verified]**.

### Inferences
- The emotion rewards in the literature are mostly categorical-SER accuracy or intensity. None I found uses a novelty or "optimal arousal" (Wundt-curve) objective. Dramatizer's inverted-U expressiveness term appears to be novel relative to this literature, which tends to maximise intensity monotonically.
- The global-reward limitation noted in 2608.30325 suggests that span-level markup search (Dramatizer) addresses a granularity that utterance-level RL rewards handle poorly.

### Gaps
- No found paper reports quantitative diversity of emotional renditions after RL.
- EMORL's base model and reward implementations (e.g., whether emotion2vec is used) are unverified.

## Q4. Test-time scaling / best-of-N / verifier search for TTS

### Takeaway
Llasa is the main published example. It runs best-of-N (ORM) and PRM-guided beam search over the codec-token stream with SIM, WER and emotion verifiers, and reports that inference compute biases outputs toward each verifier's preference (better emotion, timbre or content). Training-free prompt-side methods also exist: prompt selection (ExpPro) and conditioning/duration steering (TED-TTS). I found no paper that searches over text-side paralinguistic markup with verifiers, so Dramatizer's design looks unoccupied.

### Cited Findings
- **Llasa** — arXiv 2502.04128 **[2025]** — [arXiv HTML](https://arxiv.org/html/2502.04128v1), [alphaXiv](https://www.alphaxiv.org/abs/2502.04128)
  - Verifiers: a WavLM-finetuned speaker-verification model for SIM, plus others.
  - Strategies: ORM best-of-N and PRM beam search. At matched compute (B×N candidates), PRM beam search beats ORM best-of-N.
  - Finding: SIM improves markedly as test-time compute grows. The abstract states that scaling inference compute shifts sampling toward the biases of particular verifiers, improving emotional expressiveness, timbre consistency and content accuracy.
  - Closeness: **high**. It is the same paradigm (inference-time search with WER/SIM/emotion verifiers) on a frozen TTS, but it searches over acoustic-token sampling rather than prompt markup, and has no human-in-the-loop or diversity objective.
- **Expressive Prompting (ExpPro): Improving Emotion Intensity in zero-shot TTS** — arXiv 2409.18512 (2024) — [arXiv](https://arxiv.org/pdf/2409.18512)
  - Method: training-free, two-stage static-dynamic selection of the *acoustic prompt* (reference audio), based on its text and speech, to raise emotion intensity in LM-based TTS.
  - Closeness: **high-medium**. It is inference-time and prompt-side, but selects reference audio rather than editing text markup.
- **TED-TTS: Training-Free Intra-Utterance Emotion and Duration Control** — NUS SMC — arXiv 2601.03170, 2026 **[2026]** — [arXiv](https://arxiv.org/html/2601.03170v2), [PDF](https://smcnus.comp.nus.edu.sg/archive/pdf/2026/2026_TED_TTS.pdf)
  - Method: segment-aware conditioning and duration steering on frozen AR TTS backbones. Prompt construction uses a fine-tuned Qwen3-8B.
  - Limitation acknowledged: without parameter updates, duration control is not strictly linear or fine-grained under highly expressive or out-of-domain conditions.
  - Closeness: **high**. It is training-free intra-utterance control, but rule-driven rather than verifier-searched.
- **Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching-Based Zero-Shot TTS** — arXiv 2407.12229 (2024) — [arXiv](https://arxiv.org/pdf/2407.12229)
  - Inference-time control of time-varying emotion, including laughter (details from title only).
  - Closeness: medium.

### Inferences
- Dramatizer can be framed as "test-time scaling in prompt space". Llasa scales over token samples given a fixed prompt; Dramatizer scales over prompts and also samples tokens. The two are orthogonal and could be nested: an inner best-of-N per genome, with an outer evolutionary search.
- Llasa's verifier-bias observation predicts that Dramatizer's expressiveness term will steer outputs toward whatever the prosody descriptors reward. The Wundt-curve shape and the human final pick are mitigations.

### Gaps
- I found no TTS paper using evolutionary or genetic search, novelty search, or quality-diversity (MAP-Elites) at inference time.
- I found no paper reporting best-of-N with human selection from a diverse shortlist for TTS.

## Q5. Reported problems: reward hacking, diversity loss / neutral collapse, intelligibility-expressiveness trade-off

### Takeaway
The reported failures are consistent across papers:
- WER/CER rewards produce slow, over-articulated, monotone speech (Seed-TTS; Shin et al.).
- SIM rewards destabilise training, causing non-termination, or are lost under other rewards (CosyVoice 3).
- Emotion and style reward models are exploited via artifacts or repetitive tails (RRPO; MCLP paper).
- Emotion gains cost pronunciation (DiffRO-EMO).

Mitigations are KL anchors, multi-reward combinations that include CER, entropy rewards, robust reward models, and small-scale human-preference DPO.

### Cited Findings
- Seed-TTS: to reduce WER the model produced slower, overly clear speech that hurt naturalness, and careful tuning was needed. — [arXiv HTML](https://arxiv.org/html/2406.02430)
- **No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS** — Seungyoun Shin, Dongha Ahn, Jiwoo Kim, Sungwook Jeon — arXiv 2509.18531, Sep 2025, ICASSP 2026 **[2025]** — [arXiv](https://arxiv.org/abs/2509.18531)
  - GRPO with CER/NLL lowers error rates but "collapses prosody into monotone, unnatural speech" (reduced logF0 variability).
  - Adding a SIM reward destabilised training and degraded CER. Utterances often failed to terminate, i.e., the objective was partially "hacked". — [PDF snippet](https://arxiv.org/pdf/2509.18531)
  - Fix: iterative DPO with a few hundred human preference pairs per round. On a Korean call-centre dataset (KoCC-TTS) it achieved the highest human ELO with competitive CER, beating GRPO and commercial systems.
  - Closeness: **high** in motivation. It argues prosody has no verifiable reward, which supports Dramatizer's human-in-the-loop pick.
- CosyVoice 3: RL lowers SIM (described as "hacking"); DiffRO-EMO harms pronunciation. — [arXiv HTML](https://arxiv.org/html/2505.17589v2)
- RRPO: emotion-RM hacking via acoustic artifacts. — [arXiv](https://arxiv.org/abs/2512.04552)
- MCLP role-play paper: single style reward produces repetitive end-of-utterance patterns and CER above 50%; fixed by adding CER to the reward. — [snippet](https://arxiv.org/html/2601.22661v2)
- Multi-reward GRPO adds entropy regularisation and length penalty for stability. — [arXiv](https://arxiv.org/abs/2511.21270)
- DLPO adds the diffusion loss as a regulariser to prevent drift from the base model. — [arXiv](https://arxiv.org/pdf/2405.14632)
- Llasa: verifier search biases outputs toward the verifier. — [arXiv](https://arxiv.org/html/2502.04128)
- Tian et al.: SIM and proxy-MOS can exceed human speech after DPO. This is a gain, but it also hints at metric over-optimisation; the authors claim generalisation to unseen metrics and human tests. — [arXiv](https://arxiv.org/html/2409.12403v1)

### Inferences
- "Mode collapse toward neutral delivery" is directly documented for intelligibility rewards (Seed-TTS, Shin et al.). This is the strongest argument that a reward-tuned TTS needs an explicit expressiveness or diversity term, and that Dramatizer's guard-times-expressiveness product (rather than guards alone) is well motivated.
- Dramatizer's multiplicative guards mirror the "add CER to the style reward" fix in the MCLP paper.

### Gaps
- No paper found measures sample diversity (e.g., F0-variance across seeds) before vs. after RL in a systematic way. Shin et al.'s logF0 analysis is the closest.

## Q6. Relation of inference-time markup search (Dramatizer) to reward-tuning: complementary or competing? Any combined work?

### Takeaway
The two approaches are mostly complementary. Reward tuning raises the base model's floor (WER, SIM, emotion-following) but documents prosodic flattening and hacking. Inference-time search, as in Llasa and Dramatizer, exploits a fixed model without altering it, keeps per-utterance creative control and human choice, and can use RL-tuned models as backbones. I found no published work that combines RL-tuned TTS with search over text-side paralinguistic markup.

### Cited Findings
- Training-time approaches that target the same levers as Dramatizer genes:
  - emphasis: EMORL-TTS, EmphTTS
  - pauses/prosody: Multi-reward GRPO's LLM pause reward
  - duration: DMOSpeech 2
  - laughter: GLM-TTS
  - Sources: [EMORL](https://arxiv.org/abs/2510.05758); [EmphTTS](https://arxiv.org/pdf/2609.27599); [Multi-reward GRPO](https://arxiv.org/abs/2511.21270); [DMOSpeech 2](https://arxiv.org/abs/2507.14988); [GLM-TTS](https://github.com/zai-org/GLM-TTS)
- Inference-time approaches: Llasa (verifier search over tokens), ExpPro (acoustic-prompt selection), TED-TTS (training-free segment conditioning). — [Llasa](https://arxiv.org/html/2502.04128); [ExpPro](https://arxiv.org/pdf/2409.18512); [TED-TTS](https://arxiv.org/html/2601.03170v2)
- Human-preference-in-the-loop on small data is effective for prosody where automatic rewards fail (Shin et al.) and for subjective utility (UNO). Dramatizer's human pick yields exactly this kind of preference data. — [Shin et al.](https://arxiv.org/abs/2509.18531); [UNO](https://arxiv.org/abs/2406.00654)

### Inferences
- **Complementary pipeline**:
  1. Dramatizer's human picks over search-diversified candidates produce preference pairs (winner vs. shortlisted losers) with the markup context attached.
  2. Those pairs feed DPO or iterative DPO (as in Shin et al. or UNO) on CosyVoice, ideally with markup-conditioned inputs.
  3. Dramatizer's guards and Wundt term can serve as the GRPO reward for a follow-up, with known risks: monotone collapse from the WER guard, and hacking of the prosody-descriptor term.
- **Competing aspects**: a strongly reward-tuned model (e.g., SER-maximised) may become less responsive to markup, or may saturate emotion so the Wundt term has less room. RL that reduces output variance also reduces the space an evolutionary search can explore. This is speculative; no paper measures markup responsiveness after RL.
- Dramatizer's novelty relative to the catalogue:
  - search space = text-side paralinguistic markup, rather than tokens or acoustic prompts
  - a non-monotone (Wundt) expressiveness objective, rather than maximising emotion accuracy
  - diversity-aware shortlisting plus a human final choice
  - no weight updates

### Gaps
- No found work combines the two. I also found no study of how RL post-training changes a TTS model's sensitivity to inline instructions or markup tags. CosyVoice 3 does not report DiffRO's effect on instructed generation.
- MiniMax-Speech, IndexTTS2 and Spark-TTS RL details were not confirmed from primary sources.
