# Expressive and controllable TTS with inline paralinguistic tags or natural-language instructions (prior work for Dramatizer)

Research date: 2026-10-27. Sources were opened via WebFetch or seen in WebSearch results. Items marked "(list-only)" come from the README of the awesome-controllable-speech-synthesis list that accompanies the EMNLP 2025 survey. I did not open those papers, so treat their details as unverified. The list was fetched from https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md; the repo name really is spelled "controllabe".

Closeness scale relative to Dramatizer:
- **high**: same mechanism, meaning inline tags or instructions in a zero-shot cloning TTS, plus some form of automatic selection or optimisation.
- **medium**: same control surface, but humans or an LLM write the controls, or the controls are trained into the model rather than searched.
- **low**: a different control mechanism, such as reference style, global descriptions or embeddings.

---

## Q1. CosyVoice 2 and CosyVoice 3 (Fun-CosyVoice3): fine-grained tokens, instruct capabilities, training, and controllability results

### Takeaway
Dramatizer's gene vocabulary maps onto two mechanisms in CosyVoice 2/3:
- a natural-language instruction prefix, terminated by `<|endofprompt|>`;
- inline fine-grained tags: `[laughter]` and `[breath]` as vocal bursts, `<strong>…</strong>` for emphasis, and `<laughter>…</laughter>` for laughing while speaking.

Both were learned through supervised training on instruction data: about 1.5k hours for CosyVoice 2 and about 5k hours for CosyVoice 3. Evaluation uses human MOS-I, style-similarity and WER on in-house and Expresso sets. Neither paper searches over tag placement.

### Cited findings
- **CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models.** Du, Wang, Chen, Shi, Lv, Zhao, Gao, … Zhou (Alibaba). arXiv:2412.10117, Dec 2024. — [arXiv HTML](https://arxiv.org/html/2412.10117)
  - About 1,500 hours of instructed data were added to base training. The instruction is prepended to the text and ended with `<|endofprompt|>`. — [arXiv HTML](https://arxiv.org/html/2412.10117)
  - Instruction categories: [arXiv HTML](https://arxiv.org/html/2412.10117)
    - emotion: happy, sad, surprised, angry, fearful, disgusted, calm, serious;
    - speaking rate: fast, very fast, slow, very slow;
    - dialect: Cantonese, Sichuanese, Shanghainese, Zhengzhou, Changsha, Tianjin;
    - role-play: mysterious, fierce, curious, elegant, lonely, robot, Peppa, and others.
  - Fine-grained controls: vocal bursts `[laughter]` and `[breath]` inserted between text tokens, `<strong>…</strong>` for emphasis, and `<laughter>…</laughter>` for speaking with laughter. — [arXiv HTML](https://arxiv.org/html/2412.10117)
  - Evaluation set: an in-house Chinese set of 290 samples (29 instruction types × 10 texts) over 5 speakers. The metric is MOS-I (instruction accuracy plus naturalness, 1–5), rated by 10 native speakers. — [arXiv HTML](https://arxiv.org/html/2412.10117)
  - Results: [arXiv HTML](https://arxiv.org/html/2412.10117)

    | Model | CER (%) | SS | NMOS | MOS-I |
    |---|---|---|---|---|
    | CosyVoice-Instruct | 1.72 | 0.797 | 3.94 | 3.09 |
    | CosyVoice 2 | 1.52 | 0.804 | 3.94 | 4.06 |
    | CosyVoice 2 without instruction | 0.97 | 0.817 | 4.02 | 2.28 |

  - Adding instructions costs some CER and speaker similarity. The authors conclude that controllability "is hard to emerge implicitly from the content text". — [arXiv HTML](https://arxiv.org/html/2412.10117)
- **CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training.** Du, Gao, Wang, Yu, Zhao, Wang, Lv, … Ye (22 authors). arXiv:2505.17589, submitted 23 May 2025 (v2 27 May 2025). **[2025]** — [arXiv abs](https://arxiv.org/abs/2505.17589)
  - Instruction data grew from 1,500 to 5,000 hours and covers more than 100 styles: emotions, speed, voice tones, dialects, accents and role-play (Sec. 2.5, Table 1). It keeps the same inline tags: `[laughter]`, `[breath]` and `<strong>XXX</strong>`. — [arXiv HTML](https://arxiv.org/html/2505.17589)
  - Instruction-following evaluation (Sec. 5.6, Table 14) used a 3,000-sample Expresso subset (8 styles) and an internal 3,600-sample set (more than 50 styles). — [arXiv HTML](https://arxiv.org/html/2505.17589)

    | Set | Model | WER | Style SIM | MOS |
    |---|---|---|---|---|
    | Expresso | CosyVoice 2 | 9.42 | 60.98 | 3.54 |
    | Expresso | CosyVoice 3-1.5B | 13.43 | 68.25 | 3.56 |
    | Expresso | ground truth | 10.0 | — | — |
    | Internal | CosyVoice 2 | 7.75 | 72.99 | — |
    | Internal | CosyVoice 3-1.5B | 7.31 | 81.06 | — |

    The paper attributes the higher Expresso WER partly to ASR bias against emotional speech. — [arXiv HTML](https://arxiv.org/html/2505.17589)
  - Stated limitation: timbre cannot be controlled by text instruction, and singing is weak (Sec. 7). — [arXiv HTML](https://arxiv.org/html/2505.17589)
  - During speaker fine-tuning, the speaker and style fields in the instruction prompt are randomly masked to aid transfer and limit forgetting (Sec. 2.6.2). — [arXiv HTML](https://arxiv.org/html/2505.17589)
  - DiffRO post-training optimises speech tokens with Gumbel-Softmax, an ASR-like Token2Text reward and token-level KL regularisation. — [arXiv HTML](https://arxiv.org/html/2505.17589)
    - A multi-task reward adds speech emotion recognition, MOS prediction and audio-event detection.
    - RL gives 20–50% relative WER improvements, and speaker similarity drops slightly, which the paper calls possible reward hacking.
    - DiffRO-EMO raises "happy" emotion accuracy from 0.64 to 0.98 for CosyVoice3-1.5B, but can hurt pronunciation.
  - The tokenizer is trained multi-task on ASR, speech emotion recognition, language ID, audio-event detection and speaker analysis. — [arXiv abs](https://arxiv.org/abs/2505.17589)
- **Fun-CosyVoice3 open release.** The models are Fun-CosyVoice3-0.5B-2512 and Fun-CosyVoice3-0.5B-2512_RL, released **Dec 2025**. **[2025]** — [GitHub FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
  - The README claims support for "various instructions such as languages, dialects, emotions, speed, volume, etc.", 9 languages, 18+ Chinese dialects, and pinyin/CMU phoneme pronunciation inpainting. — [GitHub](https://github.com/FunAudioLLM/CosyVoice)
  - The README points to `example.py` for usage and does not document the fine-grained tags. — [GitHub](https://github.com/FunAudioLLM/CosyVoice)
  - The roadmap lists CosyVoice2 GRPO training support in Aug 2025 and an eval set release in Jul 2025. — [GitHub](https://github.com/FunAudioLLM/CosyVoice)
- **SynParaSpeech** (arXiv:2509.14946, Sep 2025) automatically synthesises paralinguistic datasets. Fine-tuning CosyVoice2 on it reportedly gave a large improvement in paralinguistic synthesis. Seen in a search snippet only; the paper was not opened. **[2025]** — [arXiv PDF](https://arxiv.org/pdf/2509.14946)

### Inferences
- Closeness of CosyVoice 2/3 to Dramatizer is **high as a substrate**: they are its control surface. They are **not prior work on searching** that surface, because the papers evaluate fixed, human-written instructions and tags.
- The trade-off reported in the CosyVoice 2 table matters for Dramatizer's fitness function: instructions raise MOS-I but worsen CER and speaker similarity. Optimising WER and similarity alone would therefore push the search toward fewer or no tags. This argues for the novelty/expressiveness term.
- DiffRO and DiffRO-EMO are the closest "automatic optimisation" in the CosyVoice line, but they optimise **model weights** via reward, not **per-utterance markup**. Both use ASR and emotion-recognition rewards, which mirror Dramatizer's Whisper WER and validity checks.

### Gaps
- I did not open the full tag inventory in the CosyVoice tokenizer or frontend code. Additional tokens may exist in code, for example `[noise]`, `[cough]`, `[quick_breath]`, `<endofprompt>` variants, or `[lipsmack]`. I could not verify these, so check `cosyvoice/tokenizer/tokenizer.py` directly.
- Neither paper reports a separate quantitative evaluation of the inline `[breath]`, `[laughter]` and `<strong>` tags. MOS-I aggregates all 29 instruction types.

---

## Q2. Commercial and open systems with audio or event tags: control markers each accepts

### Takeaway
Tag vocabularies are proliferating, but they are model-specific and mostly document-defined rather than standardised. Most come with caveats that effects depend on the voice and that tags should be used sparingly.

Every system below expects a human or an LLM to place the tags. None documents automatic placement or search.

### Cited findings
- **ElevenLabs Eleven v3 (and v4) audio tags.** Official prompting docs (date not stated on the page). — [ElevenLabs docs](https://elevenlabs.io/docs/best-practices/prompting/eleven-v3)
  - Voice tags: `[laughs]`, `[laughs harder]`, `[starts laughing]`, `[wheezing]`, `[whispers]`, `[sighs]`, `[exhales]`, `[sarcastic]`, `[curious]`, `[excited]`, `[crying]`, `[snorts]`, `[mischievously]`.
  - Sound effects: `[gunshot]`, `[applause]`, `[clapping]`, `[explosion]`, `[swallows]`, `[gulps]`.
  - Experimental: `[strong X accent]`, `[sings]`, `[woo]`, `[fart]`.
  - The list is described as not exhaustive.
  - Ellipses add pauses, and capitalisation increases emphasis. **SSML `<break>` is not supported in v3.**
  - Tags work best when the delivery is in the voice's training data. Experimental tags are less consistent across voices, and Professional Voice Clones are "not fully optimized for v3".

  Closeness to Dramatizer: medium. It is the same inline-tag paradigm with explicit voice-dependence of tags, which is the problem Dramatizer's per-voice search addresses, but it relies on manual prompting.
- **Bark (Suno).** Released 20 Apr 2023. — [GitHub suno-ai/bark](https://github.com/suno-ai/bark)
  - Markers: `[laughter]`, `[laughs]`, `[sighs]`, `[music]`, `[gasps]`, `[clears throat]`, `—` or `...` for hesitation, `♪` for lyrics, CAPITALISATION for emphasis, and `[MAN]`/`[WOMAN]`.
  - The list is "not exhaustive", and the output is not guaranteed ("creative liberties").

  Closeness: low–medium. It has an early tag vocabulary but no cloning-quality control and no search.
- **Dia-1.6B (Nari Labs).** English-only dialogue TTS with `[S1]`/`[S2]` speaker tags. — [GitHub nari-labs/dia](https://github.com/nari-labs/dia)
  - Non-verbal tags: (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles).
  - Caveats: use tags sparingly, because overuse or unlisted tags may cause artifacts. Inputs shorter than about 5 s sound unnatural and inputs longer than about 20 s are rushed.
  - Cloning works by prepending the transcript of a 5–10 s reference.
  - The README notes a Dia2 release on "11/19" (assumed Nov 2025, but the year is not stated). **[2025, uncertain]**

  Closeness: medium. It has the same tag-in-prompt plus clone setup, and the documented fragility of tag placement motivates validity checks.
- **Orpheus TTS (Canopy Labs).** Llama-3B backbone. — [GitHub canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
  - Tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`. Multilingual models have different tags.
  - Fixed voices: tara, leah, jess, leo, dan, mia, zac, zoe.
  - Multilingual release Apr 2025. Tag training is not described. **[2025]**

  Closeness: medium–low, because it uses fixed voices rather than zero-shot cloning.
- **OpenAudio S1 / S1-mini (Fish Audio).** S1 has 4B parameters (proprietary), and S1-mini has 0.5B (distilled, open). Trained on more than 2M hours, with online RLHF. Release date not stated on the card. — [HF model card](https://huggingface.co/fishaudio/openaudio-s1-mini)
  - 66 parenthesised markers:
    - 51 emotions, e.g. (angry), (sarcastic), (hesitating), (amused);
    - 5 tones: (in a hurry tone), (shouting), (screaming), (whispering), (soft tone);
    - 10 special: (laughing), (chuckling), (sobbing), (crying loudly), (sighing), (panting), (groaning), (crowd laughing), (background laughter), (audience laughing).
  - Annotation method is not described.

  Closeness: medium. It is the largest open inline emotion vocabulary found, and a plausible alternative backend.
- **Chatterbox Turbo (Resemble AI).** Seen in search results only. **[2025, date uncertain]** — [Chatterbox tag guide (third-party mintlify)](https://www.mintlify.com/yocxy2/chatterboxyocxy/guides/paralinguistic-tags); [fal blog](https://blog.fal.ai/chatterbox-turbo-is-now-available-on-fal)
  - Reported tags: [laugh], [chuckle], [sigh], [gasp], [cough], [clear throat], [sniff], [groan], [shush].
  - Tags reportedly work **only on Turbo**. Standard and multilingual Chatterbox read them aloud as text.

  Closeness: medium.
- **Step-Audio-EditX (StepFun).** 3B LLM-based audio editing model. Technical report arXiv:2511.03601 (6 Nov 2025), weights 12 Nov 2025, GRPO training code 23 Jan 2026. **[2025–2026]** — [HF README](https://huggingface.co/stepfun-ai/Step-Audio-EditX/blob/main/README.md)
  - 14 emotion tags, e.g. happy, angry, humour, coldness.
  - 33 style tags, e.g. whisper, recite, news, murmur, shout, act_coy.
  - 10 paralinguistic types: Breathing, Laughter, Surprise-oh, Confirmation-en, Uhm, Surprise-ah, Surprise-wa, Sigh, Question-ei, Dissatisfaction-hnn.
  - Iterative editing: each round re-edits the previous output, and emotion accuracy improves over rounds. More than one iteration is recommended for whisper.
  - Training uses "large-margin data" in SFT and PPO.

  Closeness: medium–high. It is iterative refinement of expressive control applied to the **audio**. It is a fixed edit loop rather than a population search, and it has no cross-voice genome transfer.
- **Step-Audio** (arXiv:2502.11946, Feb 2025) is listed with prosody, timbre, emotion and description control. (list-only) **[2025]** — [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)
- **IndexTTS2.** Zhou, Zhou, He, Zhou, Wang, Deng, Shu (bilibili). arXiv:2506.21619, Jun 2025 (v2 Sep 2025). **[2025]** — [arXiv abs](https://arxiv.org/abs/2506.21619)
  - Timbre and emotion are disentangled: the timbre prompt and the emotion/style prompt come from separate audio.
  - A "soft instruction" mechanism lets a fine-tuned Qwen3 map text descriptions to emotion.
  - Explicit token-count duration control is available, alongside a free mode.
  - The abstract reports gains in WER, speaker similarity and emotional fidelity.

  Closeness: medium. It offers emotion and duration control separated from voice identity, which is relevant to cross-voice transfer, but the controls are global rather than inline positional tags. The list also notes **IndexTTS 2.5** (arXiv:2601.03888, Jan 2026, RL) as list-only. **[2026]**
- **Spark-TTS** (arXiv:2503.01710, Mar 2025): pitch, speed, prosody and timbre control (attribute-level, coarse). (list-only) **[2025]** — [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)
- **Practitioner report.** A DEV Community post finds that bracketed tags such as [laugh] and [sigh] were silent or read literally on Inworld TTS, and that punctuation worked better. This is anecdotal. — [DEV Community](https://dev.to/sm1ck/inworld-tts-paralinguistic-tags-dont-work-heres-what-does-50pj)

### Inferences
- Across vendors, documentation repeatedly says tag effects are voice-dependent, unreliable or artifact-prone: ElevenLabs, Dia, Bark, Chatterbox and Inworld. This is the empirical motivation for Dramatizer: per-voice search with automatic validity checks, rather than one-shot hand-written tags.
- Vocabularies differ syntactically (`[...]`, `(...)`, `<...>`) and semantically. Dramatizer's position-based genes are tied to CosyVoice syntax, so porting would require a per-backend vocabulary map.

### Gaps
- I did not open or verify the control interfaces of Sesame CSM, Kokoro, F5-TTS/E2-TTS, MaskGCT, Higgs Audio v2, MiniMax-Speech or Kimi-Audio.
  - From prior knowledge, which is unverified: Kokoro, F5-TTS/E2-TTS and MaskGCT accept plain text plus a reference prompt, with no documented event tags (F5 does have a speed parameter). Sesame CSM conditions on conversational context rather than tags.
  - These claims need checking against each repo before use.
- I could not find an official release date for OpenAudio S1 or Eleven v3 on the pages I opened.

---

## Q3. Description- and instruction-based TTS (global style prompts)

### Takeaway
A large body of work, from 2022 to 2026, conditions TTS on free-text style descriptions. Most of it uses automatically or semi-automatically generated captions from attribute classifiers plus LLM paraphrasing. Evaluation uses attribute-classification accuracy, MOS and benchmarks such as InstructTTSEval.

These models control **global, utterance-level** style. Word- and position-level control only emerges in 2025–2026 papers.

### Cited findings
All items in this section are list-only unless stated otherwise. Source: [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md).

**Global description and instruction models**

| Model | Date | Reference | Controls | Closeness |
|---|---|---|---|---|
| PromptTTS | Nov 2022 | IEEE (ICASSP) | pitch, energy, speed, prosody, timbre, emotion via description | low |
| PromptTTS 2 | Sep 2023 | arXiv:2309.02285 | pitch, energy, speed, prosody, timbre via description | low |
| PromptStyle | May 2023 | arXiv:2305.19522 | pitch, prosody, timbre, emotion via description | low |
| PromptTTS++ | Sep 2023 | IEEE | speaker and style description | low |
| InstructTTS | listed May 2024 | IEEE/ACM TASLP | description-driven style | low |
| TextrolSpeech | 2023 | IEEE (ICASSP) | dataset of text-described styles: pitch, energy, speed, emotion labels | low |
| VoxInstruct | Aug 2024 | ACM (MM) | unified instruction plus content | low–medium |
| ControlSpeech | Jun 2024 | arXiv:2406.01205 | zero-shot timbre plus described style | medium–low |
| CosyVoice (v1) | Jul 2024 | arXiv:2407.05407 | instruct model, predecessor of CosyVoice 2 | — |
| FleSpeech | Jan 2025 **[2025]** | arXiv:2501.04644 | multimodal style control | low |
| EmoVoice | Apr 2025 **[2025]** | arXiv:2504.12867 | emotion via free-text description | low–medium |
| EmoSphere++ | Nov 2024 | arXiv:2411.02625 | emotion-intensity, style and timbre | low |
| ParaStyleTTS | Oct 2025 **[2025]** | arXiv:2510.18308 | paralinguistic style | low |
| FillerSpeech | Nov 2025 **[2025]** | no arXiv id | filler and nonverbal generation | medium–low |

- ControlSpeech (zero-shot timbre plus described style) is the closest of the global models to the "clone plus style" setting.
- ParaStyleTTS was also seen in search results. Its snippet notes that LLM-based style prompting is "highly sensitive to the phrasing of the input prompt". This is relevant because the phrasing of the natural-language instruction gene is itself a search dimension. — [arXiv PDF](https://arxiv.org/pdf/2510.18308)
- **Natural language guidance of high-fidelity TTS with synthetic annotations (Parler-TTS).** Lyth & King, arXiv:2402.01912, 2024.
  - The list includes it only as a dataset entry. — [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)
  - Title and authors are from prior knowledge, matched to the listed arXiv id. Details beyond that, such as captions synthesised from automatically labelled gender, pitch, rate and noise, are unverified here. Closeness: low.
- **Scaling Rich Style-Prompted TTS Datasets (ParaSpeechCaps).** arXiv:2503.04713, 2025. Search snippet only. Automatically annotates rich style tags at scale; the snippet reports situational-tag recall of 71.3% for automatic labels versus 68.1% for humans. **[2025]** — [alphaxiv](https://www.alphaxiv.org/abs/2503.04713)

**Surveys and benchmarks**
- **Survey:** "Towards Controllable Speech Synthesis in the Era of LLMs: A Systematic Survey". Xie, Rong, Zhang, Wang, Liu. EMNLP 2025. **[2025]** — [ACL Anthology preview](https://preview.aclanthology.org/setup/2025.emnlp-main.40)
- **InstructTTSEval** (arXiv:2506.16381, Jun 2025). 6k cases across 3 tasks in English and Chinese, with Gemini as automatic judge. It finds substantial room for improvement in instruction following. **[2025]** — [alphaxiv](https://alphaxiv.org/abs/2506.16381)
- **"Do You Hear What I Mean?"** (arXiv:2509.13989, Sep 2025). Quantifies the gap between instruction and perception for Parler-TTS, PromptTTS++, GPT-4o-mini-TTS, UniAudio and others. **[2025]** — [arXiv PDF](https://arxiv.org/pdf/2509.13989)
- **EmergentTTS-Eval** (arXiv:2505.23009, May 2025). Evaluates TTS on complex prosodic and expressive prompts. Search result only. **[2025]** — [arXiv PDF](https://arxiv.org/pdf/2505.23009)

**2026 instruction and RL entries** (list-only, unverified; titles and ids as listed) **[2026]**
- Qwen-Audio-3.0-TTS (arXiv:2607.23938, Jul 2026): free instruction control.
- VoiceDesigner (arXiv:2608.13613): voice design from descriptions.
- Interactive TTS (arXiv:2609.25707).
- CycleSpeech (arXiv:2609.24771): instruction control with RL.
- "Listen, Critique, and Refine" (arXiv:2609.24163): RL self-refinement of instruction-following and prosody.
- "Diagnose, Then Refine" / LoopTTS (arXiv:2608.28970).
- Stabilizing Instruction Supervision for Instruct-TTS (arXiv:2609.08204).
- Post-Training Zero-Shot TTS (arXiv:2609.11523).
- InstCharVoice (arXiv:2609.36287): character-level instruction control.

Source for this group: [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)

### Inferences
- Description-based TTS mostly controls **global** style and rarely inline or positional events, so it maps only onto Dramatizer's "natural-language instruction" gene.
- The 2025–2026 "critique and refine" and "diagnose then refine" loops are the closest conceptually to an automatic feedback loop. From their list descriptions, they appear to refine the **model** or the generated output via RL or self-critique, not to search an explicit markup genome. This is unverified.

### Gaps
- UniStyle, the original EmoSphere, MiniMax-Speech, MaskGCT, F5-TTS, EmoKnob and Kimi-Audio did not appear in the list fragments I read, and I did not open them.
- InstructTTS, VoxInstruct and TextrolSpeech have no arXiv id in the list, so their authors are unverified here.
- None of the list-only papers were opened, so their training-data and evaluation specifics are not captured.

---

## Q4. Emphasis, prominence and word-level prosody control

### Takeaway
Emphasis control has moved from annotated-score conditioning in non-autoregressive TTS (2023–2025) to RL and preference-aligned emphasis in flow-matching and LLM TTS (2026). Word-level multi-attribute control appears in WeSCon (NeurIPS 2025) and WordVoice (2026).

### Cited findings
- **Explicit Emphasis Control in Text-to-Speech Synthesis.** Bauer et al., ISCA SSW 2025. **[2025]** — [ISCA Archive](https://www.isca-archive.org/ssw_2025/bauer25_ssw.html); [ePrint PDF](https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2025_BauerZMD_EmphasisControl_SSW_ePrint.pdf)
  - Training data is annotated with emphasis scores, and ForwardTacotron is conditioned on them. Seen via search snippet.
  - Closeness: low–medium, since it is not zero-shot.
- **Controllable Emphasis with zero data for TTS.** Joly et al., arXiv:2307.07062, 2023. Emphasis on a selected word, used for narrow or contrastive focus, without emphasis-labelled data. Search snippet only. — [arXiv PDF](https://arxiv.org/pdf/2307.07062)
  - Closeness: medium–low.
- **Word-Level Emotional Expression Control in Zero-Shot TTS (WeSCon).** Wang et al., arXiv:2509.24629, NeurIPS 2025. **[2025]** — [arXiv](https://arxiv.org/abs/2509.24629v1); [NeurIPS](https://neurips.cc/virtual/2025/poster/117924)
  - Adds word-level emotion and speaking-rate control to a pretrained zero-shot TTS. It uses transition smoothing, dynamic speed control and multi-round inference, then self-training.
  - Closeness: medium–high. It is word-positioned expressive control on a cloning TTS, but the controls are specified by the user rather than searched.
- **Lina-Style** (Lemerle et al., Aug 2025): word-level style. (list-only) **[2025]** — [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)
- **2026 emphasis and word-level entries** (list-only, unverified) **[2026]** — [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)
  - **EmphTTS** (Li et al., arXiv:2609.27599, Sep 2026): emphasis plus duration via RL.
  - **Refining Emphasis Control in Flow-Matching TTS** (Ye et al., Interspeech 2026): preference alignment and RL. — [OpenReview PDF](https://openreview.net/pdf?id=AFE0Str79J)
  - **WordVoice** (Nie et al., arXiv:2607.06461, Jul 2026): word-level multi-dimensional control in LLM TTS.
  - **EmoTra-TTS** (arXiv:2608.23791): intra-utterance emotion transitions.
  - **CtrlSpeech** (arXiv:2608.08362): coarse-to-fine prosody.
  - **Nonverbal vocalization modeling** (Zhang et al., arXiv:2609.14231).
- **Improving French Synthetic Speech Quality via SSML Prosody Control** (arXiv:2508.17494, Aug 2025). Search result only. It appears to generate SSML prosody markup automatically to improve quality. **[2025]** — [arXiv PDF](https://arxiv.org/pdf/2508.17494)
  - Closeness: medium. The automatic markup generation looks LLM/pipeline-predicted rather than search-based, but this is unverified and the paper should be checked.

### Inferences
- The 2026 trend is to use RL or preference optimisation to make emphasis tags **work better inside the model**. Dramatizer's complementary angle is to keep the model fixed and optimise **which tags go where**, per voice.

### Gaps
- I did not verify the WordVoice, EmphTTS or Ye et al. abstracts, so claims about their mechanisms rely on the list's one-line tags.

---

## Q5. Does any work automatically search the control space per voice, transfer controls across voices, or use quality-diversity?

### Takeaway
**None found.** In the sources searched, I found no TTS work that runs an evolutionary, quality-diversity or black-box search over inline paralinguistic markup or instructions per voice, or that transfers a chosen markup genome across voices or lines.

The nearest neighbours fall into four groups:
- model-side RL or preference optimisation for controllability (CosyVoice 3 DiffRO-EMO, EMORL-TTS, RRPO, Emo-LiPO, EmphTTS);
- iterative editing or self-refinement loops (Step-Audio-EditX, "Listen, Critique, and Refine", LoopTTS);
- automatic annotation and synthesis of paralinguistic data (ParaSpeechCaps, SynParaSpeech);
- an old interactive-evolution voice-conversion paper.

### Cited findings
- A targeted search for automatic optimisation of paralinguistic tags or prompts returned no direct match. — [search results incl. ParaStyleTTS](https://arxiv.org/pdf/2510.18308), [ParaSpeechCaps](https://www.alphaxiv.org/abs/2503.04713), [SynParaSpeech](https://arxiv.org/pdf/2509.14946)
  - The adjacent work found was dataset auto-annotation (ParaSpeechCaps), synthetic paralinguistic data (SynParaSpeech) and prompt-sensitivity observations (ParaStyleTTS).
- A search for evolutionary or quality-diversity search over TTS prosody returned no TTS application. — [ProsodyEval / DS-WED](https://arxiv.org/pdf/2509.19928)
  - **"Voice quality conversion using interactive evolution of prosodic control"** (Applied Soft Computing, about 2004). Interactive evolutionary computation tunes prosodic coefficients, and listeners select among 5 variants per emotive term. This is the closest evolutionary precedent, with a human in the loop, but it is voice conversion rather than tag-based TTS. Search snippet only. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494604000675)
  - **ProsodyEval / DS-WED** (arXiv:2509.19928, Sep 2025): a prosody-diversity metric and benchmark over 7 zero-shot TTS systems. It reports that RL and duration control affect prosody diversity. Potentially useful for Dramatizer's novelty term. **[2025]** — [arXiv PDF](https://arxiv.org/pdf/2509.19928)
  - The quality-diversity literature (MAP-Elites, and a 2026 distributed QD paper for LLM toxicity, arXiv:2606.24166) surfaced with no TTS application. — [arXiv PDF](https://arxiv.org/pdf/2606.24166)
- RL and preference work on controllability, all model-side: [arXiv HTML](https://arxiv.org/html/2505.17589); [awesome list README](https://raw.githubusercontent.com/imxtx/awesome-controllabe-speech-synthesis/main/README.md)
  - CosyVoice 3 DiffRO / DiffRO-EMO.
  - EMORL-TTS (arXiv:2510.05758).
  - RRPO (arXiv:2512.04552).
  - Koel-TTS.
  - Emo-LiPO (arXiv:2606.13006, 2026).
  - Sparse-autoencoder emotion steering (arXiv:2606.01479, 2026). This is activation steering, a different kind of control knob.
- **Step-Audio-EditX** iterative editing improves emotion and style over repeated rounds. It is a fixed refinement loop on audio, not a search over markup. — [HF README](https://huggingface.co/stepfun-ai/Step-Audio-EditX/blob/main/README.md)
- Speaker-transfer–related design: CosyVoice 3 masks speaker and style fields during speaker fine-tuning to improve transfer (Sec. 2.6.2). IndexTTS2 disentangles emotion from timbre. Both support the premise that delivery controls can be separated from voice, but neither transfers searched control configurations. — [arXiv HTML](https://arxiv.org/html/2505.17589); [arXiv abs](https://arxiv.org/abs/2506.21619)

### Inferences
- Dramatizer appears novel in combining four things:
  1. black-box population search over **inference-time markup**, without changing model weights;
  2. **position-relative genes** that transfer across lines and voices;
  3. a multi-objective fitness of WER, speaker similarity and validity, plus a Wundt-curve novelty term;
  4. a diverse human shortlist whose winner seeds the search for the nearest voice.
- Precedents to cite and contrast:
  - interactive evolution of prosody (about 2004);
  - RL for controllability (DiffRO-EMO, EMORL-TTS, EmphTTS);
  - iterative editing and refinement (Step-Audio-EditX, LoopTTS, "Listen, Critique, and Refine");
  - prosody-diversity metrics (DS-WED).
- The voice-dependence caveats in vendor docs (ElevenLabs, Dia, Chatterbox) provide citable motivation for per-voice search.

### Gaps
- The search coverage was limited. I made about 5 search queries, and GECCO/EvoMUSART proceedings and ICASSP 2026 were not checked directly. Absence is "not found", not "proven absent".
- I did not open the 2026 refinement papers (arXiv:2609.24163, 2608.28970), so I cannot rule out that one of them searches over explicit control tokens at inference time. **Those two should be checked first.**
- The 2026 arXiv ids come from a community list I fetched once. I did not verify them individually.
