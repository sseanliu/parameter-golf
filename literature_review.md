# Literature Review: Meta-Learned Test-Time Training for Parameter Golf

**Date:** 2026-03-20
**Goal:** Break through ~1.145 bpb SOTA on 16MB/10min constraint via theoretical innovation
**Hypothesis:** MAML + aggressive TTT can turn a fixed model into a "learning algorithm" that adapts per-document at eval time

---

## 1. Test-Time Training (TTT) for Language Models

### 1.1 TTT Layers: Learning to (Learn at Test Time)
- **Title:** "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
- **Authors:** Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Dhiman, Kilian Weinberger, et al.
- **Venue:** ICML 2024
- **ArXiv:** 2407.04620
- **Key idea:** The hidden state itself IS a machine learning model (linear model or MLP). The update rule is a step of self-supervised learning. Two variants: TTT-Linear (hidden state = linear model) and TTT-MLP (hidden state = 2-layer MLP). Since the hidden state is updated by training even on test sequences, these are called TTT layers.
- **Results:** Competitive with Transformers and Mamba on language modeling benchmarks. TTT-Linear matches Mamba at 125M-1.3B scale. TTT-MLP outperforms both on longer sequences.
- **Relevance to us:** HIGH. The core idea of "learning at test time" is exactly our hypothesis. However, the original TTT layers are designed as sequence model replacements (replacing attention), not as a document-level adaptation mechanism. We need to adapt this to our setting: use TTT as a way to specialize the entire model per-document during evaluation.
- **Feasibility:** MEDIUM. The TTT layer architecture is non-trivial to implement correctly. But the *principle* (gradient steps at test time) is simple. We can use standard gradient descent on the model weights at test time without needing the full TTT layer architecture.

### 1.2 TTT-E2E: End-to-End Test-Time Training for Long Context
- **Title:** "End-to-End Test-Time Training for Long Context"
- **Authors:** Yu Sun et al.
- **Venue:** ArXiv 2512.23675 (December 2025)
- **Key idea:** Formulates long-context LM as continual learning. The model continues learning at test time via next-token prediction, compressing context into weights. Critically, it uses META-LEARNING at training time to improve the initialization for test-time learning. This is EXACTLY our MAML + TTT idea.
- **Results:** For 3B models trained with 164B tokens, TTT-E2E scales with context length like full-attention Transformers while Mamba 2 and Gated DeltaNet do not. 2.7x faster than full attention at 128K context.
- **Relevance to us:** CRITICAL. This paper validates our core hypothesis. They use meta-learning to optimize the initialization for test-time gradient steps. The difference: they do it for long context; we want to do it for compression/bpb improvement on any document.
- **Feasibility:** MEDIUM-HIGH. The meta-learning outer loop adds complexity but is well-understood (it's essentially MAML). The main challenge is fitting within 16MB and 10min training budget.

### 1.3 Test-Time Training on Nearest Neighbors (TTT-NN)
- **Title:** "Test-Time Training on Nearest Neighbors for Large Language Models"
- **Authors:** Moritz Hardt, Yu Sun
- **Venue:** ICLR 2024
- **Key idea:** For each test input, retrieve nearest neighbors from training set, fine-tune model on their text for just 1 gradient iteration, then predict. Dramatically simple but effective.
- **Results:** Training on just 20-50 neighbors for 1 gradient step reduces bits-per-byte by 20%. Narrows the gap between GPT-2 (117M) and GPT-Neo (1.3B) -- a 10x size difference nearly closed by TTT.
- **Relevance to us:** VERY HIGH. The 20% bpb reduction from TTT is enormous. If we could achieve even half of that, going from 1.145 to ~0.97 bpb would be a massive win. BUT: we can't do retrieval in the competition (no external data at eval time). However, we CAN do TTT on the test document itself (self-supervised next-token prediction on context).
- **Feasibility:** HIGH. The simplicity is the point -- 1 gradient step on nearest neighbors. We adapt this: instead of neighbors, use the test document's own context as self-supervised signal.

### 1.4 Test-Time Learning for LLMs (TLM)
- **Title:** "Test-Time Learning for Large Language Models"
- **Authors:** (Multiple groups)
- **Venue:** ICML 2025 Poster
- **Key idea:** TTL paradigm dynamically adapts LLMs to target domains using only unlabeled test data during testing. Uses LoRA for lightweight adaptation to prevent catastrophic forgetting.
- **Results:** Improves performance by at least 20% compared to original LLMs on domain knowledge adaptation.
- **Relevance to us:** HIGH. Using LoRA-style low-rank adaptation at test time is more parameter-efficient than full fine-tuning. Could be the right balance for our 16MB constraint.
- **Feasibility:** HIGH. LoRA is well-understood and easy to implement.

---

## 2. Meta-Learning for Fast Adaptation (MAML and Variants)

### 2.1 MAML-en-LLM
- **Title:** "MAML-en-LLM: Model Agnostic Meta-Training of LLMs for Improved In-Context Learning"
- **Authors:** Amazon Science team
- **Venue:** KDD 2024
- **ArXiv:** 2405.11446
- **Key idea:** Applies classical MAML (with second-order gradients) to meta-train LLMs for improved few-shot adaptation. Unlike MetaICL/MetaICT that skip the two-step optimization, MAML-en-LLM benefits from gradient-of-gradients guiding meta-updates.
- **Results:** Average 2% increase on unseen domains, 4% improvement on adaptation. Outperforms meta-training baselines especially in limited data settings.
- **Relevance to us:** DIRECT. This proves MAML works for LLMs. We want to use the same principle but optimize for bpb reduction rather than task accuracy.
- **Feasibility:** MEDIUM. Second-order MAML is expensive (requires gradient-of-gradients). May need to use first-order approximation (Reptile/FOMAML) to fit in 10 minutes.

### 2.2 Reptile (First-Order MAML)
- **Title:** "On First-Order Meta-Learning Algorithms" (Reptile)
- **Authors:** Alex Nichol, Joshua Achiam, John Schulman (OpenAI)
- **Venue:** ArXiv 1803.02999 (2018)
- **Key idea:** Reptile repeatedly samples a task, performs k steps of SGD on it, then moves the initialization toward the resulting parameters. No second-order gradients needed. Mathematically similar to first-order MAML but simpler to implement.
- **Results:** Similar performance to MAML on Omniglot and Mini-ImageNet.
- **Relevance to us:** HIGH. Reptile is the practical choice for our 10-minute training budget. Train by: (1) sample a document chunk from training set, (2) do k gradient steps on it, (3) move initialization toward the result. At test time: do the same k gradient steps on the test document.
- **Feasibility:** VERY HIGH. Can implement in ~50 lines of PyTorch. The key question is whether Reptile converges well enough in 10 minutes.

### 2.3 Meta-Learning the Difference
- **Title:** "Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation"
- **Authors:** Mosbach et al.
- **Venue:** TACL 2023
- **Key idea:** Prepares LLMs for efficient adaptation by meta-learning which parameters should change and by how much during few-shot adaptation.
- **Relevance to us:** MEDIUM. The idea of learning *what to adapt* (not just the initialization) could help us be more selective in TTT -- only update the most impactful parameters at test time.
- **Feasibility:** MEDIUM. Requires an additional "mask" or "scaling" mechanism.

---

## 3. Bayesian Online Prediction / Universal Compression

### 3.1 Language Modeling Is Compression
- **Title:** "Language Modeling Is Compression"
- **Authors:** Gregoire Deletang, Anian Ruoss, et al. (Google DeepMind)
- **Venue:** ICLR 2024
- **Key idea:** Formally establishes the equivalence between language modeling (prediction) and lossless compression via arithmetic coding. Shows that LLMs are powerful general-purpose compressors. Chinchilla 70B compresses ImageNet patches to 43.4% (beating PNG at 58.5%).
- **Results:** Larger models compress better on larger test sets but worse on small test sets (scaling laws depend on test set size). The compression viewpoint provides insights into tokenization and in-context learning.
- **Relevance to us:** FOUNDATIONAL. This paper is the theoretical backbone of Parameter Golf. Our bpb metric IS compression. The insight about test-set-size dependence is critical: with adaptive (TTT) models, we effectively have a different model per document, which should help on the per-document compression.
- **Feasibility:** N/A (theoretical framework, not an algorithm).

### 3.2 In-Context Learning Through Bayesian Model Averaging
- **Title:** "In-Context Learning Through the Bayesian Lens"
- **Authors:** Multiple groups
- **Venue:** ICLR 2024
- **Key idea:** ICL implicitly implements Bayesian model averaging. Analysis from an online learning perspective establishes O(1/T) regret bound where T is input sequence length. This means transformers already do something like online adaptation through their attention mechanism.
- **Relevance to us:** MEDIUM. Suggests that even without explicit TTT, the model's in-context learning is doing a form of online prediction. Our TTT approach would be COMPLEMENTARY -- adapting weights where ICL adapts attention patterns.
- **Feasibility:** N/A (theoretical insight).

### 3.3 Context Tree Weighting (CTW) and Bayesian Context Trees
- **Title:** "Bayesian Context Trees: Modelling and Exact Inference for Discrete Time Series"
- **Authors:** Kontoyiannis et al.
- **Venue:** JRSS-B 2022, with follow-up Bayesian Analysis 2024
- **Key idea:** CTW is a classic lossless compression algorithm offering both theoretical guarantees and good practical performance. Recent extensions include continuous CTW for real-valued time series and posterior sampling via MCMC.
- **Results:** CTW achieves near-optimal compression rates with linear time complexity. Neural variants could potentially combine the theoretical guarantees with learned representations.
- **Relevance to us:** LOW-MEDIUM. CTW is elegant but fundamentally limited to Markov-chain-like dependencies. Neural models vastly outperform CTW on natural language. However, the PRINCIPLE of Bayesian model averaging at different context lengths is valuable -- we could ensemble TTT-adapted models at different adaptation depths.
- **Feasibility:** LOW for direct use, MEDIUM for the ensemble principle.

---

## 4. Learned Quantization / Neural Compression

### 4.1 GPTQ: Accurate Post-Training Quantization
- **Title:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- **Authors:** Frantar, Ashkboos, Hoefler, Alistarh
- **Venue:** ICLR 2023
- **ArXiv:** 2210.17323
- **Key idea:** One-shot weight quantization using approximate second-order (Hessian) information. Quantizes layer-by-layer, compensating for quantization error of each weight in remaining weights using the Hessian.
- **Results:** Quantizes 175B models to 3-4 bits in ~4 GPU hours with negligible accuracy loss.
- **Relevance to us:** HIGH. If we train a larger model and then GPTQ-compress to 16MB, we might get better bpb than training a small model directly. The key question is: does quantization-then-TTT work? (TTT operates in quantized space.)
- **Feasibility:** HIGH. GPTQ is well-implemented in multiple libraries.

### 4.2 AWQ: Activation-Aware Weight Quantization
- **Title:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- **Authors:** Lin, Tang, et al. (MIT Han Lab)
- **Venue:** MLSys 2024 (Best Paper Award)
- **Key idea:** Not all weights are equally important. AWQ identifies salient weight channels (by activation magnitude) and protects them from quantization, applying per-channel scaling before quantization.
- **Results:** Significant speedup over GPTQ with similar or better quality. State-of-the-art for 4-bit quantization.
- **Relevance to us:** HIGH. AWQ's insight about protecting salient weights maps directly to our problem: when quantizing to fit 16MB, protect the weights most important for compression.
- **Feasibility:** HIGH. Well-supported in open-source tools.

### 4.3 QuIP# and QTIP: Extreme Quantization
- **Title:** "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks" + "QTIP: Quantization with Trellises and Incoherence Processing"
- **Authors:** Cornell RelaxML group (Tseng, Chee, et al.)
- **Venue:** QuIP# (2024), QTIP at NeurIPS 2024 (Spotlight)
- **Key idea:** Make weights incoherent (approximately i.i.d. Gaussian) via random Hadamard transform, then quantize. QTIP uses trellis coded quantization (TCQ) for ultra-high-dimensional quantization with linear cost in dimension (vs exponential for VQ codebooks).
- **Results:** QuIP# achieves Llama 2 70B at 2 bits with PPL 4.16. QTIP is state-of-the-art, >3x faster than unquantized while achieving best quality at 2-bit.
- **Relevance to us:** VERY HIGH. 2-bit quantization means we could fit a model with 64M parameters in 16MB (64M * 2 bits = 16MB). That's a substantial model. Combined with weight sharing (effective depth multiplication), this could be very powerful.
- **Feasibility:** MEDIUM. QTIP is complex to implement from scratch. QuIP# is more accessible. For the competition, we likely want QAT (quantization-aware training) rather than PTQ.

### 4.4 AQLM: Additive Quantization of Language Models
- **Title:** "Extreme Compression of Large Language Models via Additive Quantization"
- **Authors:** Egiazarian et al.
- **Venue:** ICML 2024
- **ArXiv:** 2401.06118
- **Key idea:** Uses additive (multi-codebook) quantization where each weight vector is approximated as a sum of entries from multiple learned codebooks. First algorithm to achieve Pareto-optimality at <3 bits per parameter.
- **Results:** Llama 2 70B at ~2 bits: PPL 3.94 (better than QuIP# at 4.16).
- **Relevance to us:** HIGH. Additive codebook quantization could be extremely efficient for our constraint. We train at full precision, then compress using learned codebooks. The codebooks themselves become part of the 16MB artifact.
- **Feasibility:** MEDIUM. Requires implementing the multi-codebook quantization scheme, but the math is straightforward.

### 4.5 BitNet: 1.58-bit Language Models
- **Title:** "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
- **Authors:** Ma et al. (Microsoft Research)
- **Venue:** ArXiv 2402.17764 (2024), JMLR 2025
- **Key idea:** Train from scratch with ternary weights {-1, 0, +1} using absmean quantization in the forward pass. Replaces nn.Linear with BitLinear. NOT post-training quantization -- the model learns to work with ternary weights during training.
- **Results:** BitNet b1.58 matches half-precision Transformer LLM in both perplexity and end-task performance at same model size. 2B parameter model trained on 4T tokens released open-source.
- **Relevance to us:** VERY HIGH. At 1.58 bits per weight, we could fit ~80M parameters in 16MB. Combined with weight sharing (3 blocks recycled 4x = 12 effective layers, which we're already doing), this is very promising.
- **Feasibility:** HIGH. BitLinear is simple to implement. The key question is whether it converges well enough in 10 minutes of training.

### 4.6 Finite Scalar Quantization (FSQ)
- **Title:** "Finite Scalar Quantization: VQ-VAE Made Simple"
- **Authors:** Google Research
- **Venue:** ICLR 2024
- **Key idea:** Replace vector quantization with simple scalar quantization (round each dimension independently to a small set of values). Achieves ~100% codebook utilization (vs VQ-VAE's codebook collapse problems).
- **Results:** Matches or exceeds VQ-VAE quality with much simpler implementation.
- **Relevance to us:** MEDIUM. Could apply to weight quantization: instead of complex VQ schemes, simply round weights to a small number of levels per dimension.
- **Feasibility:** VERY HIGH. Trivially simple to implement.

---

## 5. Minimum Description Length (MDL) and Language Models

### 5.1 Compressibility Measures Complexity (Singular MDL)
- **Title:** "Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory"
- **Authors:** Timaeus Research
- **Venue:** ArXiv 2510.12077 (2025)
- **Key idea:** Derives singular MDL principle where asymptotic redundancy involves the local learning coefficient (LLC). For neural networks, degeneracy (not curvature/Hessian) is the leading-order contribution to compressibility. Experiments on Pythia suite show LLC is linearly correlated with compressibility.
- **Results:** Provides theoretical framework for understanding limits of model compression. LLC predicts how much a model can be compressed without quality loss.
- **Relevance to us:** MEDIUM-HIGH (theoretical). Tells us that degenerate (weight-shared) models might be inherently more compressible, which validates our ALBERT-style approach. Also suggests there's a theoretical limit to how small we can go.
- **Feasibility:** LOW (theory paper, not directly implementable).

### 5.2 MDL Regularization for Neural Networks
- **Title:** "A Minimum Description Length Approach to Regularization in Neural Networks"
- **Authors:** Various
- **Venue:** ArXiv 2505.13398 (2025)
- **Key idea:** MDL-based regularization avoids overfitting by balancing data fit with model complexity. Prevents memorization by imposing simplicity requirement.
- **Relevance to us:** MEDIUM. Could inform our training objective: instead of pure cross-entropy, add an MDL-style regularizer that encourages compressible weights.
- **Feasibility:** MEDIUM. Adds a regularization term to training.

### 5.3 Bridging Kolmogorov Complexity and Deep Learning
- **Title:** "Bridging Kolmogorov Complexity and Deep Learning"
- **Authors:** Various
- **Venue:** ArXiv 2509.22445 (2025)
- **Key idea:** Formalizes complexity-error tradeoffs for neural networks. Neural scaling laws demonstrate predictable relationships between model size and performance.
- **Relevance to us:** MEDIUM. Provides theoretical grounding for optimal model size given our 16MB constraint.
- **Feasibility:** LOW (theoretical).

---

## 6. Efficient Architecture and Training

### 6.1 Relaxed Recursive Transformers
- **Title:** "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA"
- **Authors:** Sridhar et al.
- **Venue:** ArXiv 2410.20672 (2024)
- **Key idea:** Share parameters across layers but add per-layer LoRA modules for differentiation. A single block of unique layers repeated in a loop, with tiny LoRA adapters providing layer-specific behavior.
- **Results:** Minimal performance loss compared to non-shared models. Significant parameter reduction.
- **Relevance to us:** VERY HIGH. We're already doing ALBERT-style weight sharing (3 blocks x 4 recycles). Adding per-recycle LoRA adapters could improve quality without much parameter cost. Each LoRA adapter is tiny (rank 4-8).
- **Feasibility:** VERY HIGH. LoRA is trivial to add. Can implement in an afternoon.

### 6.2 Basis Sharing: Cross-Layer Parameter Sharing
- **Title:** "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression"
- **Authors:** Various
- **Venue:** ICLR 2025
- **Key idea:** Represent weights as linear combinations of shared basis vectors + unique coefficients. Unlike ALBERT (identical weights), this preserves layer-specific behavior while sharing most parameters.
- **Results:** Better compression-quality tradeoff than naive weight sharing.
- **Relevance to us:** HIGH. Instead of sharing identical blocks, share basis vectors and learn small per-layer coefficients. More parameter-efficient than per-layer LoRA.
- **Feasibility:** HIGH. Clean mathematical formulation, straightforward to implement.

### 6.3 Mixture-of-Recursions (MoR)
- **Title:** "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"
- **Authors:** Various
- **Venue:** ArXiv 2507.10524 (2025)
- **Key idea:** Train lightweight routers to assign token-specific recursion depths. Each token gets as many passes through the shared block as it needs. Ties weights to cut parameters, routes tokens to cut FLOPs, and caches KV recursion-wise.
- **Results:** Simultaneously reduces parameters, FLOPs, and memory.
- **Relevance to us:** MEDIUM. Interesting but complex. The routing adds overhead that may not be worth it for our scale.
- **Feasibility:** MEDIUM. Router adds architectural complexity.

### 6.4 Byte Latent Transformer (BLT)
- **Title:** "Byte Latent Transformer: Patches Scale Better Than Tokens"
- **Authors:** Meta AI
- **Venue:** ACL 2025 (ArXiv 2412.09871)
- **Key idea:** Tokenizer-free architecture that groups bytes into dynamic patches based on entropy. Allocates more compute to complex byte sequences. No fixed vocabulary.
- **Results:** Matches Llama 3 at scale while using up to 50% fewer inference FLOPs.
- **Relevance to us:** MEDIUM-HIGH. Since Parameter Golf uses bits-per-byte (tokenizer-agnostic), a byte-level model avoids the "tokenizer tax." Our current 1024-token vocabulary is tiny; a byte-level approach (256 "tokens") with dynamic patching could be more efficient.
- **Feasibility:** MEDIUM. Requires significant architecture changes from our current GPT baseline.

### 6.5 Muon Optimizer
- **Title:** "Muon is Scalable for LLM Training"
- **Authors:** Keller Jordan et al.
- **Venue:** ArXiv 2502.16982 (2025)
- **Key idea:** Orthogonalizes momentum using Newton-Schulz iteration. Approximately 2x compute efficiency over AdamW. Requires only ~52% of training FLOPs for comparable performance.
- **Results:** 2x compute efficiency. Used to train 3B/16B MoE model (Moonlight) with 5.7T tokens.
- **Relevance to us:** VERY HIGH. In a 10-minute training budget, 2x efficiency is huge. Effectively doubles our training compute.
- **Feasibility:** VERY HIGH. Drop-in optimizer replacement. Implementations available on GitHub.

### 6.6 Compute-Optimal QAT
- **Title:** "Compute-Optimal Quantization-Aware Training"
- **Authors:** Various
- **Venue:** ArXiv 2509.22935 (2025), ICLR 2026
- **Key idea:** Derives scaling laws predicting optimal QAT-to-FP training ratio and final model performance across different bit widths. The optimal QAT fraction increases with total compute.
- **Results:** Novel cooldown + QAT fusion eliminates redundant FP updates, achieving significant compute savings.
- **Relevance to us:** HIGH. Tells us how to split our 10 minutes between full-precision training and QAT. Likely: train FP for 7 minutes, QAT for 3 minutes.
- **Feasibility:** HIGH. QAT is standard; the scaling law just tells us the optimal split.

### 6.7 BabyLM Challenge Findings
- **Title:** "Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora"
- **Authors:** Warstadt, Mueller et al.
- **Venue:** CoNLL-BabyLM 2024
- **ArXiv:** 2412.05149
- **Key idea:** Challenge for sample-efficient LM training on small data. Best 2024 submission: hybrid causal-masked LM architecture. Key lesson: combined changes to data, training objective, AND architecture work best.
- **Results:** Models trained on child-scale data (~100M words) can approach or exceed models trained on billions of tokens.
- **Relevance to us:** MEDIUM. Different constraint (data-limited vs compute-limited) but same spirit. The hybrid causal-masked objective is interesting -- could improve per-token learning efficiency.
- **Feasibility:** MEDIUM. Would require changing training objective.

---

## 7. Test-Time Compute Scaling

### 7.1 Scaling Test-Time Compute Survey
- **Title:** "A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?"
- **Authors:** Chen et al.
- **Venue:** ArXiv 2503.24235 (2025)
- **Key idea:** Comprehensive framework for test-time compute scaling along four dimensions. Key finding: scaling inference compute with strategies can be MORE efficient than scaling model parameters.
- **Results:** Smaller models + advanced inference algorithms offer Pareto-optimal tradeoffs.
- **Relevance to us:** HIGH. Validates our approach: a small 16MB model + TTT at inference can beat a larger static model.
- **Feasibility:** N/A (survey/framework).

---

## Synthesis: Recommended Strategy

Based on this literature review, here is the recommended approach combining the highest-impact, most feasible ideas:

### Core Architecture (Weekend 1)
1. **ALBERT-style weight sharing** with 3 wide blocks x 4 recycles = 12 effective layers (already implemented)
2. **Per-recycle LoRA adapters** (from Relaxed Recursive Transformers) for layer differentiation
3. **Muon optimizer** for 2x training efficiency
4. **BitNet/QAT training** with ternary or 2-bit weights to maximize parameters in 16MB

### Meta-Learning + TTT (Weekend 2)
5. **Reptile-style meta-learning** during training: sample document chunks, do k inner steps, update initialization
6. **Test-time training** at eval: for each test document, do 1-5 gradient steps of next-token prediction on preceding context before scoring
7. **LoRA-only TTT**: only adapt the LoRA parameters at test time (tiny, fast, prevents catastrophic forgetting)

### Quantization Pipeline (Weekend 2)
8. **QAT with learned codebooks** for final compression to 16MB
9. **AWQ-style salient weight protection** during quantization

### Expected Impact
- Muon optimizer: ~0.02-0.03 bpb improvement (more efficient training)
- Recursive LoRA: ~0.01-0.02 bpb improvement (better layer differentiation)
- BitNet/extreme quantization: enables larger model in 16MB, ~0.03-0.05 bpb
- Meta-learned TTT: potentially ~0.05-0.10 bpb (based on TTT-NN's 20% reduction, scaled down for self-supervised TTT without retrieval)
- **Total estimated improvement: 0.11-0.20 bpb, reaching ~0.95-1.05 bpb target**

### Key Risk
The main risk is that TTT at evaluation time may not be permitted by competition rules, or that the adaptation overhead exceeds the evaluation time budget. Need to verify the rules carefully. If TTT is not allowed, the remaining improvements (Muon + Recursive LoRA + BitNet) still yield ~0.06-0.10 bpb improvement, reaching ~1.05-1.09 bpb.

---

## Sources

- [TTT Layers - Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620)
- [TTT-E2E - End-to-End Test-Time Training](https://arxiv.org/abs/2512.23675)
- [TTT-NN - Test-Time Training on Nearest Neighbors](https://openreview.net/forum?id=CNL2bku4ra)
- [Test-Time Learning for LLMs (ICML 2025)](https://icml.cc/virtual/2025/poster/44367)
- [MAML-en-LLM (KDD 2024)](https://arxiv.org/abs/2405.11446)
- [Reptile - First-Order Meta-Learning](https://arxiv.org/abs/1803.02999)
- [Meta-Learning the Difference (TACL 2023)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00517/113851)
- [Language Modeling Is Compression (ICLR 2024)](https://arxiv.org/abs/2309.10668)
- [In-Context Learning as Bayesian Inference (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/d81cd83e7f6748af351485d73f305483-Paper-Conference.pdf)
- [GPTQ (ICLR 2023)](https://arxiv.org/abs/2210.17323)
- [AWQ (MLSys 2024 Best Paper)](https://github.com/mit-han-lab/llm-awq)
- [QuIP# - Lattice Codebook Quantization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12395268/)
- [QTIP - Trellis Coded Quantization (NeurIPS 2024 Spotlight)](https://arxiv.org/abs/2406.11235)
- [AQLM - Additive Quantization (ICML 2024)](https://arxiv.org/abs/2401.06118)
- [BitNet b1.58 - Ternary Weight LLMs](https://arxiv.org/abs/2402.17764)
- [Finite Scalar Quantization (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf)
- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672)
- [Basis Sharing (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
- [Byte Latent Transformer (ACL 2025)](https://arxiv.org/abs/2412.09871)
- [Muon Optimizer](https://arxiv.org/abs/2502.16982)
- [Compute-Optimal QAT (ICLR 2026)](https://arxiv.org/abs/2509.22935)
- [BabyLM Challenge 2024 Findings](https://arxiv.org/abs/2412.05149)
- [Singular MDL (Timaeus 2025)](https://arxiv.org/abs/2510.12077)
- [Kolmogorov Complexity and Deep Learning](https://arxiv.org/abs/2509.22445)
- [Test-Time Scaling Survey (2025)](https://arxiv.org/abs/2503.24235)
- [OpenAI Parameter Golf](https://openai.com/index/parameter-golf/)
- [Parameter Golf GitHub](https://github.com/openai/parameter-golf/)
