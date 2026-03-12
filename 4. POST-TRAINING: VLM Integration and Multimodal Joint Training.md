# Seedream 3.0 → 4.0: A NeurIPS/CVPR/ICML-Grade Technical Comparison

---

## 1. Framing and Scope

The transition from Seedream 3.0 to 4.0 is not an incremental refinement but a **paradigm shift**: from a specialist text-to-image (T2I) DiT to a unified multimodal generative system in which a VLM and a DiT are jointly optimized across generation and editing tasks simultaneously. This analysis proceeds through six technical axes—architecture, training objectives, data pipeline, post-training alignment, inference acceleration, and benchmark methodology—before synthesizing the multi-task learning theory underpinning the joint training claim.

---

## 2. Architecture

### 2.1 Seedream 3.0: Dual-Encoder DiT Backbone

Seedream 3.0 is built on a Diffusion Transformer (DiT) with a **dual-encoder text conditioning stack**:

- A **bilingual CLIP** model (Chinese + English), producing dense token embeddings for low-level semantic grounding.
- A **large language model encoder** (from the Seed series), producing autoregressive contextual embeddings for long-form, compositional prompts.

The two streams are fused via cross-attention within each DiT block. Formally, letting $\mathbf{c}_{\text{clip}} \in \mathbb{R}^{L_1 \times d_1}$ and $\mathbf{c}_{\text{llm}} \in \mathbb{R}^{L_2 \times d_2}$ denote the respective sequence embeddings, the conditioning signal $\mathbf{c}$ fed to each DiT block is:

$$\mathbf{c} = \text{Proj}_1(\mathbf{c}_{\text{clip}}) \oplus \text{Proj}_2(\mathbf{c}_{\text{llm}})$$

where $\oplus$ denotes concatenation along the sequence dimension after projection to a shared hidden dimension $d$. Cross-attention then computes:

$$\text{Attn}(\mathbf{z}, \mathbf{c}) = \text{softmax}\!\left(\frac{(\mathbf{z}\mathbf{W}_Q)(\mathbf{c}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right)\mathbf{c}\mathbf{W}_V$$

where $\mathbf{z}$ is the latent patch sequence. This dual-conditioning is motivated by the complementary representational biases of contrastive and generative text encoders: CLIP provides grounded visual-semantic alignment while the LLM encoder provides long-horizon syntactic structure.

**Native resolution generation** is a key Seedream 3.0 capability: the model directly generates at any resolution in $[512^2, 2048^2]$ without a separate upsampler or Refiner stage. This is achieved by training with variable aspect-ratio packing and RoPE-based positional embeddings that generalize across spatial scales.

### 2.2 Seedream 4.0: Causal Diffusion DiT with Multimodal Input

Seedream 4.0's single most consequential architectural change is the introduction of **causal diffusion design** in the DiT. To understand this precisely, we must distinguish the standard DiT attention pattern from the causal variant.

**Standard DiT attention** treats all patch tokens symmetrically — every token attends to every other token (full bidirectional attention). The attention mask $\mathbf{M}$ is the all-ones matrix:

$$\mathbf{M}_{ij} = 1 \quad \forall i,j$$

**Causal diffusion attention** partitions the token sequence into a **reference image segment** $\mathcal{R}$ and a **target image segment** $\mathcal{T}$, and applies an asymmetric mask:

$$\mathbf{M}_{ij} = \begin{cases} 1 & \text{if } i \in \mathcal{T},\; j \in \mathcal{R} \cup \mathcal{T} \\ 1 & \text{if } i \in \mathcal{R},\; j \in \mathcal{R} \\ 0 & \text{if } i \in \mathcal{R},\; j \in \mathcal{T} \end{cases}$$

This means: **(a)** target tokens can attend to all reference tokens (edit-aware generation), **(b)** reference tokens can attend only to other reference tokens (reference is treated as a clean, conditioning-only context), and **(c)** target tokens can attend to each other (standard bidirectional generation within the output).

The causal asymmetry is the enabling mechanism for **joint T2I + editing in a single forward pass**. During T2I, $\mathcal{R} = \emptyset$ and the model degenerates to standard bidirectional generation. During editing, $\mathcal{R}$ holds the (noisy or clean) reference image tokens, and the DiT propagates reference structure into $\mathcal{T}$ during denoising. The elegance is that **no architectural bifurcation is required**—the same DiT serves both modalities.

Formally, the denoising objective in the editing case can be written as:

$$\mathcal{L}_{\text{edit}} = \mathbb{E}_{t, \mathbf{x}_0^{\mathcal{T}}, \mathbf{x}_0^{\mathcal{R}}, \boldsymbol{\epsilon}} \left\| \boldsymbol{\epsilon}_\theta\!\left(\mathbf{x}_t^{\mathcal{T}}, \mathbf{x}_0^{\mathcal{R}}, t, \mathbf{c}_{\text{edit}}\right) - \boldsymbol{\epsilon} \right\|^2$$

where $\mathbf{x}_0^{\mathcal{R}}$ enters at noise level $t=0$ (clean) and $\mathbf{x}_t^{\mathcal{T}}$ is noised. The causal mask ensures the reference segment is never corrupted by gradient flow from the target's denoising, which is the training-time analogue of inference-time conditioning.

**Comparison with InstructPix2Pix and IP-Adapter approaches:** Prior editing-capable diffusion models either (a) concatenate reference and noisy target in the channel dimension (InstructPix2Pix — doubles input channels, incompatible with T2I without zeroing the reference channels), or (b) inject reference as a cross-attention bias (IP-Adapter — decoupled from main attention, weaker structural transfer). The causal attention mask is architecturally cleaner: it operates within the *same* attention head pool, allowing learned heads to specialize in reference-to-target transfer, and it is trivially disabled for T2I.

### 2.3 VLM Integration: From Conditioning to Co-Processor

In 3.0, the LLM is used **only as a text encoder** — its weights are frozen post-pretraining and contribute no generation-time reasoning. In 4.0, the **Prompt Engineering (PE) module** is a fully fine-tuned VLM (derived from Seed1.5-VL) that operates as an intelligent pre-processor. The system pipeline changes from:

$$\text{Prompt} \xrightarrow{\text{CLIP}+\text{LLM enc.}} \mathbf{c} \xrightarrow{\text{DiT}} \mathbf{x}_0$$

to:

$$\text{Prompt} \xrightarrow{\text{VLM}_{PE}} (\mathbf{c}_{\text{rewritten}}, \mathbf{c}_{\text{ref}}, \mathbf{c}_{\text{tgt}}, r^*) \xrightarrow{\text{DiT}} \mathbf{x}_0$$

where $r^*$ is the predicted optimal aspect ratio. This is not merely a preprocessing step but a **learned semantic interface** — the VLM can engage in chain-of-thought reasoning before producing the conditioning signal, which has no analogue in 3.0.

---

## 3. Training Objectives

### 3.1 Seedream 3.0: Flow Matching with Resolution-Aware Noise Schedule

Seedream 3.0 adopts **Rectified Flow** (Liu et al., 2022) as its base continuous-time objective. The forward process is a linear interpolation:

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), \quad t \in [0,1]$$

The velocity field target is constant: $\mathbf{v}^* = \boldsymbol{\epsilon} - \mathbf{x}_0$, and the model is trained to minimize:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\boldsymbol{\epsilon} - \mathbf{x}_0) \right\|^2$$

A critical subtlety is the **resolution-dependent noise schedule**: at high resolutions, the effective signal-to-noise ratio (SNR) of a noisy latent differs from low resolution because spatial frequency content scales with resolution. Seedream 3.0 applies a log-SNR shift that is a function of the token count $N$ (number of patches):

$$\lambda_{\text{shifted}}(t) = \lambda(t) + \log\!\left(\frac{N_{\text{base}}}{N}\right)$$

where $\lambda(t) = \log\!\left(\frac{(1-t)^2}{t^2}\right)$ for rectified flow and $N_{\text{base}}$ is the reference resolution (e.g., $512^2 / p^2$ for patch size $p$). This shift compensates for the fact that at higher resolutions, high-frequency components (which carry most of the image content energy) are corrupted earlier in the forward process. Without this correction, high-resolution generation would train predominantly on overdestroyed latents, leading to blurry outputs.

### 3.2 Seedream 4.0: Unified Multimodal Objective and Causal Flow Matching

Seedream 4.0 trains a single model jointly on T2I and editing via a combined objective. Let $\mathcal{D}_{T2I}$ and $\mathcal{D}_{\text{edit}}$ denote the respective training distributions. The total loss is:

$$\mathcal{L}_{\text{total}} = \mathbb{E}_{(\mathbf{x}_0, \mathbf{c}) \sim \mathcal{D}_{T2I}} \mathcal{L}_{\text{FM}}(\mathbf{x}_0, \mathbf{c}) + \lambda \cdot \mathbb{E}_{(\mathbf{x}_0^{\mathcal{T}}, \mathbf{x}_0^{\mathcal{R}}, \mathbf{c}) \sim \mathcal{D}_{\text{edit}}} \mathcal{L}_{\text{edit}}(\mathbf{x}_0^{\mathcal{T}}, \mathbf{x}_0^{\mathcal{R}}, \mathbf{c})$$

The $\lambda$ hyperparameter controls the task mixing ratio and is likely annealed during training (higher $\lambda$ early for editing initialization, then balanced). During the editing loss, only the target segment's denoising error is backpropagated:

$$\mathcal{L}_{\text{edit}} = \mathbb{E}_{t, \boldsymbol{\epsilon}} \left\| \mathbf{v}_\theta(\mathbf{x}_t^{\mathcal{T}}, \mathbf{x}_0^{\mathcal{R}}, t, \mathbf{c}_{\text{edit}}) - (\boldsymbol{\epsilon} - \mathbf{x}_0^{\mathcal{T}}) \right\|^2$$

The reference segment contributes no denoising loss — it acts as a frozen conditioning signal through the causal attention path. This is analogous to treating $\mathbf{x}_0^{\mathcal{R}}$ as a second conditioning variable alongside $\mathbf{c}$, but encoded spatially rather than semantically.

---

## 4. Data Pipeline

### 4.1 Seedream 3.0: Scale, Bilingual Curation, and Quality Filtering

Seedream 3.0's data pipeline emphasizes **scale and linguistic coverage**:

- A large-scale crawled corpus covering both Chinese and English web images.
- Multi-stage quality filtering: aesthetic scoring, NSFW filtering, deduplication, and resolution-based cutoffs.
- **Re-captioning** using a VLM to generate dense, structured captions replacing original alt-text, improving text-image alignment for training.

The captioning strategy distinguishes domain-specific aesthetic attributes. For art/photography images, specialist captioners are trained to annotate **style, lighting, composition, and palette** — these captions enter the SFT stage to improve aesthetic controllability, which is not achievable with generic BLIP/LLaVA-style captions that tend toward object-level description.

### 4.2 Seedream 4.0: Triplet Construction and Three-Granularity Augmentation

Seedream 4.0's most distinctive data engineering innovation is the construction of **editing triplets** $(\mathbf{x}_0^{\mathcal{R}}, \mathbf{x}_0^{\mathcal{T}}, \mathbf{c}_{\text{inst}})$ and the attachment of **three caption granularities** to each:

1. $\mathbf{c}_{\text{ref}}$: A dense description of the reference image (what is in it).
2. $\mathbf{c}_{\text{tgt}}$: A dense description of the target/edited image.
3. $\mathbf{c}_{\text{inst}}$: The editing instruction (what transformation to apply).

Formally, the conditioning triple $(\mathbf{c}_{\text{ref}}, \mathbf{c}_{\text{tgt}}, \mathbf{c}_{\text{inst}})$ spans a higher-dimensional semantic space than any single caption. This acts as **structured data augmentation** in the following sense: at training time, any subset $\{(\mathbf{c}_{\text{inst}})\}$, $\{(\mathbf{c}_{\text{tgt}})\}$, or the full triple can be used as conditioning, creating multiple training samples from each triplet. This is analogous to dropout-based augmentation in vision models, but operating in semantic conditioning space.

The three-granularity scheme also **improves classifier-free guidance (CFG) during inference**. With standard CFG, the unconditional score estimate $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ is obtained by dropping all conditioning. With triple conditioning, partial dropping (e.g., retaining $\mathbf{c}_{\text{ref}}$ while dropping $\mathbf{c}_{\text{inst}}$) enables fine-grained guidance interpolation:

$$\mathbf{v}_{\text{guided}} = \mathbf{v}_\theta(\mathbf{x}_t | \emptyset) + w_1 [\mathbf{v}_\theta(\mathbf{x}_t | \mathbf{c}_{\text{ref}}) - \mathbf{v}_\theta(\mathbf{x}_t | \emptyset)] + w_2 [\mathbf{v}_\theta(\mathbf{x}_t | \mathbf{c}_{\text{ref}}, \mathbf{c}_{\text{inst}}) - \mathbf{v}_\theta(\mathbf{x}_t | \mathbf{c}_{\text{ref}})]$$

This decomposes guidance into a **reference fidelity term** ($w_1$) and an **instruction adherence term** ($w_2$), giving inference-time control over the fidelity-editability tradeoff without retraining.

**Triplet sourcing**: editing triplets come from (a) synthetic generation using existing editing models to produce reference-target pairs with known instructions, (b) video frame extraction (consecutive frames as reference-target with optical flow–derived edit descriptions), and (c) human-curated image editing demonstrations. The diversity of triplet sources prevents the model from overfitting to any single editing modality.

---

## 5. Post-Training Pipeline: Full Comparative Analysis

### 5.1 Shared Structure: CT → SFT → RLHF → PE

Both versions share the same four-stage template:

| Stage                       | Purpose                                | 3.0                      | 4.0                                    |
| --------------------------- | -------------------------------------- | ------------------------ | -------------------------------------- |
| **CT** (Continued Training) | Domain adaptation, instruction seeding | T2I focus                | T2I + editing jointly                  |
| **SFT**                     | Demonstration learning                 | Aesthetic captions       | Consistency for ref-target + aesthetic |
| **RLHF**                    | Human preference alignment             | VLM reward (1B→20B)      | VLM reward, multimodal queries         |
| **PE**                      | Prompt preprocessing                   | Rule-based / lightweight | Full VLM (Seed1.5-VL fine-tuned)       |

### 5.2 RLHF: Reward Model Scaling Laws

**Seedream 3.0 RLHF** introduces a critical methodological advancement over CLIP-based reward: a **VLM-based generative reward model (GenRM)**.

The reward formulation is: given an image $\mathbf{x}_0$ and prompt $\mathbf{c}$, construct a query $q = $ "Does this image accurately depict: [c]? Answer Yes or No." The reward is:

$$r(\mathbf{x}_0, \mathbf{c}) = \frac{\log p_{\text{VLM}}(\text{"Yes"} \mid q, \mathbf{x}_0)}{\log p_{\text{VLM}}(\text{"Yes"} \mid q, \mathbf{x}_0) + \log p_{\text{VLM}}(\text{"No"} \mid q, \mathbf{x}_0)}$$

(or equivalently, a normalized log-probability softmax over the binary vocabulary $\{\text{"Yes"}, \text{"No"}\}$). This formulation has deep connections to **Bayesian reward inference**: $p_{\text{VLM}}(\text{"Yes"} \mid q, \mathbf{x}_0)$ is the VLM's posterior belief that the image satisfies the condition, and the normalization ensures it lies in $(0, 1)$ as a proper probability.

Why does this outperform CLIP? The CLIP reward is:

$$r_{\text{CLIP}}(\mathbf{x}_0, \mathbf{c}) = \frac{\mathbf{f}_I(\mathbf{x}_0) \cdot \mathbf{f}_T(\mathbf{c})}{\|\mathbf{f}_I\| \|\mathbf{f}_T\|}$$

CLIP's contrastive training optimizes for **discrimination** (can we tell which caption matches this image among $N$ negatives?), not for **verification** (does this image satisfy a complex compositional condition?). For prompts with spatial relations, counts, or negations, CLIP embeddings are known to saturate and fail. The VLM reward models compositional semantics through autoregressive generation, capturing these failure modes explicitly.

**Reward model scaling** from 1B → >20B parameters: the observed emergent improvement in reward quality with scale follows the now-standard reward model scaling law (Ouyang et al., 2022; Guo et al., 2025 for DeepSeek-R1):

$$\mathcal{Q}(N_r) \approx a \cdot N_r^b + c$$

where $\mathcal{Q}$ is reward model quality (e.g., correlation with human preference), $N_r$ is the reward model parameter count, and $a, b, c$ are empirically fit constants. The key implication is that **reward model quality is a bottleneck** for RLHF alignment — a generator trained against a weak reward model hits a ceiling at the reward model's expressiveness. Seedream 3.0's jump to >20B demonstrates that this bottleneck was binding at 1B.

The RL policy gradient update uses REINFORCE or a variant. For a denoising step at time $t$, the policy gradient is:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\mathbf{x}_0 \sim p_\theta}\left[r(\mathbf{x}_0, \mathbf{c}) \cdot \nabla_\theta \log p_\theta(\mathbf{x}_0 \mid \mathbf{c})\right] - \beta \cdot D_{\text{KL}}(p_\theta \| p_{\text{ref}})$$

The KL penalty against a reference (pre-RLHF) policy prevents reward hacking — without it, the model can learn to fool the reward model by generating adversarial patterns that exploit its failure modes (analogous to "reward model overoptimization" in LLM RLHF, Gao et al., 2023).

**Seedream 4.0 RLHF** extends this to multimodal queries. For editing tasks, the reward query becomes:

$$q = \text{"Given reference image } \mathbf{x}_0^{\mathcal{R}} \text{, does the edit '} c_{\text{inst}} \text{' produce the result shown? Answer Yes or No."}$$

The VLM now must jointly evaluate **(a)** fidelity to the reference (structure preservation) and **(b)** adherence to the editing instruction. This is a strictly harder reward modeling problem, which likely necessitates even larger reward models or multi-head reward decomposition.

### 5.3 PE Module: Rule-Based vs. VLM-Based Prompt Engineering

**Seedream 3.0 PE** is relatively lightweight — likely a rule-based prompt expansion system that adds quality tags, style descriptors, and resolution hints to user prompts.

**Seedream 4.0 PE** is qualitatively different: a fine-tuned **Seed1.5-VL** VLM performing four tasks:

1. **Task routing**: Classify the input as T2I, editing, in-painting, style transfer, etc. This is a discrete classification head or autoregressive prefix classification.

2. **Prompt rewriting with auto-thinking (AdaCoT)**: The VLM generates a chain-of-thought before emitting the rewritten prompt. Inspired by **Adaptive Chain-of-Thought** (AdaCoT), thinking budget is conditioned on estimated task complexity $\kappa$:

$$\text{Tokens}_{\text{thinking}} = f(\kappa), \quad f \text{ is monotone increasing}$$

For simple T2I requests ($\kappa$ low), the VLM emits a direct rewrite. For complex editing instructions with spatial constraints ($\kappa$ high), the VLM engages in explicit reasoning before writing the final prompt. This is analogous to System 1 / System 2 processing in dual-process theory.

3. **Aspect ratio estimation**: Given the semantic content of the prompt (portrait vs. landscape subject, panel vs. poster layout), the VLM predicts the optimal aspect ratio $r^* \in \mathcal{A}$ from a discrete set of supported ratios. This is learned from human preference data on image cropping and composition.

4. **Reference/target caption generation**: For editing tasks, the VLM generates $\mathbf{c}_{\text{ref}}$ and $\mathbf{c}_{\text{tgt}}$ from $(\mathbf{x}_0^{\mathcal{R}}, \mathbf{c}_{\text{inst}})$, providing the DiT with fully specified conditioning. This is a captioning task conditioned on the edit instruction, requiring the VLM to *hallucinate forward* what the edited image should contain.

The end-to-end fine-tuning of PE on task-specific data means the VLM's internal representations are optimized for DiT conditioning quality — not just natural language fluency or generic visual understanding.

---

## 6. Joint Training Theory: Multi-Task Learning and Positive Transfer

### 6.1 The Gradient Interference Problem

In multi-task learning (MTL), training task $\mathcal{T}_1$ and $\mathcal{T}_2$ on shared parameters $\theta$ can cause **gradient interference**: when $\nabla_\theta \mathcal{L}_1$ and $\nabla_\theta \mathcal{L}_2$ point in conflicting directions, gradient updates for one task impede convergence on the other.

Formally, the cosine similarity of task gradients:

$$\rho_{12} = \frac{\langle \nabla_\theta \mathcal{L}_1, \nabla_\theta \mathcal{L}_2 \rangle}{\|\nabla_\theta \mathcal{L}_1\| \cdot \|\nabla_\theta \mathcal{L}_2\|}$$

When $\rho_{12} < 0$, the tasks interfere. Empirically, task gradient interference increases when tasks require **conflicting inductive biases** or **incompatible feature representations**. The classic mitigation strategies (GradNorm, PCGrad, MGDA) project gradients to remove the conflicting component.

### 6.2 Why T2I + Editing Is a Positive Transfer Case

The Seedream 4.0 claim that joint training surpasses separate training is theoretically grounded in the following argument:

**Claim**: T2I generation and image editing share a deep latent structure — namely, the distribution over image patches conditioned on semantic content. The representations learned for T2I (mapping semantic descriptions to visual structures) are *exactly* the representations needed for editing (mapping reference patches + semantic delta to new patches).

More formally, consider the DiT's internal representation at layer $\ell$:

$$\mathbf{h}^\ell = f^\ell(\mathbf{x}_t, \mathbf{c})$$

For T2I, $\mathbf{c}$ is a text prompt. For editing, $\mathbf{c}$ is $(\mathbf{c}_{\text{inst}}, \mathbf{x}_0^{\mathcal{R}})$. The key observation is that in both cases, the model must maintain a **spatially structured semantic representation** of the target image at each denoising step. The editing task makes this requirement explicit (the reference image provides ground truth structure), which provides a **supervisory signal on the model's spatial understanding that benefits T2I generation**.

This is the multi-task learning equivalent of the "auxiliary task" literature: editing acts as an auxiliary task that regularizes the T2I representation toward better spatial-semantic consistency.

**Formal statement of positive transfer**: Let $\mathcal{L}^*_{\text{T2I, separate}}$ and $\mathcal{L}^*_{\text{T2I, joint}}$ denote the optimal T2I loss under separate and joint training. Positive transfer holds when:

$$\mathcal{L}^*_{\text{T2I, joint}} < \mathcal{L}^*_{\text{T2I, separate}}$$

The mechanism is: joint training over $\mathcal{D}_{\text{edit}}$ exposes the model to more (reference, target) image pairs with known structural relationships, which enriches the model's implicit world model of image transformations. At inference on T2I, this richer world model enables more coherent spatial layouts.

**The editing direction matters**: The causal attention design is crucial here. If editing were implemented via symmetric (bidirectional) attention between reference and target, the reference and target tokens would co-adapt, and the features learned would be specific to the editing task. The causal mask enforces that reference tokens can never see target tokens — meaning the reference encoder sub-network must produce representations that are **self-sufficient for T2I conditioning**, which are exactly the representations that transfer to T2I.

### 6.3 Evidence of Positive Transfer in Practice

The strongest empirical evidence for joint training superiority would come from an ablation:

| Model       | T2I Quality (FID/HPS) | Edit Quality (CLIP-I/DINO) |
| ----------- | --------------------- | -------------------------- |
| T2I only    | Baseline              | N/A                        |
| Edit only   | Degraded              | Baseline                   |
| Joint (4.0) | **Improved**          | **Improved**               |

Seedream 4.0 reports this pattern: the jointly trained model surpasses models trained on individual tasks, consistent with the positive transfer hypothesis.

---

## 7. Inference Acceleration

### 7.1 Seedream 3.0: Consistency-Based Distillation

Seedream 3.0 employs **Consistency Training (CT)** for inference acceleration. The consistency model distillation objective (Song et al., 2023) enforces self-consistency along ODE trajectories:

$$\mathcal{L}_{\text{CT}} = \mathbb{E}_{t, \mathbf{x}_{t_n}, \mathbf{x}_{t_{n+1}}} d\!\left(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}, \mathbf{c}),\; f_\theta^{-}(\mathbf{x}_{t_n}, t_n, \mathbf{c})\right)$$

where $d(\cdot, \cdot)$ is a distance metric (LPIPS in practice), $\mathbf{x}_{t_n}$ and $\mathbf{x}_{t_{n+1}}$ are adjacent points on the probability flow ODE, and $f_\theta^{-}$ is the exponential moving average (EMA) of $f_\theta$. This forces the model to map any point on a trajectory directly to the same endpoint $\mathbf{x}_0$, enabling few-step (2–4 NFE) inference.

For 3.0, the CT stage serves a dual purpose: it both accelerates inference and seeds the post-training pipeline with a model that has better instruction-following capability (since distillation implicitly regularizes toward human-preferred outputs when the teacher was trained with quality filtering).

### 7.2 Seedream 4.0: Acceleration Under Multimodal Complexity

The inference acceleration challenge in 4.0 is harder than in 3.0 because:

1. **VLM overhead**: The PE module (Seed1.5-VL fine-tuned) adds substantial prefill cost. For a complex editing request with auto-thinking, the VLM might generate hundreds of thinking tokens before the final prompt.

2. **Extended sequence length**: The DiT now processes $|\mathcal{R}| + |\mathcal{T}|$ tokens instead of $|\mathcal{T}|$ alone. For a reference image at 1024² with patch size 2×2 in an 8× compressed latent space, $|\mathcal{R}| \approx (1024/16)^2 = 4096$ tokens — doubling or tripling the sequence length.

3. **Causal attention memory**: The asymmetric attention mask in 4.0 is denser than standard masking patterns, increasing peak attention memory.

Likely mitigations (implied by the report) include:

- **VLM speculative decoding** for the PE module's thinking phase.
- **Reference token caching**: Since reference tokens are clean (not noised), their key-value cache is fixed across all denoising steps and can be precomputed once.
- **Progressive distillation** of the joint model, preserving the editing capability in the student through careful trajectory sampling from both $\mathcal{D}_{T2I}$ and $\mathcal{D}_{\text{edit}}$.

The reference token caching is particularly impactful. In standard DiT inference, all KV caches are recomputed at each denoising step. With the causal design, reference KV vectors $\mathbf{K}^{\mathcal{R}}, \mathbf{V}^{\mathcal{R}}$ are independent of $t$ and can be computed once:

$$\text{Attn}(\mathbf{z}^{\mathcal{T}}_t) = \text{softmax}\!\left(\frac{\mathbf{Q}^{\mathcal{T}}_t [\mathbf{K}^{\mathcal{R}} \| \mathbf{K}^{\mathcal{T}}_t]^\top}{\sqrt{d_k}}\right) [\mathbf{V}^{\mathcal{R}} \| \mathbf{V}^{\mathcal{T}}_t]$$

where $\mathbf{K}^{\mathcal{R}}, \mathbf{V}^{\mathcal{R}}$ are cached. This reduces attention compute for editing from $O((|\mathcal{R}| + |\mathcal{T}|)^2)$ to $O(|\mathcal{R}||\mathcal{T}| + |\mathcal{T}|^2)$ per step, saving $O(|\mathcal{R}|^2)$ FLOPs per step, amortized over all denoising steps.

---

## 8. Benchmark Methodology

### 8.1 Seedream 3.0: Comprehensive Bilingual Evaluation

Seedream 3.0 establishes a benchmark suite covering:

- **Text-image alignment**: T2I-CompBench (compositional generation), DrawBench, and custom Chinese-language prompts. Metrics: CLIP-T score, BLIP-VQA accuracy, and human preference win rates.
- **Image quality**: FID on COCO-30K (generative diversity), HPS v2 and ImageReward (human preference surrogates).
- **Aesthetic quality**: Manual evaluation on photography, illustration, and design prompts.
- **Resolution generalization**: Evaluation across multiple aspect ratios, checking for geometric distortions and semantic degradation.

The key methodological point in 3.0 is the use of **VLM-based auto-evaluation** as a proxy for human judgment, enabling scalable benchmark coverage. The VLM evaluator uses the same GenRM formulation as the RLHF reward, creating consistency between training signal and evaluation metric — which is both a strength (alignment between training and evaluation) and a potential weakness (Goodhart's Law: optimizing against the evaluator may inflate scores without improving perceptual quality).

### 8.2 Seedream 4.0: Multimodal Benchmark Expansion

4.0 requires new evaluation dimensions:

- **Editing fidelity**: CLIP-Image (DINO-v2 similarity between reference and edited output) for structure preservation; CLIP-T for instruction adherence. These are in tension: high CLIP-I means the output is similar to the reference (little was changed), high CLIP-T means the instruction was followed. The Pareto frontier of (CLIP-I, CLIP-T) characterizes the fidelity-editability tradeoff.

- **Multi-task consistency**: Does the jointly trained model degrade on pure T2I when editing capability is added? The null hypothesis (gradient interference) would predict T2I degradation; 4.0 claims the opposite.

- **PE module contribution**: Ablation of the PE module (raw prompt vs. PE-rewritten prompt) isolates the VLM's contribution to final quality.

- **AdaCoT efficiency**: Measuring quality as a function of thinking budget, verifying the claimed monotone improvement with budget up to the saturation point.

A critical methodological gap in both reports: **calibration of VLM-based evaluators**. If the same Seed1.5-VL family is used for both the PE module, the RLHF reward model, and the evaluation VLM, there is a risk of **distributional self-favoritism** — the evaluation metric may favor outputs conditioned on Seed1.5-VL's representations over those from competing models. Rigorous evaluation would require independent human judgment or VLM evaluators from a different model family (GPT-4V, Gemini 1.5 Pro).

---

## 9. Synthesis: The Theoretical Significance of the 3.0 → 4.0 Transition

The table below summarizes the key advances:

| Dimension             | Seedream 3.0                    | Seedream 4.0                          | Theoretical Significance                       |
| --------------------- | ------------------------------- | ------------------------------------- | ---------------------------------------------- |
| **Architecture**      | Bidirectional DiT, dual-encoder | Causal DiT, unified multimodal        | Enables joint task without architectural split |
| **Training task**     | T2I only                        | T2I + editing jointly                 | Positive multi-task transfer                   |
| **Text conditioning** | CLIP + frozen LLM encoder       | CLIP + LLM + VLM PE (generative)      | Reasoning-capable conditioning                 |
| **RLHF reward**       | VLM GenRM (1B→20B)              | VLM GenRM, multimodal queries         | Extended reward model scaling                  |
| **PE module**         | Lightweight                     | Seed1.5-VL + AdaCoT                   | System 1/2 reasoning for prompt                |
| **Data**              | Re-captioned T2I corpus         | T2I + editing triplets, 3-granularity | Augmentation in semantic conditioning space    |
| **Inference**         | Consistency distillation        | CT + reference KV caching             | Amortized compute via causal structure         |
| **Benchmark**         | Bilingual T2I suite             | Extended multimodal + editing metrics | Pareto evaluation of fidelity-editability      |

The overarching theoretical contribution of 4.0 is the demonstration that **the inductive biases of diffusion and autoregressive modeling are complementary rather than competing**: the causal attention mechanism borrowed from autoregressive transformers, applied to the diffusion forward process, is the enabling mechanism for multimodal unification. This is distinct from hybrid generative models (e.g., DALL-E 3's T5 conditioning, Stable Diffusion 3's MMDiT) in that the causality operates *within* the DiT's spatial attention over latent patches — not merely between a text encoder and an image decoder.

The PE module represents a second unification: **combining System 1 fast generative diffusion with System 2 deliberate VLM reasoning** in a single production pipeline. The practical implication is that prompt engineering — previously a user skill — becomes an internalized model capability, closing the loop between user intent and generation quality.

Finally, the **reward model scaling law confirmation** in 3.0 (emergent reward quality at >20B parameters) is a result with broad implications beyond Seedream: it suggests that RLHF for image generation has not yet reached its reward model capacity ceiling, and that further scaling of reward models — possibly to 100B+ parameter multimodal models — will continue to yield alignment improvements without diminishing returns, pending data and compute investment.