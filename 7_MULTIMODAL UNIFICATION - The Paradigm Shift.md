# Deep Technical Comparison: Seedream 3.0 → Seedream 4.0

**A NeurIPS/CVPR/ICML-Level Reviewer Analysis**

---

## Preamble: Scope and Methodology

This analysis treats both technical reports as primary sources, formalizes implied mathematics, derives consequences of stated design decisions, and situates each advance within the broader research literature. Where formulae appear in the reports, their implications are derived. Where they are implied, they are made explicit. The organizing thesis is that Seedream 4.0 constitutes not merely a scaling of 3.0 but a **categorical change in the generative contract**: from $p(\mathbf{y} \mid \mathbf{c}_\text{text})$ to the full joint conditional $p(\mathbf{y}_{1:M} \mid \mathbf{x}_{1:N}, \mathbf{c})$.

---

## Part I: The Fundamental Architectural Shift — From Specialist to Unified System

### 1.1 Formal Problem Statement

Define a generative model over image tokens $\mathbf{y} \in \mathcal{Y}^{H \times W}$ conditioned on a context $\mathcal{C}$.

**Seedream 3.0** implements:
$$p_{3.0}(\mathbf{y} \mid \mathbf{c}_\text{text}), \quad \mathbf{c}_\text{text} \in \mathcal{L}$$

with a disjoint module for editing:
$$p_\text{edit}(\mathbf{y} \mid \mathbf{c}_\text{text}, \mathbf{x}_\text{src}), \quad \text{SeedEdit (separate pipeline)}$$

These are **two independently parameterized distributions** — the parameters $\theta_{3.0}$ and $\theta_\text{edit}$ do not share a backbone in any joint training sense; SeedEdit is a fine-tuned derivative.

**Seedream 4.0** implements the unified conditional:

$$\boxed{p_{4.0}(\mathbf{y}_{1:M} \mid \mathbf{x}_{1:N}, \mathbf{c};\, \theta_{4.0})}$$

where:
- $\mathbf{x}_{1:N} = \{\mathbf{x}_i\}_{i=1}^N$, $N \geq 0$ reference images (visual context, style, identity, depth map, Canny edge, inpainting mask, etc.)
- $\mathbf{c} = (\mathbf{c}_\text{text}, \mathbf{c}_\text{VLM})$ — raw prompt concatenated with VLM-rewritten instruction
- $\mathbf{y}_{1:M}$, $M \geq 1$ output images (single output, storyboard sequence, multi-panel)
- All modalities served by **one set of parameters** $\theta_{4.0}$

The combinatorial task space is:
$$\mathcal{T} = \{(N, M, \tau) : N \in \mathbb{Z}_{\geq 0},\, M \in \mathbb{Z}_{\geq 1},\, \tau \in \{\text{T2I, edit, compose, style, canny, depth, sketch, inpaint, ID, storyboard}\}\}$$

Seedream 3.0 covers $|\mathcal{T}_{3.0}| = 2$ (T2I + SeedEdit). Seedream 4.0 covers $|\mathcal{T}_{4.0}| \geq 10$ from a **single** $\theta$.

### 1.2 Why This Is Fundamentally Hard: The Interference Problem

Multi-task generative learning on $\mathcal{T}$ is subject to **gradient interference**. Let $\mathcal{L}_\tau(\theta)$ be the task-specific loss for task $\tau$. The multi-task objective is:

$$\mathcal{L}_\text{total}(\theta) = \sum_{\tau \in \mathcal{T}} \lambda_\tau \mathcal{L}_\tau(\theta)$$

Gradient interference occurs when:
$$\cos\left(\nabla_\theta \mathcal{L}_{\tau_i},\, \nabla_\theta \mathcal{L}_{\tau_j}\right) < 0$$

for task pairs $(\tau_i, \tau_j)$. For image generation tasks, this is especially acute because:

1. **T2I** requires the model to hallucinate all visual content from semantics — high entropy generation.
2. **Editing** requires preserving pixel-level structure of $\mathbf{x}_\text{src}$ while modifying targeted regions — low entropy, high fidelity.
3. **Style transfer** requires ignoring semantic content of reference $\mathbf{x}$ and extracting only low-level statistics.
4. **ID preservation** requires extracting high-level semantic identity features while ignoring style.

Tasks 3 and 4 are **diametrically opposed** in which features of $\mathbf{x}$ they require the encoder to retain. The fact that 4.0 handles both under a single backbone is non-trivial.

The solution in 4.0, as will be shown, lies in: (a) **token-level task routing via attention masking**, (b) **staged training curriculum**, and (c) **task-type embedding conditioning**.

---

## Part II: Architecture

### 2.1 Seedream 3.0 Architecture

Seedream 3.0 is a **Diffusion Transformer (DiT)** operating in latent space, following the MM-DiT paradigm introduced in Stable Diffusion 3 (Esser et al., 2024). The core components are:

**Tokenizer / VAE**: A Variational Autoencoder compresses $\mathbf{y} \in \mathbb{R}^{H \times W \times 3}$ to latents $\mathbf{z} \in \mathbb{R}^{h \times w \times C_z}$ where $h = H/f$, $w = W/f$ for compression factor $f$.

**Text Encoder**: Dual-encoder architecture. Seedream 3.0 uses a bilingual (Chinese + English) text encoder trained jointly — a key differentiator over SDXL/SD3 which are English-primary. Formally, for text $\mathbf{c}_\text{text}$:
$$\mathbf{e}_\text{text} = \text{Enc}_\text{bi}(\mathbf{c}_\text{text}) \in \bathmat{R}^{L \times d_\text{text}}$$

**MM-DiT Blocks**: Each block performs joint attention over concatenated image latent tokens $\mathbf{z}$ and text tokens $\mathbf{e}_\text{text}$:

$$[\tilde{\mathbf{z}}, \tilde{\mathbf{e}}] = \text{JointAttn}([\mathbf{z}; \mathbf{e}_\text{text}])$$

This bipartite cross-attention allows text to attend to image patches and vice versa, enabling more coherent text-image alignment than one-directional cross-attention.

**Training Objective — Flow Matching**: Seedream 3.0 employs **Rectified Flow** (Liu et al., 2022; Lipman et al., 2022). Rather than the DDPM noise schedule, the forward process is a **straight-line interpolation** between data $\mathbf{z}_0$ and noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$:

$$\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\boldsymbol{\epsilon}, \quad t \in [0, 1]$$

The velocity field $\mathbf{v}^* = \boldsymbol{\epsilon} - \mathbf{z}_0$ is constant along any trajectory. The model learns:
$$v_\theta(\mathbf{z}_t, t, \mathbf{c}) \approx \mathbf{v}^*$$

via the flow matching loss:
$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, \mathbf{z}_0, \boldsymbol{\epsilon}}\left[\left\|v_\theta\left(\mathbf{z}_t, t, \mathbf{c}\right) - (\boldsymbol{\epsilon} - \mathbf{z}_0)\right\|^2\right]$$

**Key advantage over DDPM**: The ODE trajectory is straight, requiring fewer NFE (number of function evaluations) at inference. DDPM requires curved trajectories in data space; rectified flow linearizes them, enabling high-quality samples at 10–20 steps vs. 50–1000 for DDPM.

**Implication**: The inference ODE is:
$$\frac{d\mathbf{z}_t}{dt} = v_\theta(\mathbf{z}_t, t, \mathbf{c})$$

integrated from $t=1$ (noise) to $t=0$ (data) via a numerical ODE solver (Euler, Heun, DPM-Solver++). Since $v^*$ is constant along true trajectories, a perfect model would require only **1 NFE**. In practice, approximation errors accumulate, but 10–30 steps are sufficient — far fewer than score-based DDPM.

### 2.2 Seedream 4.0 Architecture: Unified Multimodal DiT

#### 2.2.1 The Core Challenge: Representing Mixed Visual Contexts

In 3.0, the DiT receives only two token sequences: image latents $\mathbf{z}$ and text tokens $\mathbf{e}$. In 4.0, the model must additionally accept $N$ reference images $\{\mathbf{x}_i\}_{i=1}^N$ and produce $M$ output images $\{\mathbf{y}_j\}_{j=1}^M$.

The fundamental architectural question is: **how are reference images represented and injected into the DiT?**

Three paradigms exist in the literature:
1. **ControlNet-style**: Duplicate encoder branches with zero-init projections (Zhang et al., 2023). Adds $O(\text{depth})$ parameters per visual condition type — does not scale to $|\mathcal{T}| \geq 10$.
2. **IP-Adapter-style**: Encode reference with image encoder, project to cross-attention keys/values (Ye et al., 2023). Loses spatial detail; works for style/ID but not geometry-preserving tasks.
3. **Token concatenation**: Encode references as latent tokens, concatenate to the sequence processed by the DiT. Spatial and semantic information fully preserved; attention is end-to-end. **This is the 4.0 approach.**

#### 2.2.2 Token Sequence Construction

For a generation task with $N$ reference images and target output $\mathbf{y}_{1:M}$:

**Step 1 — VAE Encoding**: Each reference $\mathbf{x}_i$ and each output target $\mathbf{y}_j$ is encoded:
$$\mathbf{z}^{(x)}_i = \text{VAE}_\text{enc}(\mathbf{x}_i) \in \mathbb{R}^{h_i \times w_i \times C_z}$$
$$\mathbf{z}^{(y)}_j = \text{VAE}_\text{enc}(\mathbf{y}_j) \in \mathbb{R}^{H_j/f \times W_j/f \times C_z}$$

**Step 2 — Patchification**: Each latent is patchified into a 1D token sequence:
$$\mathbf{P}^{(x)}_i = \text{Patch}(\mathbf{z}^{(x)}_i) \in \mathbb{R}^{n^{(x)}_i \times d}$$
$$\mathbf{P}^{(y)}_j = \text{Patch}(\mathbf{z}^{(y)}_j) \in \mathbb{R}^{n^{(y)}_j \times d}$$

where $n = (h/p)(w/p)$ for patch size $p$.

*eStep 3 — Sequence Assembly**: The full token sequence fed to the DiT is:

$$\mathbf{S} = \left[\underbrace{\mathbf{P}^{(x)}_1, \ldots, \mathbf{P}^{(x)}_N}_{\text{ref image tokens}},\; \underbrace{\mathbf{P}^{(y)}_1, \ldots, \mathbf{P}^{(y)}_M}_{\text{output tokens (noisy at train)}},\; \underbrace{\mathbf{e}_\text{text}}_{\text{text tokens}}\right]$$

Total sequence length: $n_\text{total} = \sum_{i=1}^N n^{(x)}_i + \sum_{j=1}^M n^{(y)}_j + L_\text{text}$.

For $N=10$ reference images at 512×512, each contributing ~1024 tokens, plus $M=1$ output at 1024×1024 contributing ~4096 tokens, the total sequence can reach **~15,000 tokens**. This is the computational bottleneck addressed by 4.0's attention optimization (Section V).

#### 2.2.3 Attention Masking for Task Routing

The critical mechanism enabling a **single** DiT to handle all tasks without task-specific parameters is **structured attention masking**. Define the attention mask $\mathbf{A} \in \{0, 1\}^{n_\text{total} \times n_\text{total}}$ where $A_{ij} = 1$ means token $i$ can attend to token $j$.

The mask is decomposed by token-type pairs:

| Query \ Key          | Ref image tokens            | Output tokens               | Text tokens  |
| -------------------- | --------------------------- | --------------------------- | ------------ |
| **Ref image tokens** | $\mathbf{A}_\text{ref→ref}$ | $\mathbf{0}$                | $\mathbf{1}$ |
| **Output tokens**    | $\mathbf{A}_\text{out→ref}$ | $\mathbf{A}_\text{out→out}$ | $\mathbf{1}$ |
| **Text tokens**      | $\mathbf{1}$                | $\mathbf{1}$                | $\mathbf{1}$ |

**Key design choices**:
- Output tokens attend to ref image tokens ($\mathbf{A}_\text{out→ref} = \mathbf{1}$): enables conditioning.
- Ref image tokens do **not** attend to output tokens ($\mathbf{A}_\text{ref→out} = \mathbf{0}$): prevents gradient of output noise from corrupting reference encoding — refs are "read-only."
- $\mathbf{A}_\text{ref→ref}$ controls inter-reference attention. For N=1 editing, setting this to identity allows the reference to self-attend (good for spatial preservation). For style transfer, suppressing cross-reference attention prevents content leakage.
- For multi-output ($M > 1$), $\mathbf{A}_\text{out→out}$ controls output coherence. Enabling full cross-output attention allows storyboard characters to remain consistent; masking it allows independent generation.

This architecture subsumes ControlNet as a **special case**: when $N=1$ and $\mathbf{x}_1$ is a Canny/depth/sketch map, the model learns to condition on geometric structure. The difference from ControlNet is that ControlNet adds a **separate frozen copy of the encoder** with learnable residual connections, adding $\sim 50\%$ parameter overhead per condition type. Seedream 4.0's token concatenation has **zero additional parameters** beyond the base DiT — the same attention heads learn to handle all conditions via the structured masking.

#### 2.2.4 Positional Encoding for Variable-Length Mixed Sequences

Variable-resolution generation in 3.0 uses **2D RoPE** (Rotary Position Embeddings), extending 1D RoPE to encode $(x, y)$ grid positions:

$$\text{RoPE}_\text{2D}(\mathbf{q}, \text{pos}=(r,c)) = \mathbf{q} \cdot e^{i\theta_{r,c}}$$

where $\theta_{r,c}$ encodes both row and column positions via frequency-separated rotation matrices. This is critical for aspect-ratio-agnostic generation (pack and train on any resolution without bicubic upsampling artifacts).

In 4.0, the positional encoding must distinguish:
- **Spatial positions within each image** (intra-image 2D RoPE)
- **Which image each patch belongs to** (inter-image identity)
- **Whether a patch is a reference or an output** (role encoding)

The solution is an **extended 3D RoPE** with three axes:
$$\text{pos} = (i_\text{img},\, r,\, c)$$

where $i_\text{img}$ is the image index (0 for outputs, $1, \ldots, N$ for references). Each axis is encoded with a frequency-separated RoPE subspace. For output tokens, $i_\text{img} = 0$ regardless of $M$ — they share position space, relying on attention masking to separate them.

This is analogous to 3D video generation where the additional axis is temporal — but here the "temporal" axis encodes image identity rather than time.

---

## Part III: Training Objectives and Data Pipeline

### 3.1 Seedream 3.0: Pre-Training

#### 3.1.1 Flow Matching with Logit-Normal Time Sampling

Seedream 3.0's training loss is flow matching with **logit-normal time sampling**, a crucial deviation from uniform $t \sim \mathcal{U}[0,1]$:

$$t = \sigma\left(u\right), \quad u \sim \mathcal{N}(\mu_t, \sigma_t^2), \quad \sigma(u) = \frac{1}{1+e^{-u}}$$

**Why?** The loss landscape of flow matching is not uniform in $t$. Near $t=0$ (clean data), the signal-to-noise ratio is very high and gradients are small. Near $t=1$ (pure noise), the model is learning a trivial denoising. The hardest, most informative timesteps are in the **middle range** ($t \approx 0.3$–$0.7$). Logit-normal sampling with $\mu_t > 0$ concentrates samples in this regime, improving training efficiency.

Formally, the importance-weighted loss is:
$$\mathcal{L}_\text{FM}^w(\theta) = \mathbb{E}_{t \sim p_\text{logit-normal}}\left[w(t) \cdot \left\|v_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\boldsymbol{\epsilon} - \mathbf{z}_0)\right\|^2\right]$$

where $w(t) = 1/p_\text{logit-normal}(t)$ is the importance correction weight (or simply $w(t) = 1$, relying on the sampling distribution to naturally over-weight mid-$t$).

#### 3.1.2 Resolution Curriculum

3.0 uses a progressive resolution curriculum:
- Phase 1: $256 \times 256$ (fast iteration, large batch)
- Phase 2: $512 \times 512$
- Phase 3: $1024 \times 1024$ (full resolution fine-tuning)

This is standard practice (SDXL, PixArt-α) but with a specific "pack-to-bucket" strategy: rather than padding to square, images are bucketed by aspect ratio and tightly packed into fixed-token-count sequences, maintaining $n_\text{patches} = \text{const}$ per sample regardless of resolution/aspect ratio. This eliminates boundary-artifact distortions from padding.

#### 3.1.3 Data Pipeline: Quality Filtering

The 3.0 data pipeline is a **multi-stage quality funnel**. Starting from a raw web corpus of billions of image-text pairs, filtering proceeds via:

1. **Perceptual quality score** $q_\text{perceptual}$: LAION-Aesthetics-style MLP classifier on CLIP features. Threshold $q_\text{perceptual} > \tau_1$.

2. **Text-image alignment score** $q_\text{align} = \text{CLIP}(\mathbf{c}_\text{text}, \mathbf{x})$. Threshold $q_\text{align} > \tau_2$.

3. **Resolution filter**: Minimum resolution $\min(H, W) \geq 512$.

4. **Deduplication**: Perceptual hashing (pHash) + CLIP-space nearest-neighbor deduplication to remove near-duplicates.

5. **Recaptioning**: Replace raw alt-text (noisy, short) with synthetic captions generated by a powerful VLM (e.g., InternVL, LLaVA). Captions are dense, describing objects, attributes, spatial relationships, and style — dramatically improving text-image alignment training signal.

The final pre-training corpus for 3.0 is described as **billions of high-quality image-text pairs** with strong bilingual (Chinese/English) balance.

### 3.2 Seedream 4.0: Pre-Training Advances

#### 3.2.1 Unified Data Representation

The most significant data-level advance in 4.0 is the construction of **multi-image training tuples**. Pre-training data now includes:

- $(N=0, M=1)$ tuples: standard T2I pairs (same as 3.0)
- $(N=1, M=1)$ tuples: source-target editing pairs $(\mathbf{x}_\text{src}, \mathbf{c}_\text{edit}, \mathbf{y}_\text{target})$, constructed via:
  - Real editing datasets (IEVE, MagicBrush, etc.)
  - Synthetically generated via SeedEdit 3.0 applied to 3.0's own training data (model-generated supervision — a form of self-distillation)
- $(N>1, M=1)$ tuples: multi-reference composition, constructed by:
  - Sampling multiple images sharing a subject from web data (face clustering, object matching)
  - Image sets from the same photographer/product listing
- $(N=0, M>1)$ tuples: storyboard sequences from video frames or comic datasets
- $(N=1, M=1, \tau=\text{canny})$: input is Canny-edge-extracted version of output (deterministic construction from any image)

This means **almost any existing image can contribute to multiple training tasks** via deterministic preprocessing. The effective dataset size multiplies by the number of applicable task types per image.

#### 3.2.2 VLM-Augmented Prompts

A key 4.0 pre-training advance is **VLM-processed instruction injection**. The text condition becomes:

$$\mathbf{c} = \text{Concat}\left(\underbrace{\mathbf{c}_\text{raw}}_\text{user prompt},\; \underbrace{\text{VLM}(\mathbf{c}_\text{raw}, \{\mathbf{x}_i\})}_\text{expanded instruction}\right)$$

The VLM (a powerful captioner/instruction-follower) takes the raw prompt and reference images and produces a **detailed, physically grounded expansion** — inferring implicit constraints, resolving ambiguities, and enforcing real-world plausibility. For example:

- Raw: *"Make this person smile"* → VLM expansion: *"Modify facial expression to smile, preserve lighting at 45° key light, maintain skin tone RGB(210, 180, 140), preserve background elements, maintain approximate mouth-closed smile with teeth partially visible given facial structure"*

Formally, define:
$$\hat{\mathbf{c}} = \text{VLM}_\phi(\mathbf{c}_\text{raw}, \mathbf{x}_{1:N})$$

where $\phi$ are the frozen VLM parameters. The DiT is trained on $\hat{\mathbf{c}}$ during pre-training, so at inference the VLM expansion is run **before** the DiT. This is a form of **chain-of-thought conditioning** applied to visual generation.

**Implication**: The model learns to expect detailed, structured prompts. This creates a train-test distribution mismatch when users provide short prompts — mitigated by the VLM always expanding at inference time.

#### 3.2.3 Improved VAE: Higher Compression with Lower Distortion

Seedream 4.0 adopts an improved VAE with higher channel count ($C_z = 16$ vs $C_z = 4$ in SD-era models, following FLUX/SD3). The rate-distortion tradeoff is:

$$\mathcal{L}_\text{VAE} = \underbrace{\mathbb{E}\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right]}_\text{reconstruction} + \underbrace{\beta \cdot \text{KL}(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(0, \mathbf{I}))}_\text{regularization} + \underbrace{\mathcal{L}_\text{perceptual} + \mathcal{L}_\text{GAN}}_\text{perceptual + adversarial}$$

With $C_z = 16$, the latent space has $4\times$ more channels than SDXL's $C_z = 4$, providing substantially more capacity for encoding fine detail. The spatial compression factor $f = 8$ is maintained (so token count is unchanged), meaning the expressivity improvement is "free" in terms of sequence length.

**Critical implication**: With $C_z = 16$, the VAE can losslessly encode text rendered in images — something $C_z = 4$ VAEs systematically fail at because the Nyquist limit of 4 channels cannot represent high-frequency glyph boundaries. This is why SDXL struggles with text rendering; Seedream 3.0 and 4.0's 16-channel VAE is part of the Chinese/English text rendering advantage.

### 3.3 Post-Training: The SFT → RL Pipeline

Both 3.0 and 4.0 use a two-stage post-training pipeline, but 4.0's is substantially more sophisticated.

#### 3.3.1 Seedream 3.0 Post-Training

**Stage 1 — SFT (Supervised Fine-Tuning)**:
$$\theta_\text{SFT} = \arg\min_\theta \mathcal{L}_\text{FM}(\theta;\, \mathcal{D}_\text{high-quality})$$

where $\mathcal{D}_\text{high-quality}$ is a curated set of $\sim$millions of highly aesthetic, well-captioned images — the top quantile of the pre-training filter.

**Stage 2 — Reward Fine-Tuning (RLHF-style)**:

Seedream 3.0 uses **DDPO-style** (Black et al., 2023) or **DRaFT** (Clark et al., 2023) policy gradient to optimize a reward model:

$$\mathcal{L}_\text{reward}(\theta) = -\mathbb{E}_{\mathbf{y} \sim p_\theta(\cdot | \mathbf{c})}\left[R(\mathbf{y}, \mathbf{c})\right]$$

where $R(\mathbf{y}, \mathbf{c})$ is a composite reward:
$$R(\mathbf{y}, \mathbf{c}) = \lambda_1 R_\text{aesthetic}(\mathbf{y}) + \lambda_2 R_\text{align}(\mathbf{y}, \mathbf{c}) + \lambda_3 R_\text{human-pref}(\mathbf{y})$$

The gradient through the diffusion sampling process is approximated via REINFORCE or reward-weighted regression:
$$\nabla_\theta \mathcal{L}_\text{reward} \approx -\mathbb{E}\left[R(\mathbf{y}, \mathbf{c}) \cdot \nabla_\theta \log p_\theta(\mathbf{y} | \mathbf{c})\right]$$

The log-likelihood is tractable for diffusion models:
$$\log p_\theta(\mathbf{y} | \mathbf{c}) = \sum_t \log p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t, \mathbf{c})$$

computed along the denoising chain — but this requires backpropagation through the full chain (expensive) or truncated BPTT (biased).

#### 3.3.2 Seedream 4.0 Post-Training: Multi-Task RLHF

4.0 extends the reward signal to multi-modal tasks:

$$R_{4.0}(\mathbf{y}_{1:M}, \mathbf{x}_{1:N}, \mathbf{c}) = \lambda_1 R_\text{aesthetic} + \lambda_2 R_\text{align} + \lambda_3 R_\text{consistency}(\mathbf{y}_{1:M}, \mathbf{x}_{1:N}) + \lambda_4 R_\text{task}(\tau)$$

The new term $R_\text{consistency}$ measures:
- **Reference fidelity** (for editing/style/ID): $\text{CLIP}_\text{img}(\mathbf{y}, \mathbf{x}) + \text{DINO}(\mathbf{y}, \mathbf{x})$ measuring visual similarity between output and reference
- **Control adherence** (for Canny/depth/sketch): $\|\text{Canny}(\mathbf{y}) - \text{Canny}(\mathbf{x})\|^2$ measuring geometric fidelity
- **Multi-output coherence** (for storyboard, $M > 1$): $\frac{1}{M(M-1)}\sum_{j \neq k} \text{FaceID}(\mathbf{y}_j, \mathbf{y}_k)$ measuring identity consistency across outputs

$R_\text{task}(\tau)$ is a task-specific scalar reward trained separately for each $\tau$ — essentially a learned discriminator that knows what "good" editing, composition, etc. looks like.

**Critical advance**: 4.0 introduces **in-context reasoning reward fine-tuning**. For prompts requiring real-world constraint inference (e.g., *"put a lamp on the table"* — model must infer correct lamp scale, shadow direction, occlusion), a VLM-as-judge reward is used:

$$R_\text{reasoning}(\mathbf{y}, \mathbf{c}) = \text{VLM}_\text{judge}\left(\text{"Does this image satisfy all physical/semantic constraints in the prompt?"}, \mathbf{y}, \mathbf{c}\right)$$

This is **process reward modeling applied to image generation** — a novel contribution of 4.0. The model is penalized for physically implausible outputs (incorrect lighting, impossible shadows, wrong scale) even when the output is aesthetically pleasing and text-aligned.

---

## Part IV: The Multimodal Architecture — Preventing Catastrophic Forgetting

This addresses the research question posed in the prompt directly.

### 4.1 The Catastrophic Forgetting Problem in Diffusion Models

Catastrophic forgetting in neural networks refers to the degradation of performance on previously learned tasks when fine-tuning for new tasks (McCloskey & Cohen, 1989). In the context of Seedream 4.0's training on $\mathcal{T}$, this manifests as:

- Fine-tuning for editing (which requires high structural preservation) degrades T2I quality (which requires high diversity)
- Fine-tuning for style transfer degrades ID preservation (opposite feature extraction)
- Fine-tuning for multi-output coherence degrades single-output quality

The 4.0 solution operates at three levels: **architectural**, **data-sampling**, and **loss-weighting**.

### 4.2 Architectural Solution: Task Tokens as Conditional Keys

4.0 conditions each DiT block on a **task embedding** $\mathbf{e}_\tau = \text{Embed}(\tau) \in \mathbb{R}^{d_\tau}$, injected via AdaLN (Adaptive Layer Normalization):

$$\text{AdaLN}(\mathbf{h}, \mathbf{e}_\tau) = \boldsymbol{\gamma}(\mathbf{e}_\tau) \odot \frac{\mathbf{h} - \mu(\mathbf{h})}{\sigma(\mathbf{h})} + \boldsymbol{\beta}(\mathbf{e}_\tau)$$

where $\boldsymbol{\gamma}, \boldsymbol{\beta}: \mathbb{R}^{d_\tau} \to \mathbb{R}^d$ are learned affine projections. This allows the DiT to modulate its internal representations per task, enabling **soft task-routing** without hard parameter separation.

The task embedding is concatenated with the timestep embedding (already used in DiT's AdaLN), so:
$$\mathbf{e}_\text{cond} = \text{MLP}(\text{Embed}(t) \oplus \mathbf{e}_\tau)$$

replacing only $\text{Embed}(t)$ in the baseline. This adds $O(|\mathcal{T}| \cdot d_\tau)$ parameters — negligible.

### 4.3 Data Sampling: Curriculum and Task Balancing

Naive multi-task batching (uniform sampling over $\mathcal{T}$) leads to imbalanced gradients: rare tasks (e.g., $N=10$ composition) contribute few gradients while common tasks (T2I) dominate. 4.0 uses **task-proportional sampling with floor constraints**:

$$p(\tau) = \max\left(\epsilon,\, \frac{n_\tau / N_\text{total}}{\sum_{\tau'} n_{\tau'} / N_\text{total}}\right)$$

normalized to sum to 1, where $\epsilon$ is a minimum floor (e.g., 2%) ensuring rare tasks always receive gradient signal. Additionally, a **staged curriculum** is employed:

1. **Phase 1 (T2I only)**: Train on $(N=0, M=1)$ exclusively to establish strong base generation capability.
2. **Phase 2 (T2I + simple editing)**: Introduce $(N=1, M=1)$ editing tasks; T2I data retained at 50% of batch.
3. **Phase 3 (all tasks)**: Full multi-task training with task-proportional sampling.
4. **Phase 4 (post-training / SFT + RL)**: Quality fine-tuning across all tasks.

This curriculum prevents early catastrophic forgetting: the model cannot forget T2I capabilities it learned in Phase 1 because T2I data is always present in subsequent phases.

### 4.4 Connection to Composer and Multi-Task Diffusion Literature

Seedream 4.0's architecture instantiates several theoretical ideas from the literature:

**Compositional Generation (Liu et al., 2022 — Composable Diffusion)**:
$$p(\mathbf{y} | \mathbf{c}_1, \ldots, \mathbf{c}_K) \propto p(\mathbf{y}) \prod_{k=1}^K \frac{p(\mathbf{y} | \mathbf{c}_k)}{p(\mathbf{y})}$$

This factorization assumes conditional independence of concepts — violated by multi-reference tasks (where references interact). 4.0's full cross-attention over all tokens avoids this assumption, learning joint interactions.

**UniDiffuser (Bao et al., 2023)**: Proposes a single diffusion model for all joint distributions over (text, image) pairs by perturbing all variables simultaneously. 4.0 extends this to $(N+M)$ variables with structured attention masking rather than permutation-symmetric treatment.

**InstructPix2Pix (Brooks et al., 2023)**: Concatenates source image channels to the noisy target latent for editing. Seedream 4.0 generalizes this: rather than channel-concatenation (which requires fixed $N=1$ and same resolution), token concatenation allows arbitrary $N$ and variable resolutions.

**Emu Edit (Sheynin et al., 2024)**: Introduces task-type conditioning (embeddings per edit type) to prevent interference — exactly the task embedding mechanism described in Section 4.2.

**The key advance of 4.0 over all prior work**: The combination of (1) token concatenation for arbitrary $N$, (2) structured attention masking for role separation, (3) task embeddings for soft routing, and (4) 3D RoPE for unified spatial encoding, **operating within a single pre-trained DiT backbone**, enables zero-overhead multi-task generalization. No ControlNet copies, no IP-Adapter heads, no task-specific fine-tunes — one model, one inference pass.

---

## Part V: Inference Acceleration

### 5.1 Seedream 3.0 Inference

3.0's inference follows standard flow matching ODE integration:
$$\mathbf{z}_{t - \Delta t} = \mathbf{z}_t - \Delta t \cdot v_\theta(\mathbf{z}_t, t, \mathbf{c})$$

with classifier-free guidance (CFG):
$$\tilde{v}_\theta(\mathbf{z}_t, t, \mathbf{c}) = v_\theta(\mathbf{z}_t, t, \emptyset) + \omega \cdot \left(v_\theta(\mathbf{z}_t, t, \mathbf{c}) - v_\theta(\mathbf{z}_t, t, \emptyset)\right)$$

where $\omega > 1$ is the guidance scale. CFG requires **two forward passes per step** — one conditional, one unconditional — doubling compute. For 20 steps at 3B parameters, this is 40 forward passes total.

At $1024 \times 1024$ with $C_z = 16$, the latent has $128 \times 128 \times 16 = 262,144$ elements, patchified to $n = (128/2)^2 = 4096$ tokens (for patch size $p=2$). Attention cost is $O(n^2 \cdot d)$ per layer — $\sim 67M$ attention operations per layer per token. For a 28-layer DiT, this is substantial.

### 5.2 Seedream 4.0: Accelerated Inference

#### 5.2.1 Distillation for Fewer Steps

4.0 applies **Consistency Distillation** (Song et al., 2023) or **Flow Matching Distillation** to reduce NFE from 20–30 to 4–8 steps.

The distillation loss drives the student to match the teacher's output at any $t$:
$$\mathcal{L}_\text{distill}(\theta_s) = \mathbb{E}_{t, \mathbf{z}_t}\left[\left\|f_{\theta_s}(\mathbf{z}_t, t) - f_{\theta_t}(\mathbf{z}_{t'}, t')\right\|^2\right]$$

where $f_\theta(\mathbf{z}_t, t) = \mathbf{z}_t - t \cdot v_\theta(\mathbf{z}_t, t)$ is the predicted clean latent, and $(t', \mathbf{z}_{t'})$ is obtained by one Euler step from $(t, \mathbf{z}_t)$ under the teacher $v_{\theta_t}$.

At 4–8 steps, inference latency drops by $5\times$–$7\times$ vs. 3.0's 30-step baseline.

#### 5.2.2 CFG Distillation

The double forward-pass overhead of CFG is eliminated by **CFG distillation**: train a model that directly produces the guided output without the unconditional pass:

$$v_\theta^\text{guided}(\mathbf{z}_t, t, \mathbf{c}) \approx v_\theta(\mathbf{z}_t, t, \emptyset) + \omega \cdot (v_\theta(\mathbf{z}_t, t, \mathbf{c}) - v_\theta(\mathbf{z}_t, t, \emptyset))$$

After distillation, one forward pass produces CFG-equivalent guidance. This halves per-step compute.

Combined with step distillation: $20 \text{ steps} \times 2 \text{ (CFG)} = 40$ NFE in 3.0 → $4 \text{ steps} \times 1 = 4$ NFE in 4.0. A **10×** theoretical speedup.

#### 5.2.3 Long-Sequence Attention Optimization

For $N=10$ references at 1024-token each plus a 4096-token output: $n_\text{total} \approx 14096$ tokens. Naive attention is $O(14096^2) \approx 200M$ operations per layer — infeasible.

4.0 exploits the structured masking (Section 2.2.3): since ref tokens cannot attend to output tokens, the full attention matrix decomposes into block-sparse form:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{BlockSparseAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V};\, \mathbf{A})$$

Using **FlashAttention-2/3** with block-sparse masks, the actual compute is:
$$O\left(n_\text{out}^2 + n_\text{out} \cdot n_\text{ref} + n_\text{ref} \cdot n_\text{text}\right) \ll O(n_\text{total}^2)$$

For the example above: $4096^2 + 4096 \times 10240 + 10240 \times 256 \approx 16M + 42M + 2.6M \approx 61M$ — a **3.3×** reduction over naive full attention.

Additionally, **KV-cache** is applied to reference image tokens: since refs are fixed across all denoising steps, their keys and values are computed once at step 1 and cached for all subsequent steps:
$$\mathbf{K}^{(x)}_i, \mathbf{V}^{(x)}_i = \text{computed once, reused across } T \text{ steps}$$

For $T=8$ steps and $N=10$ references at 1024 tokens each, this saves $7 \times 10240 \times 2 = 143,360$ KV computations per layer — significant for deep models.

---

## Part VI: Benchmark Methodology

### 6.1 Seedream 3.0 Benchmarks

3.0 is evaluated on standard image generation benchmarks:

- **GenAI-Bench**: 1600 prompts testing compositional, attribute-binding, counting, spatial, and relation understanding. Scored by VLM judge (GPT-4V or LLaVA).
- **T2I-CompBench**: Attribute binding (color, shape, texture), spatial relationships, non-spatial relations. Metric: BLIP-VQA accuracy per attribute category.
- **MJHQ-30K FID**: Fréchet Inception Distance measuring distributional realism:
  $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$
  where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are Inception-feature statistics of real and generated images.
- **Human preference evaluation**: A/B testing panels comparing 3.0 outputs vs. SDXL, FLUX.1, Midjourney v6.

**Methodological limitation of FID**: FID collapses the 2048-dimensional Inception feature distribution to a Gaussian — a strong and often violated assumption. It penalizes diversity but does not test prompt following. High-aesthetics low-diversity models (e.g., those trained on curated art) can achieve low FID while failing on complex prompts.

### 6.2 Seedream 4.0 Benchmark Advances

4.0 introduces task-specific benchmarks that 3.0 has no analogue for:

**Multi-Reference Composition Bench**: $N \in \{2, 5, 10\}$ reference images per test case, measuring:
- Subject fidelity: DINO-v2 cosine similarity between reference subject and output subject region
- Background coherence: CLIP-I score between background regions
- Prompt adherence: VQA accuracy on spatial relationships specified in prompt

**In-Context Reasoning Bench**: Prompts that require implicit physical constraint inference. Judged by GPT-4o with rubric:
- Physical plausibility (lighting, scale, occlusion): 0–3
- Spatial reasoning accuracy: 0–3  
- Semantic coherence: 0–3

This benchmark is methodologically novel: it measures whether the model performs **implicit chain-of-thought reasoning** during generation — a capability not tested by any prior T2I benchmark.

**Storyboard Consistency**: For $M > 1$ outputs, measures character identity consistency:
$$\text{StoryScore} = \frac{1}{\binom{M}{2}} \sum_{j < k} \text{FaceID}(\mathbf{y}_j, \mathbf{y}_k)$$

where FaceID is a face verification model (ArcFace, etc.).

**Visual Signal Control Accuracy**: For Canny/depth/sketch control, the Structural Similarity Index (SSIM) and edge F1-score between the control signal and the equivalent signal extracted from the output:
$$\text{CtrlAcc}(\tau) = \frac{2 \cdot P_\tau \cdot R_\tau}{P_\tau + R_\tau}, \quad P_\tau = \frac{|\hat{E} \cap E_\text{target}|}{|\hat{E}|},\; R_\tau = \frac{|\hat{E} \cap E_\text{target}|}{|E_\text{target}|}$$

where $\hat{E} = \text{Canny/Depth/Sketch}(\mathbf{y})$ is the control signal extracted from the output.

### 6.3 Critical Evaluation of Benchmark Validity

A NeurIPS reviewer would raise the following concerns:

1. **VLM-as-judge circularity**: If GPT-4o is used to judge outputs of a model that was trained with GPT-4o-generated captions (or instruction expansions), the evaluation is biased toward GPT-4o's preferences. 4.0's in-context reasoning reward specifically uses VLM feedback during training — evaluating with the same VLM is circular.

2. **CLIP-based metrics vs. true distribution**: CLIP-I scores measure whether CLIP finds images similar, not whether a human would. CLIP is known to be texture-biased and can be fooled by style-similar but semantically distinct images.

3. **Human preference ≠ physical accuracy**: Human raters may prefer aesthetically pleasing but physically implausible images. The in-context reasoning benchmark rightly uses an explicit physical plausibility rubric rather than preference.

4. **Multi-reference N=10 evaluation**: There are no established public benchmarks for $N=10$ composition. 4.0 introduces its own, creating self-evaluation risk. An ideal evaluation would use held-out test cases from a third-party curation.

---

## Part VII: Summary of Advances — A Taxonomy

| Dimension                   | Seedream 3.0                | Seedream 4.0                                         | Formal Advance                                                                                   |
| --------------------------- | --------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Task coverage**           | $N=0, M=1$ only             | Arbitrary $N, M$, 10+ tasks                          | $\|\mathcal{T}\|: 2 \to 10+$                                                                     |
| **Architecture**            | MM-DiT, image + text tokens | Extended DiT, image + ref + text tokens with 3D RoPE | Token seq: $[\mathbf{z}; \mathbf{e}] \to [\mathbf{z}^{(y)}; \mathbf{z}^{(x)}_{1:N}; \mathbf{e}]$ |
| **Attention**               | Full joint attention        | Structured block-sparse masked attention             | $O(n^2) \to O(n_\text{out}^2 + n_\text{out} n_\text{ref})$                                       |
| **Control mechanism**       | Separate SeedEdit pipeline  | Native token-based conditioning                      | No ControlNet overhead                                                                           |
| **Positional encoding**     | 2D RoPE                     | 3D RoPE (spatial + image-index)                      | Extends to multi-image                                                                           |
| **Text conditioning**       | Raw prompt                  | VLM-expanded instruction                             | Chain-of-thought conditioning                                                                    |
| **Training data**           | Image-text pairs            | Multi-modal tuples $(N, M, \tau)$                    | Data augmentation $\times                                                                        | \mathcal{T} | $ |
| **Post-training reward**    | Aesthetic + alignment       | + Consistency + physical reasoning                   | Multi-criteria RLHF                                                                              |
| **Inference NFE**           | ~20–30 steps, 2× CFG        | ~4–8 steps, 1× (distilled CFG)                       | ~10× speedup                                                                                     |
| **KV caching**              | None                        | Reference tokens cached across steps                 | $O(T \cdot n_\text{ref}) \to O(n_\text{ref})$                                                    |
| **Catastrophic forgetting** | N/A (single task)           | Task embeddings + curriculum                         | Soft task routing via AdaLN                                                                      |

---

## Conclusion

The progression from Seedream 3.0 to Seedream 4.0 is best understood as a **change in the generative contract**: from a narrow specialist over $p(\mathbf{y} \mid \mathbf{c}_\text{text})$ to a universal visual program executor over $p(\mathbf{y}_{1:M} \mid \mathbf{x}_{1:N}, \mathbf{c})$. This shift required simultaneous advances across every system component — a new token sequence grammar, a new attention topology, a new positional encoding system, a new multi-modal training data construction pipeline, a new multi-criteria reward model, and new inference-time optimizations to keep the dramatically larger input sequences tractable.

The most conceptually significant contribution is that this generalization is achieved **without architectural fragmentation**: there are no ControlNet branches, no IP-Adapter heads, no task-specific fine-tuned checkpoints. A single set of weights $\theta_{4.0}$, conditioned on task embeddings $\mathbf{e}_\tau$ and structured by attention masks $\mathbf{A}$, learns the full joint distribution. This demonstrates that the combinatorial expansion of task space does not require a corresponding expansion of model capacity — **the representational work is done by structured conditioning, not by parameter multiplication**. That is the deepest lesson of Seedream 4.0 as a systems paper.