# Seedream 3.0 → 4.0: Deep Technical Comparison
### At the Level of a NeurIPS / CVPR / ICML Paper Reviewer

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture](#2-architecture)
3. [Training Objectives](#3-training-objectives)
4. [Data Engineering Philosophy](#4-data-engineering-philosophy)
5. [Post-Training](#5-post-training)
6. [Inference Acceleration](#6-inference-acceleration)
7. [Benchmark Methodology](#7-benchmark-methodology)
8. [Synthesis: Knowledge-Centric Generation, Compositional Generalization, and Data-Centric AI](#8-synthesis-knowledge-centric-generation-compositional-generalization-and-data-centric-ai)
9. [Remaining Limitations and Open Questions](#9-remaining-limitations-and-open-questions)

---

## 1. Executive Summary

The progression from Seedream 3.0 to 4.0 is not a marginal incremental update: it represents a *system-level paradigm shift* across every pipeline stage. The table below summarises the principal axes of advancement before each is analysed in rigorous detail.

| Dimension                       | Seedream 3.0                                                 | Seedream 4.0                                                                     |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Backbone architecture**       | MMDiT (parameter-scaled from 2.0)                            | New efficient DiT; >10× FLOPs reduction                                          |
| **VAE compression**             | Standard latent compression                                  | High-compression-ratio VAE; far fewer image tokens                               |
| **Native resolution**           | 512²–2048² (up to 2K)                                        | 1024²–4096² (up to 4K)                                                           |
| **Training stages**             | 256² pre-train → 512²–2048² fine-tune                        | 512² pre-train → 1024²–4096² fine-tune                                           |
| **Core training loss**          | Flow matching + REPA (λ=0.5)                                 | Flow matching + REPA (inherited) + adversarial post-training (ADP→ADM)           |
| **Data paradigm**               | Defect-aware masking + dual-axis top-down resampling         | Dual-axis + dedicated knowledge sub-pipeline + module-level upgrades             |
| **Deduplication**               | Semantic cross-modal retrieval embedding                     | Combined semantic + low-level visual embeddings                                  |
| **Post-training scope**         | T2I only (CT→SFT→RLHF→PE)                                    | Unified T2I + image editing, joint multimodal post-training                      |
| **PE model**                    | Implicit rewriting                                           | End-to-end VLM (Seed1.5-VL) with AdaCoT task routing                             |
| **Reward model**                | VLM, 1B–>20B scaling                                         | Inherited + improved base model                                                  |
| **Acceleration algorithm**      | Consistent noise expectation + SSD-based importance sampling | ADP + ADM adversarial distillation + speculative decoding                        |
| **Quantization**                | Inherited from Seedream 2.0                                  | Adaptive 4/8-bit hybrid with search-based granularity + hardware kernels         |
| **Inference speed (2K, no PE)** | ~3.0 s                                                       | ~1.4 s                                                                           |
| **Multimodal tasks**            | T2I only (SeedEdit separate)                                 | Natively unified: T2I + single-image edit + multi-image reference + multi-output |
| **Knowledge generation**        | Incidental                                                   | First-class: formulas, charts, UI schematics, chemical equations                 |
| **Benchmark suite**             | Bench-377, Artificial Analysis Arena                         | MagicBench 4.0 + DreamEval (128 sub-tasks, 1,600 prompts, tiered difficulty)     |

---

## 2. Architecture

### 2.1 Backbone: MMDiT (3.0) → Efficient DiT (4.0)

**Seedream 3.0** inherits the **Multi-Modal Diffusion Transformer (MMDiT)** from Seedream 2.0 and scales its parameter count. The defining property of MMDiT is dual-stream attention: image tokens $\mathbf{Z}^{\text{img}} \in \mathbb{R}^{n \times d}$ and text tokens $\mathbf{Z}^{\text{txt}} \in \mathbb{R}^{L \times d}$ are processed by separate but *interacting* transformer streams, with joint self-attention across both modality sequences at every layer. Each MMDiT block implements:

$$\text{Attn}([\mathbf{Z}^{\text{img}}; \mathbf{Z}^{\text{txt}}], [\mathbf{Z}^{\text{img}}; \mathbf{Z}^{\text{txt}}])$$

enabling dense cross-modal interaction. The computational cost of the joint attention is $\mathcal{O}((n + L)^2 \cdot d)$ per layer.

**Seedream 4.0** replaces this with a *fundamentally redesigned DiT backbone* achieving >10× training and inference FLOPs reduction while improving performance — a strict Pareto improvement. While architectural specifics are not fully disclosed, the following properties are reported or implied:

- **Hardware-friendly design**: supports Hybrid Sharded Data Parallelism (HSDP) without tensor or expert parallelism, implying efficient activation memory layout and reduced inter-GPU communication volume.
- **Strong scalability**: performance continues to improve as model size and data are scaled (demonstrated by Seedream 4.5).
- **Causal diffusion structure in the DiT framework**: enables multi-image conditioning by treating reference image tokens as preceding context tokens in a causal attention mask, allowing the DiT to autoregressively attend to an arbitrary number of input images without architectural modification.

> **Formal implication of the 10× FLOPs claim.** Let $\mathcal{F}_3$ and $\mathcal{F}_4$ denote the FLOPs per denoising step for Seedream 3.0 and 4.0, respectively. The claim is $\mathcal{F}_3 / \mathcal{F}_4 \geq 10$. This is not achieved by quantization or distillation alone (those are handled separately); it reflects a change in the model's *computational primitives*. If attention complexity contributes dominantly, and the new DiT uses a linear or block-sparse attention approximation reducing per-layer cost from $\mathcal{O}(n^2 d)$ to $\mathcal{O}(n d)$, the FLOPs reduction factor scales as $n / k$ for some block size $k$. More conservatively, even a 3× reduction in sequence length from VAE compression (see §2.2) combined with a 3–4× reduction from architectural efficiency gives the reported factor. The most likely explanation is a *combination* of both.

### 2.2 VAE: Standard (3.0) → High-Compression-Ratio (4.0)

This is arguably the single most *mechanistically consequential* change in the entire pipeline.

**Seedream 3.0** uses a standard VAE at compression ratio $r$ such that a $W \times H$ image maps to a latent of spatial dimensions $\lfloor W/r \rfloor \times \lfloor H/r \rfloor$. For $r = 8$ (typical for SD-class models), a $1024^2$ image yields $128^2 = 16{,}384$ tokens; a $2048^2$ image yields $256^2 = 65{,}536$ tokens.

**Seedream 4.0** introduces a **high-compression-ratio VAE** that substantially reduces the number of image tokens. Let $r_4 > r_3$ denote the new compression ratio. The cascade of implications:

**Attention cost.** Self-attention FLOPs scale as $\mathcal{O}(n^2)$ where $n = (W \cdot H)/r^2$. The ratio of attention costs is:

$$\frac{\mathcal{F}_{\text{attn}}^{(3)}}{\mathcal{F}_{\text{attn}}^{(4)}} = \left(\frac{n_3}{n_4}\right)^2 = \left(\frac{r_4}{r_3}\right)^4$$

Even a modest $r_4 / r_3 = 1.7$ yields $\approx 8.4\times$ attention FLOPs reduction. A factor of 2 in compression radius gives $16\times$. Combined with architectural efficiency gains, this straightforwardly explains the stated >10× total FLOPs reduction.

**4K feasibility.** At $r_3 = 8$ and resolution $4096^2$, a standard VAE yields $512^2 = 262{,}144$ tokens — the resulting attention matrix has $\sim 6.87 \times 10^{10}$ entries, making full self-attention completely intractable. The high-compression VAE reduces this by $r_4^4/r_3^4$, rendering 4K training tractable without requiring approximate attention mechanisms.

**Memory bandwidth.** Reducing $n$ reduces the memory footprint of key-value caches by $\mathcal{O}(n \cdot d \cdot L_{\text{layers}})$, directly enabling larger effective batch sizes on fixed hardware — important for stable RLHF training at high resolution.

> **Reviewer critique.** The paper does not report the reconstruction FID or rFID of the new VAE, nor its LPIPS or SSIM versus the 3.0 VAE. For a model claiming 4K commercial-quality outputs, it is essential to verify that increased compression does not sacrifice perceptual fidelity at high spatial frequencies — a standard concern in VAE compression (e.g., the FP8-VAE literature). The omission of an ablation on VAE compression ratio vs. generation quality is a methodological gap.

### 2.3 Cross-Modality RoPE (3.0) and Its Architectural Role

A notable architectural contribution *specific to Seedream 3.0* is **Cross-modality Rotary Position Embedding (RoPE)**. In Seedream 2.0, Scaling RoPE was used for resolution generalisation. Seedream 3.0 extends this to jointly encode spatial positions of image tokens and textual positions of text tokens in a *shared* 2D positional space.

Concretely, text tokens are treated as 2D tokens of shape $[1, L]$. The column-wise position IDs of text tokens are assigned consecutively *after* the corresponding image tokens. For a 2D image token at grid position $(i, j)$ and a text token at sequence position $\ell$, their 2D RoPE positions are:

$$\text{pos}^{\text{img}}_{(i,j)} = (i, j), \qquad \text{pos}^{\text{txt}}_\ell = (1,\; j_{\max} + \ell)$$

The cross-attention between image token $(i,j)$ and text token $\ell$ then carries a positional bias:

$$\text{score}_{(i,j),\ell} = \frac{(\mathbf{q}_{(i,j)} \cdot e^{i\theta_{(i,j)}}) \cdot (\mathbf{k}_\ell \cdot e^{i\theta_{(1,j_{\max}+\ell)}})^*}{\sqrt{d}}$$

The complex phase difference $e^{i(\theta_{(i,j)} - \theta_{(1,j_{\max}+\ell)})}$ encodes the *relative geometric distance* between image pixel positions and their corresponding text descriptions in a unified 2D space. This geometric coupling is precisely why Cross-modality RoPE improves text rendering accuracy: the spatial character at position $(i,j)$ in the image "sees" its text token as positionally adjacent in the shared manifold, reinforcing local visual-linguistic alignment critical for rendering individual glyphs.

Seedream 4.0 inherits this principle; the new DiT's positional encoding scheme is not explicitly described but the VLM-based PE model (§5.2) provides an independent channel of multimodal spatial grounding at the semantic level.

### 2.4 Mixed-Resolution Training: Elevation from 2K to 4K

| Stage                       | Seedream 3.0                       | Seedream 4.0                       |
| --------------------------- | ---------------------------------- | ---------------------------------- |
| Pre-training resolution     | avg $256^2$, various aspect ratios | avg $512^2$, various aspect ratios |
| Fine-tuning resolution      | $512^2$–$2048^2$                   | $1024^2$–$4096^2$                  |
| Size embedding conditioning | Yes                                | Yes (implied)                      |

The elevation of Stage 1 from $256^2$ to $512^2$ is non-trivial: it doubles the spatial resolution at the lowest pre-training stage, meaning the model has exposure to more spatially structured training signal earlier in curriculum. This, combined with the high-compression VAE making $512^2$ pre-training cost-equivalent to $256^2$ under the 3.0 VAE, represents an effective *free upgrade* in training data quality.

---

## 3. Training Objectives

### 3.1 Seedream 3.0: Flow Matching + REPA

The stated training objective is:

$$\mathcal{L} = \mathbb{E}_{(\mathbf{x}_0, \mathcal{C})\sim \mathcal{D},\; t\sim p(t; \mathcal{D}),\; \mathbf{x}_t\sim p_t(\mathbf{x}_t|\mathbf{x}_0)} \left\|\mathbf{v}_\theta(\mathbf{x}_t, t; \mathcal{C}) - \frac{d\mathbf{x}_t}{dt}\right\|_2^2 + \lambda\, \mathcal{L}_{\text{REPA}}$$

with linear interpolant $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, so the velocity target is:

$$\frac{d\mathbf{x}_t}{dt} = \boldsymbol{\epsilon} - \mathbf{x}_0$$

This is the **Rectified Flow / flow matching objective** (Lipman et al., 2022; Liu et al., 2022). The target velocity is *constant along each trajectory* — i.e., the straight-line path from $\mathbf{x}_0$ to $\boldsymbol{\epsilon}$ has a constant vector field, which is what makes low-NFE sampling possible: a learned straight trajectory can be traced accurately in far fewer Euler steps than the curved trajectories of DDPM or score-matching objectives.

#### 3.1.1 REPA: Representation Alignment Loss

$$\mathcal{L}_{\text{REPA}} = d_{\cos}\!\left(h_\theta^{(l)}(\mathbf{x}_t), f_{\text{DINOv2}}(\mathbf{x}_0)\right)$$

where $h_\theta^{(l)}$ is the intermediate feature at layer $l$ of the MMDiT, and $f_{\text{DINOv2}}$ is a frozen DINOv2-L encoder. With $\lambda = 0.5$:

**Mechanism.** REPA functions as online *knowledge distillation* from a discriminative vision encoder into the generative model. DINOv2-L encodes semantic object structure, texture statistics, and compositional priors learned from $\sim$1B images. By regularising $h_\theta^{(l)}$ toward this manifold, REPA achieves two simultaneous effects:

1. **Low-variance semantic gradient signal**: The REPA gradient $\partial \mathcal{L}_{\text{REPA}}/\partial \theta$ is deterministic given $\mathbf{x}_0$ (no stochastic timestep sampling noise), providing a stable semantic anchor to the otherwise high-variance flow matching gradient. This explains the observed convergence acceleration.
2. **Inductive bias injection**: The model learns *what representations to build* (semantic, compositionally grounded features) before fully learning *how to denoise* along stochastic trajectories. This is analogous to representation pre-shaping in curriculum learning.

#### 3.1.2 Defect-Aware Masked Loss (Complete Derivation)

The standard flow-matching loss operates over the full latent spatial domain. Seedream 3.0's **defect-aware training paradigm** modifies this to:

$$\mathcal{L}_{\text{defect}} = \mathbb{E}\!\left[\mathbf{m} \odot \left\|\mathbf{v}_\theta(\mathbf{x}_t, t; \mathcal{C}) - \frac{d\mathbf{x}_t}{dt}\right\|_2^2\right]$$

where $\mathbf{m} \in \{0,1\}^{H_\ell \times W_\ell}$ is the binary latent-space defect mask. For a defect bounding box $\Omega_{\text{def}} \subset [0,W]\times[0,H]$ detected in pixel space, the mask is obtained by downsampling by the VAE compression factor $r$:

$$m_{i,j} = \mathbf{1}\!\left[\left(\frac{i \cdot r}{H},\, \frac{j \cdot r}{W}\right) \notin \Omega_{\text{def}}\right]$$

**Gradient flow analysis.** The standard gradient with respect to $\theta$ for spatial position $(i,j)$ is:

$$\frac{\partial \mathcal{L}}{\partial \theta}\bigg|_{(i,j)} \propto \left(v_\theta^{(i,j)} - \dot{x}_t^{(i,j)}\right) \cdot \frac{\partial v_\theta^{(i,j)}}{\partial \theta}$$

Without masking, defect regions contribute a *corrupted* gradient signal: $\dot{x}_t^{(i,j)}$ at defect positions is computed from a latent that contains watermark/subtitle artifacts, meaning the model receives a gradient that pushes $v_\theta$ toward a velocity field for defect-corrupted data. This acts as structured label noise at rate $\leq 20\%$ of the spatial domain. With masking, $m_{i,j} = 0$ at defect positions sets their gradient contribution to exactly zero, yielding a *strictly proper* learning signal over the clean spatial domain.

The 21.7% dataset expansion follows directly: samples previously discarded under the Seedream 2.0 binary keep/reject rule (which discarded *any* sample with detected defects, encompassing ~35% of the dataset) are now retained if $\text{area}(\Omega_{\text{def}}) / (W \cdot H) < 0.2$. The effective recovery rate of $21.7\%$ implies that the majority of the discarded 35% had defect areas between 0% and 20%, with the remainder (13.3%) having defects too large to recover.

> **Formal equivalence.** The masked loss is equivalent to importance-weighted empirical risk minimisation where $w(i,j) = m_{i,j}$, i.e., a self-normalised importance weight of 0 for defect sites and 1 elsewhere. This is a valid estimator of the true loss over the clean spatial distribution, consistent in the sense of converging to the correct population loss as data increases — provided the clean regions constitute the majority of each retained sample, which the 20% threshold enforces.

#### 3.1.3 Resolution-Aware Timestep Sampling (Formal Specification)

The timestep distribution $p(t; \mathcal{D})$ is constructed as a logit-normal distribution with resolution-dependent shift. Let $\sigma_{\text{LN}}$ denote the logit-normal base distribution and $s(R)$ the shift factor for average training resolution $R$:

$$p(t; \mathcal{D}) \propto \sigma\!\left(\frac{\text{logit}(t) - \mu}{\sigma_{\text{LN}}} - s(\bar{R}_\mathcal{D})\right)$$

The shift $s(\bar{R}_\mathcal{D})$ is *increasing* in $\bar{R}_\mathcal{D}$: at higher resolution, more probability mass is assigned to large $t$ (low SNR) timesteps. The theoretical justification: high-resolution images have more high-frequency spatial content. The flow matching loss at large $t$ (heavily noised $\mathbf{x}_t$) trains the model to learn coarse structure, while small $t$ trains fine detail. At higher resolution, there is *more* fine detail to learn, so upweighting large $t$ relative to the standard logit-normal distribution prevents the model from spending too many compute steps on coarse-structure denoising steps that are already well-learned.

At inference, $s$ is computed from the requested output resolution, providing a consistent resolution-conditional denoising schedule without requiring retraining.

### 3.2 Seedream 4.0: Adversarial Post-Training Objectives

Seedream 4.0 retains flow matching + REPA as its pre-training objective. The major addition is a two-stage **adversarial post-training** framework that operates after pre-training to achieve acceleration without quality loss.

#### 3.2.1 Adversarial Distillation Post-training (ADP): Stage 1

ADP trains a student (few-step) model $G_\phi$ to match the distribution of a teacher (full-step) diffusion model. Formally, given teacher samples $\mathbf{x} \sim p_{\text{teacher}}$ and student samples $G_\phi(\mathbf{z}, \mathbf{c})$ from noise $\mathbf{z}$ and conditioning $\mathbf{c}$, a hybrid discriminator $D_\psi$ is trained:

$$\max_\psi\; \mathbb{E}_{\mathbf{x} \sim p_{\text{teacher}}}[\log D_\psi(\mathbf{x})] + \mathbb{E}_{\mathbf{z}}[\log(1 - D_\psi(G_\phi(\mathbf{z}, \mathbf{c})))]$$

$$\min_\phi\; \mathbb{E}_{\mathbf{z}}[\log(1 - D_\psi(G_\phi(\mathbf{z}, \mathbf{c})))] + \mathcal{L}_{\text{consistency}}(G_\phi, f_{\text{teacher}})$$

The **hybrid discriminator** operates in both pixel space and feature space simultaneously. This prevents *mode collapse* — the principal failure mode of pure GAN training — because the feature-space discriminator penalises deviations in semantic content distribution even when pixel-space samples pass the pixel discriminator. This is analogous to perceptual discriminators in StyleGAN-XL (Sauer et al., 2022).

The consistency term $\mathcal{L}_{\text{consistency}}$ is implied to be a distance between the student's denoised outputs and the teacher's, analogous to consistency distillation (Song et al., 2023) — ensuring that even after adversarial refinement, the student does not diverge from the teacher's output manifold.

ADP provides a **stable initialisation**: the student begins adversarial training from a configuration that already mimics the teacher distribution approximately, preventing the discriminator from trivially winning early in training (the classic GAN instability problem).

#### 3.2.2 Adversarial Distribution Matching (ADM): Stage 2

Following ADP, ADM fine-tunes with a **diffusion-based discriminator** $D_{\psi_{\text{diff}}}$ that is itself parameterised as a diffusion model:

$$\min_\phi \max_\psi\; \mathbb{E}_{\mathbf{x} \sim p_{\text{teacher}}}[\log D_{\psi_{\text{diff}}}(\mathbf{x})] + \mathbb{E}_{\mathbf{z}}[\log(1 - D_{\psi_{\text{diff}}}(G_\phi(\mathbf{z})))]$$

**Why a diffusion discriminator is superior to a fixed-architecture discriminator:** A standard discriminator is a deterministic function mapping images to scalars. A diffusion discriminator can represent an arbitrary smooth density over the image manifold through its score function — enabling it to capture multi-modal output distributions (e.g., the diverse stylistic modes of realistic photography, illustration, and graphic design) that a fixed-architecture discriminator would average or truncate.

Formally, the ADM objective approximates minimising the **Kernel Stein Discrepancy (KSD)** between the student and teacher distributions:

$$\text{KSD}^2(p_{\text{student}} \| p_{\text{teacher}}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{student}}, \mathbf{y} \sim p_{\text{student}}}[h_p(\mathbf{x}, \mathbf{y})]$$

where $h_p$ is a Stein kernel defined via the score function $\nabla \log p_{\text{teacher}}$ — which the diffusion discriminator provides via its denoising score network. This is the distributional analogue of the per-timestep SSD used in Seedream 3.0: where SSD minimises a *pointwise* divergence at each timestep $t$, ADM minimises a *full-manifold* distributional divergence over the entire output space.

> **Key architectural progression.** Seedream 3.0's SSD learns *which timesteps are most informative during training*. Seedream 4.0's ADM learns *which output regions best discriminate student from teacher distributions*. The former is a training efficiency tool; the latter is a generation quality tool. They are complementary, not redundant.

---

## 4. Data Engineering Philosophy

This section addresses the central research question of the comparison and synthesises it with the literature on data-centric AI and compositional generalisation.

### 4.1 Seedream 3.0: Reactive Expansion and Distribution Correction

Seedream 3.0's data strategy is fundamentally **reactive**: it begins with a quality-filtered corpus and then *expands* and *reweights* it without synthesising new data types. Two orthogonal mechanisms implement this:

#### 4.1.1 Defect-Aware Recovery (as formalised above)

The key design insight is the **active learning engine** for defect detector annotation: rather than randomly selecting 15,000 images for human annotation, active learning selects those *nearest the classifier decision boundary* — the hardest examples for the current detector model. This maximises the information content per annotation dollar, analogous to core-set selection in active learning (Sener & Savarese, 2018). The resulting 15K labels are maximally informative for reducing detector uncertainty, enabling a robust bounding-box predictor that generalises across diverse artifact typologies (watermarks, subtitles, mosaics).

#### 4.1.2 Dual-Axis Collaborative Sampling

The dual-axis framework jointly optimises along:

**Visual axis**: Hierarchical clustering of image embeddings (likely CLIP-ViT or DINO-based) defines morphological clusters $\{C_k\}_{k=1}^K$. Sampling weight per cluster:

$$w^{\text{vis}}(C_k) \propto \frac{1}{|C_k| \cdot |\mathcal{D}_k|}$$

where $|\mathcal{D}_k|$ is the empirical frequency. This flattens the visual distribution, upweighting rare morphological categories (scientific diagrams, architectural schematics) relative to the dominant natural photography cluster.

**Textual/semantic axis**: TF-IDF over caption vocabulary. For vocabulary token $v$ with empirical caption frequency $f_v$:

$$w^{\text{sem}}(v) \propto \frac{\text{IDF}(v)}{\max_{v'} \text{IDF}(v')} = \frac{\log(|\mathcal{D}|/\text{df}(v))}{\max_{v'} \log(|\mathcal{D}|/\text{df}(v'))}$$

where $\text{df}(v)$ is the document (caption) frequency of $v$. This upweights rare semantic concepts and downweights ubiquitous n-grams ("a photo of", "portrait of a").

**Joint sampling weight.** The formal joint objective (not stated, but implied):

$$w(\mathbf{x}) = w^{\text{vis}}(\text{cluster}(\mathbf{x})) \cdot w^{\text{sem}}(\text{caption}(\mathbf{x}))$$

applied as self-normalised importance weights in the training objective:

$$\mathcal{L}_{\text{dual}} = \frac{\mathbb{E}_{\mathbf{x} \sim \mathcal{D}}[w(\mathbf{x}) \cdot \|\mathbf{v}_\theta - \dot{\mathbf{x}}_t\|_2^2]}{\mathbb{E}_{\mathbf{x} \sim \mathcal{D}}[w(\mathbf{x})]}$$

**The cross-modal retrieval augmentation** reduces the variance of this estimator: by actively retrieving additional samples for underrepresented concept-visual combinations (via targeted concept retrieval and similarity-weighted sampling), the framework increases the effective sample count for rare concepts, reducing the variance penalty of SNIS (Self-Normalised Importance Sampling) at small-frequency tail concepts.

### 4.2 Seedream 4.0: Proactive Synthesis and Structural Augmentation

Seedream 4.0's data strategy is fundamentally **proactive**: it identifies *structural gaps* in the achievable distribution of top-down resampling and synthetically fills them. This is the deeper epistemological advance.

#### 4.2.1 Identified Failure Modes of Top-Down Resampling

The report explicitly identifies two failure modes:

**Mode (a): Over-representation of natural images.** Even with visual-cluster balancing, the *prior* distribution of web-scraped data is dominated by natural photography at every level of the morphological cluster hierarchy. Cluster normalisation equalises weight *within* the cluster taxonomy, but if natural photography is the dominant leaf-level category across all major clusters, even perfectly balanced cluster sampling produces a training distribution dominated by natural images.

**Mode (b): Under-representation of knowledge-centric fine-grained data.** Mathematical expressions, chemical structures, instructional diagrams, and UI schematics have properties that defeat both axes of 3.0's resampling:
- *Visual axis*: They form a very small minority in any morphological cluster (most "diagram" clusters are dominated by infographics and bar charts, not LaTeX-rendered equations).
- *Semantic/textual axis*: Mathematical notation, chemical names, and UI terminology are rare in natural language captions, but TF-IDF upweighting of rare tokens does not directly map to *visual* generation capability for symbolic content — the bottleneck is image density, not caption token frequency.

#### 4.2.2 Knowledge Data Sub-pipeline: Natural Images from PDFs

**Source.** High-quality figures extracted from PDF documents: in-house textbooks, research articles, novels.

**Stage 1 — Low-quality filtering.** A domain-adapted classifier removes blurred, cluttered, or noisy-background images. This is distinct from Seedream 3.0's general defect detector: it is trained specifically on document-extracted image artifacts (OCR boundary noise, scan halftone patterns, JPEG compression at text edges, column layout bleed-through) rather than web image artifacts (watermarks, subtitles, mosaics).

**Stage 2 — Difficulty rating.** A 3-level classifier (easy / medium / hard) annotates all retained images. Hard samples are *down-sampled* during pre-training.

> **Why down-sample hard samples?** This is a principled **curriculum learning** strategy (Bengio et al., 2009). Let the training objective be decomposed by difficulty tier $d \in \{\text{easy, medium, hard}\}$. Hard samples (dense multi-line formulas, complex circuit diagrams, multi-column table layouts) produce high loss and high gradient variance early in training. This creates two risks: (1) *catastrophic forgetting* — the model over-adapts to the hard distribution, degrading performance on the general distribution; (2) *training instability* — large gradient norms from hard samples can destabilise Adam's adaptive learning rate estimates. Down-sampling hard samples implements an *implicit curriculum*: the model first masters the easy/medium distribution of knowledge images, building the necessary representational substrate before being exposed to extreme difficulty examples at higher data-mix ratios.

**Formal implication.** Let $p_d(\mathbf{x})$ be the empirical distribution over difficulty tier $d$. The effective sampling distribution with down-sampling factor $\alpha < 1$ for hard samples is:

$$p_{\text{eff}}(\mathbf{x}) \propto p_{\text{easy}}(\mathbf{x}) + p_{\text{med}}(\mathbf{x}) + \alpha \cdot p_{\text{hard}}(\mathbf{x})$$

This is a *mixture density* with $\alpha$ as a curriculum hyperparameter. As training progresses, $\alpha$ could in principle be annealed toward 1.0 (full inclusion), though the paper does not specify whether dynamic annealing is used.

#### 4.2.3 Knowledge Data Sub-pipeline: Synthetic Formula Images

**OCR-derived formulas.** Mathematical expressions extracted via OCR from document sources provide naturalistic typographic rendering with real-world font choices, spacing, and kerning distributions.

**LaTeX-compiled formulas.** When LaTeX source is available, it is compiled into images with *controlled structural variation*: layout (display vs. inline vs. multi-line), symbol density (sparse vs. dense), and resolution. This implements **systematic data augmentation at the semantic level**: the LaTeX source defines the ground-truth mathematical *semantics*, while the compilation parameters define the visual *rendering modality*. The model thereby receives multiple visual renderings of the same mathematical concept, encouraging learning of the invariant symbolic structure rather than surface rendering artifacts.

**Formal implication for emergent capability.** The flow-matching objective $\mathbb{E}[\|\mathbf{v}_\theta - \dot{\mathbf{x}}_t\|_2^2]$ trains the model to reconstruct the denoising velocity field for each training image. For a LaTeX-compiled formula image $\mathbf{x}_0$, this means the model's learned vector field $\mathbf{v}_\theta(\mathbf{x}_t, t)$ must encode the full spatial structure of the rendered formula at every noise level. With $K$ renderings of the same formula (varying layout, font, resolution), the model must learn a velocity field that is invariant to these surface variations and equivariant to the underlying symbolic structure. This is precisely the **compositional invariance** property required for robust formula generation at test time.

#### 4.2.4 Module-Level Upgrades

| Module               | Seedream 3.0                                         | Seedream 4.0                                                                   | Formal Implication                                                                                                                                                               |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Caption quality**  | No explicit text-quality filter on original captions | Text-quality classifier trained to detect garbled OCR, incoherent descriptions | Reduces noise in the conditioning signal $\mathcal{C}$; improves the SNR of the text-image alignment training signal                                                             |
| **Deduplication**    | Semantic cross-modal embedding only                  | Combined semantic + low-level visual embeddings                                | Catches near-duplicates that differ semantically but are visually similar (same scene, different caption) *and* vice versa, preventing artificial inflation of training set size |
| **Captioning model** | Multi-version aesthetic captions                     | Refined VLM with finer-grained visual descriptions                             | Richer, more spatially structured captions provide denser conditioning signal, directly improving spatial layout following and attribute binding                                 |
| **Retrieval engine** | Cross-modal retrieval embedding                      | Stronger cross-modal embedding                                                 | Improves concept injection, distribution calibration, and cross-modal enhancement in the retrieval-augmented training loop                                                       |

**The combined deduplication upgrade** is particularly important. Semantic-only deduplication removes pairs where the *caption embedding* is near-duplicate, but can miss pairs where two images of the same scene receive different captions (capturing different aspects of the same visual). Low-level visual embedding deduplication catches these. The *union* of both criteria provides a stricter deduplication regime that (a) reduces overfitting to specific visual scenes, and (b) prevents the model from learning spurious correlations between different captions and the same image.

Formally, the deduplication condition in 4.0 is:

$$(\mathbf{x}_i, \mathbf{c}_i) \text{ is a duplicate of } (\mathbf{x}_j, \mathbf{c}_j) \text{ iff } d_{\text{sem}}(\mathbf{c}_i, \mathbf{c}_j) < \tau_{\text{sem}} \;\mathbf{\lor}\; d_{\text{vis}}(\phi_{\text{low}}(\mathbf{x}_i), \phi_{\text{low}}(\mathbf{x}_j)) < \tau_{\text{vis}}$$

where $\phi_{\text{low}}$ extracts low-level visual features (e.g., pixel histograms, DCT coefficients, or shallow CNN features). This is strictly more inclusive than the 3.0 semantic-only criterion.

### 4.3 Comparative Summary: Data Engineering Philosophies

| Axis                        | Seedream 3.0                               | Seedream 4.0                                               |
| --------------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **Core strategy**           | Reactive: recover + reweight existing data | Proactive: synthesise new data for structural gaps         |
| **Data recovery mechanism** | Defect-aware masking (21.7% recovery)      | Not needed (new pipeline has better source data)           |
| **Distribution correction** | TF-IDF + visual cluster reweighting        | Same + knowledge sub-pipeline as structural correction     |
| **New data creation**       | None                                       | LaTeX compilation, PDF figure extraction                   |
| **Quality control**         | Defect detection (binary)                  | Multi-level difficulty classification (3-level curriculum) |
| **Deduplication**           | Semantic embedding only                    | Semantic + low-level visual (joint criterion)              |
| **Failure modes addressed** | Label noise in training data               | Structural gaps in training distribution                   |
| **Data-centric philosophy** | Improve quality of existing distribution   | Change the *support* of the training distribution          |

---

## 5. Post-Training

### 5.1 Seedream 3.0: T2I-only Multistage Post-training

The post-training pipeline is: **CT → SFT → RLHF → PE**, applied exclusively to T2I generation.

#### 5.1.1 Aesthetic Captions in SFT

Multiple caption model versions are trained to describe professional aesthetic qualities: cinematic lighting, compositional balance, texture richness, stylistic vocabulary. The SFT stage then fine-tunes the model on (aesthetic caption, high-quality image) pairs. This teaches the model to *respond* to aesthetic terminology in prompts rather than treating it as decorative language.

#### 5.1.2 VLM-based Reward Model Scaling

The reward is derived from the normalised probability of the "Yes" token from a VLM:

$$r(\mathbf{x}, \mathbf{c}) = \frac{\exp\!\left(\text{logit}_{\text{Yes}}(\mathbf{x}, \mathbf{c})\right)}{\exp\!\left(\text{logit}_{\text{Yes}}(\mathbf{x}, \mathbf{c})\right) + \exp\!\left(\text{logit}_{\text{No}}(\mathbf{x}, \mathbf{c})\right)}$$

This is a **soft binary preference signal** conditioned on the image-prompt pair. The VLM is prompted with a structured query: "Does this image faithfully depict [prompt]? Answer Yes or No."

**Advantages over CLIP-based rewards:**

1. *Compositionality*: VLMs can reason about multi-step semantic requirements. "A red bus parked beside a blue bicycle under a yellow umbrella" requires joint attribute binding. CLIP projects this to a single embedding vector that may represent the dominant attributes while discarding minority bindings. A VLM can explicitly reason: "I see a bus (yes), it is red (yes), there is a bicycle (yes), it is blue (yes)..."

2. *Scaling*: The report demonstrates empirical reward scaling from 1B to >20B parameters — *reward model capacity correlates with reward quality*. This is the image-generation analogue of the LLM RLHF scaling result (Bai et al., 2022), and has the same practical implication: investing compute in the reward model provides a reliable, scalable path to improved alignment.

### 5.2 Seedream 4.0: Unified Multimodal Post-training

The qualitative advance: Seedream 4.0 performs **joint post-training** on T2I generation, single-image editing, and multi-image composition in a *single shared model*.

#### 5.2.1 Joint Training Distribution

The training distribution includes three task classes:

- **T2I**: $(\mathbf{c}_{\text{text}}, \mathbf{x}_{\text{target}})$
- **Single-image editing**: $(\mathbf{c}_{\text{text}}, \mathbf{x}_{\text{ref}}, \mathbf{x}_{\text{target}})$
- **Multi-image composition**: $(\mathbf{c}_{\text{text}}, \{\mathbf{x}_{\text{ref}}^{(k)}\}_{k=1}^K, \mathbf{x}_{\text{target}})$

The causal diffusion structure in the DiT encodes reference images as preceding context tokens, enabling the model to condition generation on arbitrary numbers of input images through causal masking over the token sequence.

**Cross-task regularisation effect (formal).** Let $\mathcal{L}_{\text{T2I}}$ and $\mathcal{L}_{\text{edit}}$ denote the task-specific losses. Joint training minimises:

$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{T2I}} + \beta_1 \mathcal{L}_{\text{single-edit}} + \beta_2 \mathcal{L}_{\text{multi-edit}}$$

The editing objectives provide an implicit regularisation on the T2I objective: to perform faithful editing (preserve identity across edits), the model must learn identity-invariant representations of objects and faces. These same representations improve T2I *coherence* — the model generates objects with stable, internally consistent visual identities even under complex compositional prompts. This is an instance of multi-task learning's implicit regularisation benefit (Caruana, 1997; Ruder, 2017), where shared-representation tasks mutually constrain each other's hypothesis space.

**Editing data construction.** Each data triple includes captions at three levels of detail for both reference and target, which function as data augmentation at training time. The use of *consistent terminology* to describe similarities between reference and target is non-trivial: it forces the model to develop a shared semantic vocabulary for *identity-preserving features* (e.g., "the same facial structure", "the same hair colour") that can be referenced at inference time via prompt tokens, enabling zero-shot identity preservation.

#### 5.2.2 VLM Prompt Engineering Model (Seed1.5-VL)

The PE model performs:

| Function                    | Description                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| **Task routing**            | Classifies input as T2I / single-edit / multi-ref                         |
| **Prompt rewriting**        | Expands under-specified prompts, resolves ambiguity                       |
| **AdaCoT thinking**         | Dynamically adjusts reasoning budget based on task complexity             |
| **Aspect ratio estimation** | Infers semantically appropriate canvas dimensions                         |
| **Reference captioning**    | Generates structured captions for reference images at three detail levels |

**AdaCoT (Adaptive Chain of Thought)** is the key innovation: simple prompts bypass CoT to minimise latency; complex multi-attribute or compositional prompts receive extended reasoning that decomposes the semantic requirements before encoding them into DiT conditioning. This is inspired by the LLM literature on budget-adaptive inference (e.g., Yue et al., 2025), now applied to the *conditioning preparation* stage of image generation rather than output token generation.

> **Reviewer note.** The adaptive thinking mechanism introduces a *reasoning-time compute trade-off*: complex prompts receive more PE model compute, increasing latency for hard cases. The paper does not provide a latency distribution across prompt complexities or a Pareto analysis of quality vs. PE compute budget. This is a necessary evaluation for a production system.

---

## 6. Inference Acceleration

### 6.1 Seedream 3.0: Instance-Specific Trajectories + SSD Importance Sampling

#### 6.1.1 Consistent Noise Expectation

Standard flow matching converges all samples to the same isotropic Gaussian prior $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, causing trajectory overlap in probability space. Seedream 3.0 introduces an instance-specific noise target $\boldsymbol{\mu}_\epsilon(\mathbf{x}_0)$ estimated from a pretrained model:

$$\mathbf{x}_T^*(\mathbf{x}_0) = \boldsymbol{\mu}_\epsilon(\mathbf{x}_0) + \sigma_T \cdot \mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Theoretical justification.** The report claims this "maximises the probability of the forward-backward path from data to noise and back." Formally, this is equivalent to finding the forward process that minimises the Wasserstein-2 distance between the data distribution $p_{\text{data}}$ and the noise distribution at time $T$:

$$\min_{\{p_t\}} \mathcal{W}_2(p_{\text{data}}, p_T) \quad \text{subject to path straightness}$$

Under the optimal transport interpretation of flow matching (Lipman et al., 2022), the straight-line path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\mu}_\epsilon(\mathbf{x}_0)$ achieves lower path curvature than the path to the universal Gaussian prior, because $\boldsymbol{\mu}_\epsilon(\mathbf{x}_0)$ is "closer" to $\mathbf{x}_0$ in the transport map than an arbitrary Gaussian sample. Straighter trajectories require fewer ODE integration steps for the same accuracy — the theoretical basis for the 4–8× NFE reduction.

#### 6.1.2 Stochastic Stein Discrepancy for Timestep Importance Sampling

A neural network $f_\phi$ learns a data-dependent timestep distribution $p_\phi(t \mid \mathbf{x}_0)$ by minimising the Stochastic Stein Discrepancy (SSD):

$$\min_\phi\; \text{SSD}(p_\phi(\cdot \mid \mathbf{x}_0) \;\|\; p^*(\cdot \mid \mathbf{x}_0))$$

where $p^*(t \mid \mathbf{x}_0) \propto \mathcal{L}(t, \mathbf{x}_0)$ is the loss-proportional ideal distribution (assigning high probability to timesteps with high loss contribution). The SSD is used because $p^*$ is unnormalised and intractable to normalise — the kernelised Stein operator provides a tractable divergence measure that does not require normalisation:

$$\text{SSD}(p \| q) = \mathbb{E}_{\mathbf{t} \sim p}\!\left[k_q(\mathbf{t}, \mathbf{t}') \cdot \mathcal{S}_q(\mathbf{t}) \cdot \mathcal{S}_q(\mathbf{t}')\right]$$

where $\mathcal{S}_q(t) = \nabla_t \log q(t) - \nabla_t \log p(t)$ is the Stein score differential and $k_q$ is a Stein kernel. This network learns which timesteps contribute most to the training loss for each training sample, dynamically prioritising the most informative denoising steps — an adaptive curriculum over the noise dimension.

### 6.2 Seedream 4.0: Three-Pillar Acceleration System

#### 6.2.1 ADP → ADM Pipeline (as derived in §3.2)

The adversarial distillation pipeline replaces the SSD-based importance sampling with a *generative* acceleration mechanism. Where Seedream 3.0 improves training efficiency by focusing on informative timesteps, Seedream 4.0 distills the full multi-step generative process into a few-step student — a fundamentally different approach that produces the acceleration at inference time.

The theoretical distinction:
- **Seedream 3.0 acceleration**: Reduces NFE by straightening trajectories (fewer integration steps needed for straight-line paths)
- **Seedream 4.0 acceleration**: Reduces NFE by distilling many-step teacher into few-step student (the student learns to "skip" intermediate denoising steps)

Both reduce NFE, but through complementary mechanisms. They are not mutually exclusive, and Seedream 4.0 likely benefits from inheriting the straight-trajectory design as a prior for the distillation.

#### 6.2.2 Adaptive 4/8-bit Hybrid Quantization

The quantization objective for each layer $\ell$ is:

$$\min_{s_\ell, g_\ell, b_\ell} \|\mathbf{W}_\ell - Q_{s_\ell, g_\ell, b_\ell}(\mathbf{W}_\ell)\|_F^2 \quad \text{s.t.} \quad \text{HW\_efficiency}(b_\ell, g_\ell) \geq \tau$$

where $b_\ell \in \{4, 8\}$ bits, $g_\ell$ is quantization granularity (per-tensor/per-channel/per-group), $s_\ell$ is the scaling factor, and $\tau$ is a hardware efficiency threshold. **Offline smoothing** redistributes activation outliers across channels before quantization, following the principle of SmoothQuant (Xiao et al., 2023): outliers in activations cause large quantization errors in 4-bit regime; redistributing them makes the quantized representation more uniform. The **search-based optimization** finds the per-layer optimal $(s_\ell, g_\ell, b_\ell)$ tuple, analogous to mixed-precision quantization search (HAWQ, Kim et al., 2021).

Critically, **hardware-specific CUDA kernels** are co-designed with the quantization scheme: different bit widths and granularities have different memory access patterns and compute utilisation profiles. Without hardware-matched kernels, quantized models often fail to realise theoretical FLOPs savings due to memory bandwidth bottlenecks — the 4.0 co-design approach addresses this.

#### 6.2.3 Speculative Decoding for the VLM PE Model

The PE model (Seed1.5-VL) is autoregressive, introducing a sequential bottleneck. Standard speculative decoding uses a draft model to propose $K$ tokens in parallel, accepted/rejected by the full model:

$$\text{Accept token } k \text{ iff } \frac{P_{\text{full}}(t_k \mid t_{<k})}{P_{\text{draft}}(t_k \mid t_{<k})} \geq U[0,1]$$

**Seedream 4.0's advance over standard speculative decoding.** The key problem: VLM token sampling is stochastic (top-$p$ or temperature sampling), introducing ambiguity in draft model training because the "target" for the draft model is not deterministic. Seedream 4.0 solves this by conditioning feature prediction on both the preceding feature sequence *and* a token sequence advanced by one timestep — providing a **deterministic target** that removes the sampling ambiguity:

$$\hat{\mathbf{h}}^{\text{draft}}_{k} = f_{\text{draft}}(\mathbf{h}_{<k}, t_{k+1:k+K})$$

The additional training losses:

$$\mathcal{L}_{\text{spec}} = \mathcal{L}_{\text{CE}}(\text{logits}_{\text{draft}}, t^*) + \lambda_{\text{KV}} \|\mathbf{KV}_{\text{draft}} - \mathbf{KV}_{\text{full}}\|_2^2$$

The **KV-cache loss** is a novel contribution: it trains the draft model to produce key-value representations that are *directly compatible with efficient reuse* in the full model's KV cache. This reduces the verification overhead (the full model need not recompute KV values from scratch for draft-accepted tokens) and increases the effective acceptance rate, translating to greater practical speedup beyond the theoretical acceptance-rate analysis.

### 6.3 Speed Comparison

| Model        | Image resolution | Inference time (no PE) | Improvement                      |
| ------------ | ---------------- | ---------------------- | -------------------------------- |
| Seedream 3.0 | 1K               | ~3.0 s                 | —                                |
| Seedream 4.0 | 2K               | ~1.4 s                 | >2× faster at 4× the pixel count |

The effective speedup in pixels/second is $(2048^2 / 1024^2) \times (3.0 / 1.4) \approx 8.6\times$ — consistent with the stated >10× FLOPs reduction given that FLOPs and wall-clock time are not identical (memory bandwidth, quantization overhead, and hardware utilisation affect the ratio).

---

## 7. Benchmark Methodology

### 7.1 Seedream 3.0: Bench-377

**Design.** 377 prompts across five scenarios: cinematic, arts, entertainment, aesthetic design, practical design. Human expert evaluation on three criteria: text-image alignment, structural correctness, aesthetic quality.

**Text rendering metrics** (formally defined in the report):

$$R_a = \left(1 - \frac{N_e}{N}\right) \times 100\% \qquad (\text{accuracy rate, edit-distance based})$$

$$R_h = \frac{N_c}{N} \times 100\% \qquad (\text{hit rate, character-level})$$

$$R_{\text{avail}} = \text{proportion of images deemed perceptually acceptable}$$

**Formal relationship between $R_a$ and $R_h$.** $R_a$ is edit-distance based (Levenshtein), while $R_h$ is a strict character-level match. For pure substitution errors, $R_a = R_h$. For insertion errors, $R_a < R_h$ (edit distance counts the insertion but $N_c$ still counts correctly placed characters). The near-equivalence of 94% for both metrics in Seedream 3.0 implies that the dominant error mode is character substitution rather than structural errors (missing characters, layout failures) — validating the claim of "minimal layout or medium-related rendering errors."

> **Reviewer critique of Bench-377.** (1) At 377 prompts, the benchmark is statistically underpowered for fine-grained per-dimension analysis across five scenarios (~75 prompts per scenario). Confidence intervals on scenario-level scores are wide. (2) The report does not report inter-annotator agreement (Cohen's $\kappa$ or Krippendorff's $\alpha$) for the human evaluation — a standard methodological requirement. (3) The five-scenario taxonomy conflates task difficulty and domain, making it hard to attribute performance differences to capability vs. prompt complexity. (4) Bench-377 evaluates only T2I — no editing or multi-modal tasks are benchmarked, reflecting the model's own scope but limiting generalisability claims.

### 7.2 Seedream 4.0: MagicBench 4.0 + DreamEval

#### 7.2.1 MagicBench 4.0

Three tracks with explicit bilingual evaluation:

| Track                | Prompts | Added dimensions vs. Bench-377                                                           |
| -------------------- | ------- | ---------------------------------------------------------------------------------------- |
| T2I                  | 325     | Dense text rendering; **content understanding** (in-context reasoning, domain knowledge) |
| Single-image editing | 300     | Instruction following vs. consistency trade-off; text editing performance                |
| Multi-image editing  | 100     | GSB metric; alignment + consistency + structural integrity                               |

**The content understanding dimension** is the key addition: it directly targets knowledge-centric generation capability (domain-specific prompts requiring LaTeX, chemistry, UI design, chart generation) that was unrepresented in Bench-377's practical design category. This is a direct methodological reflection of the data pipeline upgrade.

**Bilingual design** (each prompt in Chinese and English) is methodologically important for a production system: it separates genuine language-conditional capability from evaluator language bias, and directly tests whether performance differences between languages reflect model limitations or annotation artefacts.

#### 7.2.2 DreamEval: The Principal Evaluation Advance

$$\text{DreamEval}: 128 \text{ sub-tasks} \times \sim 12.5 \text{ prompts/sub-task} = 1{,}600 \text{ prompts total}$$

**Fine-grained VQA scoring.** Each prompt is evaluated via a set of binary VQA questions targeting specific attributes: "Does the image contain a bar chart?", "Does the bar chart's y-axis label correctly read '$\texttt{Frequency}$'?", "Are the bars correctly ordered by value?". The final score is the fraction of correct VQA answers, making evaluation **interpretable** (which attribute failed?) and **deterministic** (no annotator variance — the VQA scorer's response is fixed for a given image-question pair).

**Tiered difficulty.** Three tiers:

| Tier   | Description                                                                 | What it tests              |
| ------ | --------------------------------------------------------------------------- | -------------------------- |
| Easy   | Basic generation: single object, simple composition                         | Fundamental capability     |
| Medium | Multi-attribute: spatial relationships, counting, attribute binding         | Advanced generation        |
| Hard   | Reasoning: multi-step inference, domain knowledge, consistency across edits | Higher-order understanding |

Seedream 4.0's degradation at the Hard tier, particularly for single-image editing, provides a *specific, actionable finding*: the model needs more multi-modal reasoning data (acknowledged in the report). This diagnostic granularity is entirely absent from Bench-377's flat benchmark design.

> **Reviewer note on DreamEval methodology.** VQA-based scoring has one critical vulnerability: the VQA scorer's own capabilities create a ceiling on evaluation accuracy. If the scorer cannot correctly answer "Is this a valid structural formula of benzene?", it cannot accurately evaluate chemical generation. The report does not specify which VLM is used as the DreamEval scorer. For a VQA-based benchmark to be methodologically sound, the scorer should be (a) from a different model family than the evaluated model to avoid self-evaluation bias, and (b) validated against human judgments on a calibration set. These details are absent. Additionally, the "best-of-4" evaluation mode (Seedream 4.0 has better best-of-4 than average) is equivalent to an implicit *rejection sampling* protocol; reporting it alongside average-of-1 scores without specifying the sampling budget creates a comparison bias against models evaluated under different sampling protocols.

---

## 8. Synthesis: Knowledge-Centric Generation, Compositional Generalisation, and Data-Centric AI

This section directly addresses the key research question: *how does knowledge-centric data curation relate to Seedream 4.0's emergent capability in generating LaTeX formulas, chemical equations, UI schematics, and charts?*

### 8.1 The Capability–Data Causal Chain

The causal chain is direct and mechanistically grounded:

$$\text{LaTeX-compiled training images} \xrightarrow{\text{flow matching}} \text{learned velocity field over formula manifold} \xrightarrow{\text{inference}} \text{formula generation capability}$$

The flow-matching objective trains $\mathbf{v}_\theta$ to reconstruct the denoising trajectory for every training image. For a formula image $\mathbf{x}_0^{\text{formula}}$, this means $\mathbf{v}_\theta(\mathbf{x}_t, t; \mathbf{c}_{\text{formula}})$ must capture:

1. **Syntactic structure**: The spatial arrangement of operators, variables, and structural elements (fraction bars, summation limits, subscripts) — the visual *grammar* of mathematical notation.
2. **Semantic correspondence**: The mapping between the caption $\mathbf{c}_{\text{formula}}$ (e.g., "the quadratic formula") and the specific visual configuration of the formula image.
3. **Rendering invariance**: From multiple LaTeX compilations (different fonts, layouts, symbol densities), the model learns which visual properties are *essential* to the formula's identity (the symbolic structure) vs. *incidental* (font choice, border width).

A model **can only generate what it has been trained to reconstruct** — this is a fundamental property of the flow-matching training objective with no exceptions. The prior system (Seedream 3.0) could generate approximate mathematical symbols because such symbols appear incidentally in natural photographs (textbook covers, whiteboard photos), but could not generate *structurally valid* formulas because the training distribution lacked systematic coverage of formula sub-structure.

### 8.2 Compositional Generalisation: Literature Connection

**Lake et al. (2019)** formalise compositional generalisation as the ability to understand or produce novel combinations of known sub-structures, when the sub-structures themselves have been seen in training but not in the specific combination. For formula generation:

- **Sub-structures** (seen in training): $\int$, $\frac{a}{b}$, $\sum_{i=0}^n$, $\sqrt{x}$, Greek letters, subscripts, superscripts
- **Novel combination** (test time): $\int_0^\infty \frac{e^{-x^2}}{\sqrt{\pi}} dx = \frac{1}{2}$ — a specific integral formula not present verbatim in training

Lake et al. show empirically that systematic coverage of sub-structural components in the training distribution is *necessary* for compositional generalisation. The LaTeX sub-pipeline directly implements this: by compiling diverse formulas with structural variation (layout, density), the training distribution achieves high coverage of the relevant sub-structural atoms.

**Bogin et al. (2022)** extend this to show that *functional compositionality* (not just lexical) requires examples where each sub-function appears in multiple compositional contexts. The structural variation in Seedream 4.0's synthetic formula data — the same symbol in display mode, inline mode, within a matrix, within a nested fraction — provides exactly this multi-context coverage.

### 8.3 Data-Centric AI Perspective

Ng (2021) and Zha et al. (2023) argue that for many tasks, systematic data quality and coverage improvements yield more reliable capability gains than architectural scaling alone. Seedream 4.0's knowledge sub-pipeline embodies three core data-centric principles:

**Targeted data collection.** Rather than improving the model architecture to handle rare data better, the pipeline proactively *creates* training data for underrepresented types. The causal arrow: better data → better capability, not better model → train on bad data. This is the fundamental epistemological shift.

**Structured quality labelling.** The 3-level difficulty classifier provides structured quality metadata analogous to confidence scores in label-noise learning (Northcutt et al., 2021). It enables principled curriculum training rather than the binary keep/reject decisions of Seedream 3.0. The curriculum label is not just a quality signal — it's a *learning signal* about the optimal training schedule.

**Task-specific synthetic data generation.** LaTeX-compiled formula images are an instance of synthetic data generation for a capability gap — a strategy with strong empirical support in NLP (Wei et al., 2022: "self-instruct"; Wang et al., 2023: "phi-1") and now validated in image generation. The key enabling condition is the availability of a *compositional formal language* (LaTeX) that can generate arbitrarily diverse visual outputs from a simple generative grammar — a condition satisfied for formulas, chemical structures (SMILES → 2D structure renderers), and UI schematics (wireframe markup languages).

### 8.4 Why Does This Yield Knowledge-Centric *Generation* and Not Just Recognition?

A legitimate question: training on formula images teaches the model to *denoise* formulas, not to *generate* them on demand. The generation capability arises from the *text-image conditioning* mechanism. The training objective is:

$$
\mathcal{L} = \mathbb{E}\!\left[\left\|\mathbf{v}_\theta(\mathbf{x}_t, t; \underbrace{\mathbf{c}_{\text{formula}}}_{\text{caption}}) - \dot{\mathbf{x}}_t\right\|_2^2\right]
$$

The model learns the *conditional* velocity field $\mathbf{v}_\theta(\cdot; \mathbf{c}_{\text{formula}})$ — the denoising velocity *given* a formula caption. At inference, prompting the model with "generate the quadratic formula" produces a conditioning vector $\mathbf{c}$ that, via the learned text encoder and cross-attention, activates precisely the velocity field trained on quadratic formula images. The generation capability is the *inverse* of the denoising capability, mediated by the learned text-to-visual-feature mapping. This is why knowledge-centric data curation directly produces knowledge-centric generation: it populates the conditional velocity field with valid, compositionally grounded trajectories for formula/chart/UI concepts that were previously absent or poorly covered.

---

## 9. Remaining Limitations and Open Questions

### 9.1 Hard-Level Task Degradation in DreamEval

Seedream 4.0 shows performance degradation at the Hard tier of DreamEval, particularly in single-image editing. The report attributes this to insufficient multi-modal understanding and reasoning data. The architectural solution is already indicated: scaling with reasoning-focused data (as demonstrated by the Seedream 4.5 scaling result). The underlying theoretical issue is that Hard-level tasks require *compositional reasoning across modalities* — inferring causal relationships, physical constraints, and long-range dependencies — which is a distinct capability from pattern-completion-level generation that current training paradigms handle well.

### 9.2 Output Variance in DreamEval

Seedream 4.0 exhibits greater output variance than GPT-Image-1 at the average level, but better best-of-4 performance. This variance is mechanistically explained by the adversarial training regime: GAN-based distillation produces a model that maps the same noise input to diverse outputs by design (to avoid discriminator exploitation of mode collapse). In practice, this variance is a *user-experience* problem (unreliable single-shot results) with a simple solution (best-of-N sampling), but it represents a fundamental tension between the diversity-inducing adversarial objective and the consistency requirements of professional use cases.

### 9.3 VAE Reconstruction Quality at 4K

The paper does not evaluate the high-compression VAE's reconstruction fidelity at 4K resolution. At higher compression ratios, the decoder must hallucinate high-frequency detail that was discarded by the encoder — a form of learned inpainting. For knowledge-centric content (LaTeX formulas, circuit diagrams) where precise symbol reproduction is required, VAE reconstruction artifacts could systematically corrupt the rendered output. This is an unaddressed vulnerability.

### 9.4 Speculative Decoding Acceptance Rate Under Distribution Shift

The VLM PE model's speculative decoding is evaluated at training-distribution prompts. Under significant distribution shift (unusual prompt structures, multilingual inputs, rare domain vocabulary), the draft model's acceptance rate may degrade substantially, eliminating the latency benefit. The paper provides no analysis of acceptance rate as a function of prompt complexity or out-of-distribution degree.

### 9.5 Benchmark Self-evaluation Bias

If DreamEval uses a VLM scorer from the same model family as Seedream, the evaluation is partially tautological. Similarly, the Artificial Analysis Arena's Elo scores reflect general user preference but not fine-grained capability assessment — a model with superior aesthetics but inferior compositional accuracy may outrank a more capable but less visually striking model. A fully rigorous benchmark would require externally validated, capability-specific sub-evaluations.

---

*Analysis compiled from: Seedream 3.0 Technical Report (ByteDance Seed) and Seedream 4.0: Toward Next-generation Multimodal Image Generation (ByteDance Seed). All formalisations marked "implied but not stated" are the author's derivations from stated properties; all explicit equations are quoted or directly derived from the source documents.*