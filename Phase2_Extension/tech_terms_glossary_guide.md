## Project: EC-FVLC (Faithful Vision-Language Concepts) — DA351 Phase 2

## 1. Core Model & Architecture Terms

### 1.1 Black-Box Model

- \** \*\* An AI where you feed an image in, and an answer comes out, but nobody (not even the engineer) knows *why\* the AI chose that answer. All the millions of internal math weights are a complete mystery.
- **Analogy:** A doctor who looks at an X-ray and simply says _"You have pneumonia,"_ but refuses or cannot explain what spot on the X-ray made them say that.

---

### 1.2 Concept Bottleneck Model (CBM)

- \**\*\* Instead of going directly from *Image $\rightarrow$ Final Class\*, we force the AI to take a mandatory two-step detour through human-understandable concepts:
  $$\text{Image} \longrightarrow \text{Concept Layer (e.g., "yellow belly", "pointy beak")} \longrightarrow \text{Final Class (e.g., "Goldfinch")}$$
- **Analogy:** A doctor who is forced to write a checklist first:
  1. _Spots in lung visible? [Yes]_
  2. _Fluid buildup? [Yes]_  
     $\rightarrow$ Therefore, diagnosis is Pneumonia. Now you can inspect the doctor's reasoning!
- **Why it matters:** The concept layer is the **"bottleneck"**. If an AI says a bird is a woodpecker because it has a "sharp beak", a human expert can audit the decision.

---

### 1.3 Label-Free CBM (LF-CBM)

- \***\* Normally, training a CBM requires thousands of expensive human annotations (humans marking which bird has a yellow beak, red crown, etc.). **LF-CBM\*\* is a technique that creates concepts automatically without human labels by asking large language models (like GPT) and vision models (like CLIP).
- **Role in our project:** This is our **baseline / starting point**. It works, but its concept predictions can be unstable.

---

### 1.4 FVLC (Faithful Vision-Language Concepts) — The Original Paper (ICLR 2024)

- \*\*\*\* The original research paper we were assigned. The authors (Lai et al.) noticed that standard concept models are fragile: if you rotate or blur an image slightly, the concept scores flip around randomly. FVLC added 4 math loss terms ($L_1$ to $L_4$) to freeze and stabilize the concept explanations under noise.
- **Our finding:** FVLC made models stable, but it accidentally made them **blind to real visual changes**.

---

### 1.5 EC-FVLC (Evidence-Consistent FVLC) — OUR Extension!

- \**\*\* Our proposed improvement! We keep all 4 of FVLC's stability losses, but we add a 5th loss term ($L_5$). $L_5$ forces the concepts to follow the actual visual evidence: when evidence disappears (like during an occlusion), the concept score *must\* go down.

---

### 1.6 CLIP (Contrastive Language-Image Pretraining)

- \*_\*\* A famous neural network built by OpenAI that was trained on 400 million internet image-text pairs. It can look at any image patch and tell you how well a text phrase (like _"a photo of a long beak"\*) matches what is visible in that picture.
- **Role in our project:** We use CLIP as our **"referee"** or **"ground-truth evidence detector"**. When we cut out a piece of an image, CLIP tells us how much the visual evidence dropped.

---

### 1.7 Held-Out CLIP Model (Oracle CLIP vs. Training CLIP)

- \***\* "Held-out" means a model that was **locked in a vault during training\*\* and only brought out during the final exam.
- **Why it's crucial:** If you use Model X to train your AI, and then evaluate your AI using Model X, your AI might just be memorizing Model X's quirks (circular reasoning). To prove our method works genuinely, we train with a smaller CLIP (ViT-B/32), but evaluate using a much bigger, independent **ViT-L/14 CLIP** that our model never saw.

---

## 2. The Problem & The Research Gap

### 2.1 Faithfulness vs. Stability (The Big Distinction!)

- \*_\*\* _"Does my explanation stay the same when you disturb the image?"\*  
  (Example: If I add slight camera glare to a bird photo, the explanation shouldn't suddenly say "dog".)
- **Faithfulness to Evidence:** _"Does my explanation reflect what is ACTUALLY visible right now?"_  
  (Example: If I physically paste a black square over the bird's head, the explanation for 'red crown' MUST disappear.)
- **The Conflict:** FVLC cared _only_ about stability. Because of that, even when an object was 50% covered up, FVLC's concepts barely changed!

---

### 2.2 The "Constant Explainer" (Our Killer Pilot Proof)

- \*\*\*\* A dummy model that doesn't even look at the image. It outputs the exact same numbers for every single picture in the world.
- **The Shocking Result:** On FVLC's own stability test, this completely useless dummy model scored **0.000 (a perfect score!)**, outperforming FVLC itself.
- **Why this is our strongest argument:** It proves mathematically that **measuring stability alone is flawed**. A broken clock that never moves is 100% "stable", but completely useless.

---

### 2.3 Nuisance Edits vs. Semantic/Evidence-Altering Edits

- **Semantic / Evidence-Altering Edit:** Cutting off a bird's tail, covering an airplane's wings. Here, **concepts SHOULD change**.
- **Nuisance Edit:** Changing the background wallpaper color or adding mild camera noise. The airplane is still fully there, so **concepts SHOULD NOT change**.
- **FVLC treated all edits the same.** EC-FVLC teaches the AI the difference.

---

## 3. Our Solution & Loss Terms ($L_1$ to $L_5$)

During training, neural networks learn by trying to minimize a "Loss" (a penalty score for making mistakes).

| Loss Term          | Who Invented It?       | What it Penalizes / Enforces | Plain English Meaning                                                                       |
| ------------------ | ---------------------- | ---------------------------- | ------------------------------------------------------------------------------------------- |
| **$L_1$**          | Lai et al. (FVLC)      | Prediction Accuracy          | "Make sure the final classification is correct."                                            |
| **$L_2$**          | Lai et al. (FVLC)      | Concept Similarity           | "Concept scores should match CLIP text descriptions."                                       |
| **$L_3$**          | Lai et al. (FVLC)      | In-Pair Stability            | "Clean image and slightly noisy image should have similar concept rankings."                |
| **$L_4$**          | Lai et al. (FVLC)      | Worst-Case Robustness        | "An adversarial attack shouldn't flip the top-k concepts."                                  |
| **$\mathbf{L_5}$** | **OUR TEAM (Phase 2)** | **Concept Infidelity**       | **"If CLIP sees visual evidence drop by 40%, the model's concept score MUST drop by 40%!"** |

---

### 3.1 Lambda 5 ($\lambda_5$)

- \***\* A dial (volume knob) between $0.0$ and $1.0$ that controls **how strongly\*\* the AI cares about the new $L_5$ loss during training.
  - If $\lambda_5 = 0$: You just have original FVLC.
  - If $\lambda_5$ is too high (e.g., $1.0$): The AI panics about matching CLIP and forgets how to classify images accurately.
  - **Our tuned values:** $\lambda_5 = 0.3$ for CIFAR datasets, and $\lambda_5 = 0.1$ for CUB birds. (Sweet spots where evidence correlation jumps, but classification accuracy remains undamaged).

---

### 3.2 Pearson Correlation ($r$)

- \*\*\*\* A standard high-school statistics score between $-1$ and $+1$ that measures whether two lists of numbers move together in harmony.
  - $+1.0$: Perfect lockstep. (When evidence goes down by 2, concept score goes down by 2).
  - $0.0$: Complete randomness. No relationship.
  - $-1.0$: Complete opposite.
- **In $L_5$:** We calculate Pearson $r$ between **[CLIP evidence changes]** and **[Model concept changes]**, and we penalize $(1 - r)$. If $r = 1.0$, loss is $0$!

---

## 4. Image Edits & Perturbations

To train and test our model, we apply 4 distinct operations to images:

1. **Occlude:** A solid grey box is pasted over 25% to 50% of the image. (Hides body parts).
2. **Crop:** The camera zooms into a small corner (20% to 50%) and stretches it back. (Removes outer parts).
3. **CutMix:** A rectangular patch from a completely different image (e.g., pasting a cat head onto an airplane) is stamped on top.
4. **Nuisance:** Slight tweaks to brightness, contrast, JPEG compression, or Gaussian noise. The object is completely intact.

---

## 5. Evaluation Metrics Explained

When showing the results table to the professor, these are the exact metric names you will see:

### 5.1 Held-Out Evidence Correlation (Primary Metric) $\uparrow$ (Higher is Better)

- **What it measures:** How well the model's concept changes match the held-out ViT-L/14 CLIP model's evidence changes.
- **Our numbers:**
  - CIFAR-10: $0.237 \rightarrow \mathbf{0.331}$ (**+39%**)
  - CIFAR-100: $0.185 \rightarrow \mathbf{0.310}$ (**+68%**)
  - CUB-200: $0.186 \rightarrow \mathbf{0.319}$ (**+71%**)

---

### 5.2 Nuisance TCPC $\downarrow$ (Lower is Better)

- **What it measures:** When we apply a _nuisance_ edit (just changing brightness/noise), how much do the concept scores drift?
- **Why lower is better:** Since the object didn't change, we want concept drift to be as close to **zero** as possible!
- **Our numbers:** Dropped by **51%** on CIFAR-10, **68%** on CIFAR-100, and **28%** on CUB-200. (EC-FVLC stays much calmer under noise).

---

### 5.3 TCPC (Top-Concept Pair Consistency)

- **What it is:** FVLC's original metric. It looks at the top concepts chosen for image A and image B, and calculates the mathematical distance between them.
- **Remember:** Lower distance = more consistent/stable.

---

### 5.4 Attribute AUROC (Human Biologist Ground-Truth) $\uparrow$ (Higher is Better)

- **What it is:** CUB-200 has an official dataset where professional human ornithologists (bird scientists) manually labeled attributes for birds (e.g., _"this species has a curved bill"_).
- **AUROC scale:** $0.50$ is pure random guessing (like flipping a coin). $1.00$ is a perfect doctor/biologist.
- **Our finding:**
  - Original FVLC dropped AUROC from $0.690$ down to $0.645$ (making concepts less biologically true).
  - **EC-FVLC recovered it back to $0.685$!**
- **Why this is huge:** No CLIP was used here. This is real human biology ground-truth confirming our model is better.

---

### 5.5 Ablation Study ("FVLC + edits, no $L_5$")

- **What an ablation is in AI research:** Disabling one part of your system on purpose to prove that specific part was actually doing the magic, and not something accidental.
- **What we did:** We trained FVLC with all the edited images, but **turned off the $L_5$ math loss**.
- **Result:** It only reached $0.244$ (barely higher than baseline $0.237$). This proves that seeing edited pictures isn't enough—**the $L_5$ loss function is the true reason for the +71% improvement.**

---

## 6. Datasets Used

| Dataset Name     | What is in it?                                               | Number of Classes | Number of Concepts      |
| ---------------- | ------------------------------------------------------------ | ----------------- | ----------------------- |
| **CIFAR-10**     | Low-res 32x32 images (airplanes, cars, birds, cats, dogs...) | 10                | 131                     |
| **CIFAR-100**    | 100 fine-grained object categories                           | 100               | 204                     |
| **CUB-200-2011** | High-res photos of 200 North American bird species           | 200               | 110 (with human labels) |
