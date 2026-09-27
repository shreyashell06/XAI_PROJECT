"""FVLC: Faithful Vision-Language Concept layer (Lai et al., ICLR 2024), Eq. (2) / Algorithm 1.

min_{W~c} E_x[ l1 * KL(y(x,c~), y(x,c))                       (L1: closeness of prediction)
             + l2 * L_k1(g~(x), g(x))                          (L2: similarity of explanation)
             + l3 * max_{|d|<=R2} KL(y(x,c~), y(x,c~+d))       (L3: stability of prediction)
             + l4 * max_{|r|<=R1} L_k2(g~(x), g~(x)+r) ]       (L4: stability of explanation)

Only W~c is trained. Backbone, CLIP and the final layer W_F stay frozen (Fig. 2).
Inner maximisations use l_inf PGD (Madry et al., 2018); outer minimisation uses Adam.

Implementation notes (see README "Ambiguities"):
* `pgd_mode="universal"` follows Algorithm 1 literally: one delta / rho shared by all inputs, updated with
  gradients averaged over freshly sampled batches. `"per_sample"` gives every input its own delta / rho.
* `rho_space="feature"` puts rho on the backbone features b_f(x) (Table 5 calls it "x pgd"), measured in units
  of each feature dimension's training std: b_f(x) + rho * std(b_f). Raw CLIP-RN50 features have a per-dim
  std of ~0.025, so an unscaled l_inf radius of 0.05 would be a ~2-sigma shift of *every* dimension.
  Taken literally in concept space (`"concept"`), L_k2(c~, c~+rho) only depends on W~c through the
  (non-differentiable) top-k index sets, so L4 would contribute zero gradient.
"""
import math

import torch

from .cbm import ConceptLayer, FinalLayer


def kl(target_logits, logits):
    """D = KL(softmax(target) || softmax(logits)), batch mean. Differentiable in both arguments."""
    p = torch.log_softmax(target_logits, dim=1)
    q = torch.log_softmax(logits, dim=1)
    return (p.exp() * (p - q)).sum(dim=1).mean()


def topk_surrogate(c, c_tilde, k):
    """Eq. (7): L_k(c, c~) = 1/(2k) (||c_S(c) - c~_S(c)||_1 + ||c~_S(c~) - c_S(c~)||_1), batch mean.
    S(v) = indices of the k largest entries of v (treated as constants)."""
    s_c = torch.topk(c.detach(), k, dim=1).indices
    s_t = torch.topk(c_tilde.detach(), k, dim=1).indices
    a = (c.gather(1, s_c) - c_tilde.gather(1, s_c)).abs().sum(dim=1)
    b = (c_tilde.gather(1, s_t) - c.gather(1, s_t)).abs().sum(dim=1)
    return ((a + b) / (2 * k)).mean()


class FVLCTrainer:
    def __init__(self, cfg, final: FinalLayer, init_layer: ConceptLayer, ref_layer: ConceptLayer, device,
                 use_l2=True, use_l3=True, use_l4=True, seed=0, verbose=True):
        """init_layer: W~c starts from init_layer.W (the perturbed LF-CBM, `fvlc_init="base"`).
        ref_layer : the vanilla layer; g~ uses its standardisation (mu, sigma), so that W~c = W_c
                    reproduces g(x) exactly (W~c has no bias, so it could not absorb a different mu)."""
        self.cfg, self.device, self.verbose = cfg, device, verbose
        self.final = final.to(device)
        self.use = dict(l2=use_l2, l3=use_l3, l4=use_l4)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        W0 = init_layer.W.clone()
        if cfg["fvlc_init"] == "gaussian":  # Algorithm 1, line 2 (scaled so activations stay O(1))
            W0 = torch.randn(W0.shape, generator=self.gen) / math.sqrt(W0.shape[1])
        self.W = torch.nn.Parameter(W0.to(device))
        # g~(x) = (W~ b_f(x) - mu) / sigma with the vanilla (mu, sigma), kept fixed
        self.mean, self.std = ref_layer.mean.to(device), ref_layer.std.to(device)

    # ---- model pieces -------------------------------------------------------------
    def g(self, feats):
        return (feats @ self.W.T - self.mean) / self.std

    def y(self, c):
        return self.final(c)

    def layer(self) -> ConceptLayer:
        return ConceptLayer(self.W.detach().cpu().clone(), self.mean.cpu(), self.std.cpu())

    # ---- inner maximisation (PGD, l_inf) ------------------------------------------
    def _uniform(self, shape, radius):
        return ((torch.rand(shape, generator=self.gen) * 2 - 1) * radius).to(self.device)

    def pgd_delta(self, feat_batches):
        """max_{||delta||_inf <= R2} KL(y(c~), y(c~ + delta)). W~ is fixed."""
        cfg = self.cfg
        R, step = cfg["pgd_radius"], cfg["pgd_step_size"]
        k = self.W.shape[0]
        universal = cfg["pgd_mode"] == "universal"
        delta = None
        for f in feat_batches:
            with torch.no_grad():
                c = self.g(f)
                y_c = self.y(c)
            if delta is None:
                delta = self._uniform((1, k) if universal else c.shape, R)
            delta.requires_grad_(True)
            loss = kl(y_c, self.y(c + delta))
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = (delta + step * grad.sign()).clamp(-R, R)
        return delta.detach()

    def _perturbed_concepts(self, feats, rho):
        if self.cfg["rho_space"] == "feature":
            return self.g(feats + rho * self.feat_scale)
        return self.g(feats) + rho

    def pgd_rho(self, feat_batches):
        """max_{||rho||_inf <= R1} L_k2(g~(x), g~(x)+rho). W~ is fixed."""
        cfg = self.cfg
        R, step, k2 = cfg["x_pgd_radius"], cfg["x_pgd_step_size"], cfg["k2"]
        universal = cfg["pgd_mode"] == "universal"
        dim = self.W.shape[1] if cfg["rho_space"] == "feature" else self.W.shape[0]
        rho = None
        for f in feat_batches:
            with torch.no_grad():
                c = self.g(f)
            if rho is None:
                rho = self._uniform((1, dim) if universal else (f.shape[0], dim), R)
            rho.requires_grad_(True)
            loss = topk_surrogate(c, self._perturbed_concepts(f, rho), k2)
            grad, = torch.autograd.grad(loss, rho)
            with torch.no_grad():
                rho = (rho + step * grad.sign()).clamp(-R, R)
        return rho.detach()

    # ---- outer minimisation ---------------------------------------------------------
    def fit(self, feats, c_ref, y_ref):
        """feats : (N, d) backbone features the FVLC layer sees (perturbed pipeline, e.g. noisy images for IP)
        c_ref : (N, k) vanilla concept vectors g(x)   (targets of L2)
        y_ref : (N, C) vanilla logits y(x, c)          (targets of L1)"""
        cfg, dev = self.cfg, self.device
        feats, c_ref, y_ref = feats.to(dev), c_ref.to(dev), y_ref.to(dev)
        self.feat_scale = feats.std(dim=0, keepdim=True)  # rho is expressed in per-dimension std units
        N, bs = len(feats), cfg["fvlc_batch_size"]
        steps_per_epoch = math.ceil(N / bs)
        total = cfg["fvlc_epochs"] * steps_per_epoch
        opt = torch.optim.Adam([self.W], lr=cfg["fvlc_lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        l = [cfg["lambda1"], cfg["lambda2"], cfg["lambda3"], cfg["lambda4"]]
        universal = cfg["pgd_mode"] == "universal"

        def rand_batches(n):
            return [feats[torch.randint(0, N, (bs,), generator=self.gen).to(dev)] for _ in range(n)]

        history = []
        for ep in range(cfg["fvlc_epochs"]):
            perm = torch.randperm(N, generator=self.gen).to(dev)
            sums = torch.zeros(5)
            for s in range(steps_per_epoch):
                idx = perm[s * bs:(s + 1) * bs]
                f = feats[idx]
                # Algorithm 1 lines 4-12: inner PGD for delta and rho
                delta = rho = None
                if self.use["l3"]:
                    delta = self.pgd_delta(rand_batches(cfg["pgd_steps"]) if universal else [f] * cfg["pgd_steps"])
                if self.use["l4"]:
                    rho = self.pgd_rho(rand_batches(cfg["x_pgd_steps"]) if universal else [f] * cfg["x_pgd_steps"])
                # line 14: SGD step on W~
                c_t = self.g(f)
                y_t = self.y(c_t)
                L1 = kl(y_ref[idx], y_t)
                zero = torch.zeros((), device=dev)
                L2 = topk_surrogate(c_ref[idx], c_t, cfg["k1"]) if self.use["l2"] else zero
                L3 = kl(y_t, self.y(c_t + delta)) if self.use["l3"] else zero
                L4 = topk_surrogate(c_t, self._perturbed_concepts(f, rho), cfg["k2"]) if self.use["l4"] else zero
                loss = l[0] * L1 + l[1] * L2 + l[2] * L3 + l[3] * L4
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                sums += torch.tensor([loss.item(), L1.item(), L2.item(), L3.item(), L4.item()])
            avg = (sums / steps_per_epoch).tolist()
            history.append(dict(zip(["loss", "L1", "L2", "L3", "L4"], avg)))
            if self.verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == cfg["fvlc_epochs"] - 1):
                print(f"    FVLC epoch {ep + 1:3d}/{cfg['fvlc_epochs']}  loss {avg[0]:.3e}  L1 {avg[1]:.3e}  "
                      f"L2 {avg[2]:.3e}  L3 {avg[3]:.3e}  L4 {avg[4]:.3e}")
        return self.layer(), history


def attack_features(layer: ConceptLayer, final: FinalLayer, feats, scale, radius, steps, step_size, seed=0,
                    device="cpu", batch_size=1024):
    """Per-sample l_inf PGD on backbone features: x' = x + rho * scale, |rho|_inf <= radius, maximising
    KL(y(x) || y(x')). Returns clean and attacked (concepts, logits) for every sample (on CPU)."""
    layer, final = layer.to(device), final.to(device)
    scale = scale.to(device)
    g = torch.Generator().manual_seed(seed)
    out = [[], [], [], []]
    for i in range(0, len(feats), batch_size):
        f = feats[i:i + batch_size].to(device)
        with torch.no_grad():
            c0 = layer(f)
            y0 = final(c0)
        rho = ((torch.rand(f.shape, generator=g) * 2 - 1) * radius).to(device)
        for _ in range(steps):
            rho.requires_grad_(True)
            loss = kl(y0, final(layer(f + rho * scale)))
            grad, = torch.autograd.grad(loss, rho)
            with torch.no_grad():
                rho = (rho + step_size * grad.sign()).clamp(-radius, radius)
        with torch.no_grad():
            c1 = layer(f + rho * scale)
            y1 = final(c1)
        for lst, t in zip(out, (c0, y0, c1, y1)):
            lst.append(t.cpu())
    return tuple(torch.cat(x) for x in out)
