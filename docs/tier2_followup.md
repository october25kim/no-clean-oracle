# Tier 2 follow-up — OOD-far pin, sealing pins, additive reporting rules

Timestamped follow-up commit to `tier2_amendment.md` (2026-08-13), made **before launch**
under that amendment's own terms (§1: "a change requires a timestamped follow-up commit
before launch"). It resolves the items the amendment left open and adds reporting rules.
It changes **no** registered pin: the grid, learners, optimizer, axes, oracles and selectors
of the amendment stand as written.

## 1. OOD-far pool — PINNED: ISIC2019 (ruling 44-L1)

The amendment left OOD-far as "[PIN at sealing: one disjoint-domain natural-image pool,
resized to 224; candidate list fixed from what is licensable and locally available]".

**Pinned: ISIC2019 dermoscopic images**, 50,662 images, natively 224x224 / 298x224 so the
resize to 224 introduces essentially no resampling artifact.

**Registered caveat, required in any report using this pool:** dermoscopy carries distinctive
low-level image statistics that plausibly make OOD detection *easier* than a consumer-photo
far-pool would. That direction is conservative for an existence check — an easier OOD axis
under-counts incompatibility, so a positive finding is not an artifact of a too-easy pool.
It does mean a negative finding on this axis is weak evidence.

Rejected for this role: SVHN and CIFAR. At 32x32 they must be upsampled 7x to reach 224, so
what the axis would measure is a resolution artifact rather than a domain shift.

## 2. OOD-far sensitivity pool — OfficeHome "Real World", with exclusions

Approved conditionally (44-L1) subject to removing apparel overlap. Clothing1M's 14 classes
are garments, so overlap was judged on the apparel/accessory **domain**, not on exact label
match. **Excluded, 5 of 65 categories:**

| class | images | reason |
|---|---|---|
| `Backpack` | 99 | carried accessory |
| `Flipflops` | 85 | footwear |
| `Glasses` | 60 | worn accessory |
| `Helmet` | 60 | worn accessory |
| `Sneakers` | 88 | footwear |

Excluded 392 images (9.0%); **retained 60 classes, 3,965 images (91.0%)** — above the
viability floor, so the sensitivity pool is kept rather than dropped. It is reported as a
sensitivity only and never substitutes for the pinned pool. Its value is that it is
consumer-photo distributed like Clothing1M and therefore a harder, more informative far-pool
than dermoscopy; its cost is the smaller size and the variable native resolution
(220x143 to 2688x1520) requiring resize.

## 3. Sealing pins — computed once, frozen

**R_tail k=4 membership**, bottom-k by NOISY-label training frequency over the official
`noisy_label_kv.txt` x `noisy_train_key_list.txt` (1,000,000 keys, all labelled), ties by
ascending class id. Validation-free: no clean label enters the definition.

    classes [3, 4, 10, 12] ['Chiffon', 'Sweater', 'Shawl', 'Vest']

Full ascending frequency: Sweater 18,976 | Shawl 42,312 | Chiffon 50,092 | Vest 59,663 |
Downcoat 73,318 | Underwear 75,057 | Windbreaker 80,437 | Jacket 80,699 | Hoodie 82,829 |
Knitwear 85,788 | T-Shirt 86,152 | Dress 87,958 | Shirt 88,131 | Suit 88,588.

**C1M-C-local corruption pins**, following the CIFAR-C-local generation convention exactly
(`src/data/ood_pools.py`): corruptions `["gaussian_noise", "defocus_blur", "brightness",
"contrast"]`, **severity 3**, **2,000 images per corruption**, imagecorruptions v1.1.2,
applied to the clean Clothing1M test images at 224. As with CIFAR-C-local, this is a local
substitution and is never reported as an official corruption benchmark.

## 4. Additive reporting rules (ruling 42-L2b) — additive only, nothing withdrawn

**Selector degeneracy must be disclosed.** In the exploratory reconnaissance E(tau=1) and NA
selected the **same checkpoint in all four runs**, so their J and eta coincided throughout.
That was in a pretrained-initialization regime, which Tier 2 shares. Where two registered
selectors pick the same grid point, the report must say so explicitly rather than presenting
them as independent evidence: two selectors agreeing because they are the same number is not
corroboration. The disclosure is required whether or not the coincidence recurs.

**Tier 2 reports in its own table and is never pooled** — restated from amendment §4. The
20-point half-epoch grid over a 10-epoch pretrained trajectory is not comparable to Tier 1's
24 points over 120 epochs, and Tier 2 is an existence check at scale, not a statistical
pillar.

## 5. Wall-time recalibration (ruling 42-L2c)

The amendment's probe figure (0.5462 s/iter, 2.37 h/epoch) is a **compute-only lower bound**
from shape-accurate synthetic batches: no JPEG decode, no dataloading. The reconnaissance
found dataloading to be the binding constraint at 64px, and Tier 2 decodes 1,000,000 JPEGs
per epoch at 224. The first real-data epoch is measured and reported before any completion
estimate is treated as reliable.

## 6. Sealing regime (ruling 42-L3)

The pins in §3 are **design** pins: registered in the open, frozen from here, never revised.
The salted-commitment convention applies to Tier-2 **selector outputs** end to end —
commitment = sha256(salt || F) with the 32-byte salt withheld, which is a commitment and not
encryption. No sealed plaintext or key enters stdout, logs, git history, or any report.
