# Perturbation Centroid Predicts JPEG Recompression Schedule for Adversarial Defense

## Abstract
Joint Photographic Experts Group (JPEG) recompression can suppress adversarial perturbations, but the best ordering of quality factors across repeated recompression changes with the attack and classifier. This paper studies schedule choice through the perturbation frequency centroid [math:\omega_\delta], measured in the block discrete cosine transform (DCT) domain, and an elimination threshold [math:\tau] derived from quantization and perturbation amplitude. Across eight nonadaptive CIFAR-10 model-attack conditions, the [math:\omega_\delta-\tau] sign rule gives the lower-ASR front-loaded-versus-fixed winner in 5/8 conditions. In the other 3/8 conditions, front-loaded and fixed schedules differ by [math:|\Delta|<1] percentage point, making the two schedules practically equivalent. Standard ResNet-18 gives the strongest schedule effect, with 12 to 19 percentage-point differences, while robust training shifts [math:\omega_\delta] toward lower DCT frequencies and explains the transition toward fixed schedules.

## Keywords
adversarial examples, attack success rate, discrete cosine transform, JPEG recompression, schedule selection

## Introduction

Joint Photographic Experts Group (JPEG) recompression quantizes block discrete cosine transform (DCT) coefficients and is inexpensive enough to use as a preprocessing defense against adversarial examples [cite:jpeg], [cite:shield]. Input-transformation defenses motivate asking how compression should be parameterized across repeated preprocessing steps [cite:xu]. Here that design variable is the ordered list of quality factors (QFs), called the schedule.

Adversarial examples from the Fast Gradient Sign Method (FGSM) [cite:goodfellow] and Projected Gradient Descent (PGD) [cite:madry] motivate compression defenses, but such defenses require adaptive evaluation because a preprocessing stage can look effective when gradients through it are hidden from the attacker [cite:athalye].

We study schedule selection as a frequency problem. JPEG quantization steps increase with DCT frequency, so the effect of a QF schedule depends on where perturbation energy sits in the DCT spectrum. We define the perturbation frequency centroid [math:\omega_\delta] and an elimination threshold [math:\tau], then test whether their sign comparison predicts the better front-loaded or fixed schedule.

## Methods

The CIFAR-10 sweep uses a standard ResNet-18 and three RobustBench-named robust checkpoints: Engstrom2019, Wong2020Fast, and Rice2020Overfitting [cite:cifar], [cite:robustbench], [cite:wong], [cite:rice]. The four models span clean-trained and adversarially trained classifiers with varying robustness levels.

{{FREQUENCY_MODEL}}

{{MICROSTRUCTURE_MODEL}}

For intuition, a high-frequency coefficient at QF 50 can have first-stage quantization step [math:\Delta_1=121]. A typical PGD residual at this position has [math:|\delta_0|\approx3], giving [math:\mu_1=\lceil3/121\rceil=0] under the residual-elimination convention used in the audit. A low-frequency coefficient with [math:\Delta_1=7] gives [math:\mu_1=\lceil3/7\rceil=1], so the residual survives.

The adaptivity variable [math:\alpha] equals 0 when the attack is generated against the classifier alone and equals 1 when the attacker differentiates through a JPEG surrogate. The prediction rules are summarized in [tab:prediction-rules].

{{PREDICTION_RULE_TABLE}}

Experiments used QF ranges R1 from 75 to 50, R2 from 85 to 55, and R3 from 90 to 60. Schedule families were geometric, arithmetic, fixed, and front-loaded reverse geometric. FGSM and PGD-20 used [math:\epsilon=8/255] without differentiating through the defense. JPEG-aware PGD used a vendored differentiable JPEG surrogate during attack generation [cite:shin], [cite:reich]. Defended evaluation used real Pillow JPEG recompression. ASR was computed on clean-correct images.

{{SCHEDULE_DEFINITION}}

{{ASR_DEFINITION}}

The paired source-data file records per-image success flags. McNemar tests compare front-loaded and fixed schedules on the same clean-correct images at the largest generation count. We report [math:\Delta=ASR_FL-ASR_Fix] in percentage points. Negative [math:\Delta] favours front-loaded schedules, while positive [math:\Delta] favours fixed schedules.

## Results

ASR trajectories across recompression generations are shown in [fig:asr-generation].

{{MECHANISM_FIGURE}}

The measured perturbation centroid, threshold comparison, signed margin, and predicted schedule for every CIFAR-10 condition are reported in [tab:frequency-diagnostics]. The threshold audit in [tab:tau-audit] checks that [math:\tau] is stable across the calibration variants used by the frequency diagnostic.

{{FREQUENCY_DIAGNOSTIC_TABLE}}

{{TAU_INDEPENDENCE_TABLE}}

The signed margin [math:\omega_\delta-\tau] is useful as an audit coordinate but does not by itself define an effect-size class. The scatter plot compares this coordinate with the paired ASR difference [math:\Delta].

{{CENTROID_DELTA_FIGURE}}

The four-family ASR table keeps arithmetic and geometric schedules as negative controls for average QF. If order did not matter, the first-stage QF would not separate schedule families.

{{FULL_FOUR_SCHEDULE_ASR_TABLE}}

For the eight nonadaptive CIFAR-10 conditions, the [math:\omega_\delta-\tau] sign rule predicts the lower-ASR front-loaded-versus-fixed winner in 5/8 conditions. In the other 3/8 conditions, front-loaded and fixed schedules differ by [math:|\Delta|<1] percentage point, so the choice has negligible practical effect.

The paired McNemar test for front-loaded versus fixed schedules at the largest generation count is reported in [tab:mcnemar-audit]. The table is generated directly from per-image source data, so the sample size and [math:p]-value reflect the CSV supplied to the renderer.

{{MCNEMAR_AUDIT_TABLE}}

Robust training shifts [math:\omega_\delta] toward lower frequencies, producing a smooth transition across model families and explaining why Rice2020 selects fixed schedules.

{{ROBUST_TRAINING_GRADIENT_TABLE}}

JPEG-aware rows are boundary checks rather than the main validation set. The adaptive surrogate changes the schedule preference and is audited separately from the nonadaptive threshold rule.

{{JPEG_AWARE_BOUNDARY_TABLE}}

The FL-vs-Fix winner counts at the largest generation count are summarized in [fig:schedule-boundary] and [tab:model-boundary]. This view separates condition-level direction from range-level consistency.

{{SCHEDULE_BOUNDARY_FIGURE}}

{{MODEL_BOUNDARY_TABLE}}

## Discussion

This study shows that the perturbation frequency centroid [math:\omega_\delta] provides a physical audit variable for JPEG recompression schedules under nonadaptive CIFAR-10 attacks. The elimination threshold [math:\tau] supplies a decision boundary derived from the quantization table and perturbation amplitude. The sign rule is useful, but it is not a perfect classifier.

The mechanism is direct. When [math:\omega_\delta>\tau], most perturbation energy sits at DCT positions with large quantization steps. A front-loaded first step at the lowest QF maximizes [math:\Delta_1] and drives small residuals below the surviving quantization bin at those positions. The recursion [math:\mu_2 \le r_1\mu_1+1] then contracts the residual because [math:r_1<1]. When [math:\omega_\delta\le\tau], most energy sits at positions with small quantization steps where residuals remain nonzero even under aggressive quantization. A fixed schedule avoids concentrating damage in any single step.

The direct audit is deliberately narrow. The sign rule gives the lower-ASR FL-vs-Fix winner in 5/8 nonadaptive conditions. The three reversals have [math:|\Delta|<1] percentage point, so they do not support a strong practical preference for either schedule. The signed margin [math:|\omega_\delta-\tau|] should not be read as a monotonic predictor of [math:|\Delta|].

Robust training changes schedule preference by shifting [math:\omega_\delta] toward lower frequencies, consistent with evidence that adversarially trained classifiers depend less on high-frequency components [cite:wanghfc]. Rice2020 drops below [math:\tau], and fixed schedules take over. Engstrom19 and Wong20 sit close enough to the decision boundary that small residual-placement changes can flip the FL-vs-Fix winner.

The practical workflow is simple. A defender measures [math:\omega_\delta] on a calibration set of adversarial examples and computes [math:\tau] from the midpoint-QF quantization table and the median perturbation amplitude. The sign rule selects a front-loaded or fixed schedule as an initial hypothesis. Paired source-data auditing remains necessary when the observed ASR difference is small. The measurement requires one forward pass and one block DCT per image.

Arithmetic and geometric schedules serve as negative controls for average QF. The JPEG-aware rows are boundary checks because attack generation uses a differentiable surrogate while defended evaluation uses real JPEG recompression. The ImageNet transformer observation is similarly a limitation rather than a second main result table [cite:imagenet], [cite:vit], [cite:swin], [cite:deit].

Three limitations should be stated. First, the rule has only been validated here as a CIFAR-10 main result. Second, the JPEG-aware rule remains approximate because the surrogate attack and real JPEG evaluation do not share an exact gradient. Third, [math:\tau] is estimated from data. A more complete treatment would derive [math:\tau] from the quantization table and the perturbation budget [math:\epsilon] without calibration.

JPEG schedule reports should include schedule family, QF range, generation count, attack objective, model family, [math:\omega_\delta], [math:\tau], sample size, ASR confidence limits, and paired [math:p]-values. With those elements, recompression can be audited as a frequency-targeted intervention whose effect is predictable from the perturbation spectrum.

## Data Availability

All generated metrics, per-sample source data, frequency diagnostics, figures, LaTeX source, and PDF outputs are written under the results directory during local execution. CIFAR-10 and model checkpoints are mounted input resources under the data directory. The experiment runner does not download datasets or weights at runtime. The repository code, resource documentation, and vendored differentiable JPEG source are included with this source release.
