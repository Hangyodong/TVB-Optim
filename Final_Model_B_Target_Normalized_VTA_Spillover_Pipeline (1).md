# Final Model B Pipeline with Target-Normalized VTA Spillover Weighting

> Updated: 2026-08-24  
> Status: **Current primary DBS stimulation pipeline**

---

## 1. 핵심 개념

본 모델은 DBS 자극 시 선택된 **primary target ROI는 항상 stimulation weight = 1**로 두고,
VTA가 target ROI를 넘어 주변 ROI까지 확장될 경우 주변 ROI에는 **target-relative spillover weight (0.xx)** 를 부여한다.

즉 공간적 자극은 다음과 같이 표현한다.

```text
Primary target ROI
    → weight = 1.0

Extra ROI overlapped by VTA
    → weight = 0.xx

No VTA overlap
    → weight = 0
```

이 weight는 실제 neuron activation fraction이 아니라:

> **target-normalized VTA spillover weight**

로 해석한다.

---

# 2. 전체 Model B causal pipeline

```text
DBS condition
(target, amplitude, PW, frequency, contact, waveform)
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Spatial stimulation                Temporal stimulation
A + PW + target + contact          frequency + pulse timing
        │                              │
        ▼                              ▼
Canonical DBS electrode            DBS pulse-event sequence
        │                              │
        ▼                              ▼
VTA calculation                    TM_E + TM_I
        │                              │
        ▼                              ▼
Target-normalized                  q_E(t), q_I(t)
spillover weights w_i                  │
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Effective DBS event rate
             H_DBS,E,i / H_DBS,I,i
                       │
                       ▼
                 E/I RWW gating
                       │
                       ▼
         Subject-specific SC + tract length
                       │
                       ▼
                   TVB simulation
                       │
                       ▼
                  BOLD / FC
                       │
                       ▼
       DBS response validation / optimization
```

---

# 3. Optimizer input

최적화 또는 실제 DBS setting은 다음으로 정의한다.

\[
\boxed{
\theta=
[target,A,PW,f,contact,waveform]
}
\]

- `target`: primary stimulation ROI
- `A`: DBS stimulation amplitude
- `PW`: pulse width
- `f`: frequency
- `contact`: active contact/configuration
- `waveform`: pulse waveform

---

# 4. Spatial/temporal 역할 분리

## Spatial branch

\[
\boxed{
(A,PW,target,contact)
\rightarrow
VTA
\rightarrow
w_i
}
\]

Amplitude와 pulse width는 spatial recruitment를 결정한다.

---

## Temporal branch

\[
\boxed{
f
\rightarrow
pulse\ events
\rightarrow
TM_E/TM_I
\rightarrow
q_E(t),q_I(t)
}
\]

Frequency는 TM short-term synaptic dynamics를 결정한다.

---

# 5. Target ROI centroid

Target ROI \(t\)의 atlas centroid:

\[
\mathbf c_t^{voxel}
=
\frac{1}{N_t}
\sum_{v\in ROI_t}\mathbf v
\]

NIfTI affine을 사용하여 world/mm coordinate로 변환한다.

Canonical Medtronic DBS electrode의 active contact center를 해당 centroid에 배치한다.

이 위치는:

> patient-specific postoperative lead location이 아니라 macro-scale canonical stimulation location이다.

---

# 6. Canonical VTA model

Baseline에서는 대표 Medtronic conventional DBS lead geometry를 사용한다.

고정 assumptions:

```text
Electrode:
    Medtronic conventional 4-contact DBS lead
    baseline geometry: 3389

Tissue:
    literature-fixed GM conductivity
    literature-fixed WM conductivity
    isotropic baseline

Activation:
    literature-fixed representative axon diameter
    pulse-width-dependent E-field activation threshold
```

VTA:

\[
VTA
=
F(
A,
PW,
contact,
electrode\ geometry,
position,
\sigma,
d_{axon},
activation\ threshold
)
\]

---

# 7. DBS amplitude와 VTA

Amplitude \(A\)는 RWW current에 직접 넣지 않는다.

Amplitude는 VTA field extent를 통해 반영한다.

\[
\boxed{
A \uparrow
\Rightarrow
|E(\mathbf r)| \uparrow
\Rightarrow
VTA\ volume \uparrow
}
\]

Activation criterion:

\[
\boxed{
VTA
=
\{
\mathbf r:
|\mathbf E(\mathbf r;A)|
\ge
E_{th}(PW)
\}
}
\]

따라서:

```text
1.0 mA → VTA_1
1.5 mA → VTA_2
2.0 mA → VTA_3
2.5 mA → VTA_4
3.0 mA → VTA_5
...
```

---

# 8. 기존 alpha 방식은 primary stimulation weight로 사용하지 않음

기존:

\[
\alpha_i
=
\frac{|VTA\cap ROI_i|}
{|ROI_i|}
\]

는 ROI 전체 크기에 영향을 많이 받는다.

예를 들어 큰 cortical ROI는 동일한 VTA overlap volume을 가져도 작은 subcortical ROI보다
\(\alpha_i\)가 작아질 수 있다.

따라서 최종 Model B에서는 \(\alpha_i\)를 primary stimulation weight로 직접 쓰지 않고,
**target-normalized VTA spillover weighting**을 사용한다.

---

# 9. Target-normalized VTA spillover weight

Primary target ROI를 \(t\)라고 한다.

Target weight:

\[
\boxed{
w_t=1
}
\]

주변 ROI \(i\neq t\)에 대해:

\[
\boxed{
w_i
=
\frac{|VTA\cap ROI_i|}
{|VTA\cap ROI_t|}
}
\]

실제 구현에서는 1을 초과하지 않도록:

\[
\boxed{
w_i
=
\min
\left(
1,
\frac{|VTA\cap ROI_i|}
{|VTA\cap ROI_t|}
\right)
}
\]

를 사용한다.

VTA와 겹치지 않으면:

\[
\boxed{
w_i=0
}
\]

따라서 최종 spatial rule:

\[
\boxed{
w_i=
\begin{cases}
1,&i=t\\[4pt]
\min\left(
1,\frac{|VTA\cap ROI_i|}
{|VTA\cap ROI_t|}
\right),
&i\neq t,\ |VTA\cap ROI_i|>0\\[8pt]
0,&\text{otherwise}
\end{cases}
}
\]

---

# 10. Spatial weighting example

예를 들어 STN을 primary target으로 설정하고 VTA overlap이:

```text
STN   = 80 mm³
ZI    = 20 mm³
GPi   = 4 mm³
GPe   = 0 mm³
```

라면:

\[
w_{STN}=1
\]

\[
w_{ZI}
=
\frac{20}{80}
=
0.25
\]

\[
w_{GPi}
=
\frac{4}{80}
=
0.05
\]

\[
w_{GPe}=0
\]

따라서 stimulation spatial vector:

```text
STN = 1.00
ZI  = 0.25
GPi = 0.05
GPe = 0.00
```

---

# 11. 해석

이 값들은:

```text
STN neuron 100% activated
ZI neuron 25% activated
```

을 의미하지 않는다.

정확한 해석:

> target ROI의 direct stimulation을 기준값 1로 정규화하고,
> 주변 ROI의 VTA overlap 정도를 상대적 spillover stimulation weight로 표현한다.

따라서 \(w_i\)는:

\[
\boxed{
\text{target-normalized spatial stimulation weight}
}
\]

이다.

---

# 12. Amplitude 증가 시 behavior

예를 들어 같은 STN target에서:

## 1.5 mA

```text
STN overlap = 60 mm³
ZI overlap  = 3 mm³
```

\[
w_{STN}=1
\]

\[
w_{ZI}=0.05
\]

---

## 3.0 mA

```text
STN overlap = 90 mm³
ZI overlap  = 27 mm³
GPi overlap = 9 mm³
```

\[
w_{STN}=1
\]

\[
w_{ZI}=0.30
\]

\[
w_{GPi}=0.10
\]

즉 amplitude가 증가하면 primary target의 weight 자체를 1보다 크게 만들지 않고:

\[
\boxed{
A\uparrow
\Rightarrow
VTA\uparrow
\Rightarrow
surrounding\ ROI\ spillover\uparrow
}
\]

로 표현한다.

이 모델은 binary-threshold VTA assumption과 일관된다.

---

# 13. 중요한 amplitude 해석

현재 baseline Model B에서는:

```text
1.5 mA STN stimulation
3.0 mA STN stimulation
```

모두 target STN의 spatial weight는:

\[
w_{STN}=1
\]

이다.

Amplitude 증가 효과는:

- VTA가 커짐
- 더 많은 neighboring ROI가 VTA에 포함됨
- surrounding spillover weight가 증가함

으로 표현된다.

즉 target 자체 내부에서:

\[
3.0mA > 1.5mA
\]

에 해당하는 graded intensity effect를 별도로 추가하지 않는다.

이는 threshold-based binary VTA model의 simplifying assumption이다.

---

# 14. Optional future continuous E-field weighting

더 높은 fidelity가 필요하면 binary overlap 대신 voxel-wise continuous E-field를 사용할 수 있다.

ROI \(i\)의 field-weighted activation:

\[
a_i
=
\frac{1}{|ROI_i|}
\int_{ROI_i}
g(|E(\mathbf r)|)\,d\mathbf r
\]

Target-normalized weight:

\[
\boxed{
w_t=1
}
\]

\[
\boxed{
w_i
=
\min
\left(
1,
\frac{a_i}{a_t}
\right)
}
\]

이 방법은 동일 VTA 내부에서도 electrode에 가까운 voxel과 경계 voxel의 activation strength 차이를 표현할 수 있다.

Baseline에서는 binary VTA overlap weighting을 사용한다.

---

# 15. DBS pulse-event sequence

Frequency:

\[
f
\]

에서 pulse interval:

\[
\boxed{
T_{pulse}
=
\frac{1}{f}
}
\]

예:

\[
f=130Hz
\]

이면:

\[
T_{pulse}\approx7.69ms
\]

이미 구현된 pulse generator에서 pulse timing을 TM model로 전달한다.

Amplitude는 이 TM branch에 다시 곱하지 않는다.

---

# 16. Dynamic E/I Tsodyks-Markram

모든 ROI에서 동일한 representative E/I TM parameter set을 사용한다.

## Excitatory TM

\[
TM_E=
(U_E,\tau_{rec,E},\tau_{fac,E})
\]

\[
\frac{dx_E}{dt}
=
\frac{1-x_E}{\tau_{rec,E}}
-
u_E x_E P(t)
\]

\[
\frac{du_E}{dt}
=
\frac{U_E-u_E}{\tau_{fac,E}}
+
U_E(1-u_E)P(t)
\]

\[
\boxed{
q_E(t)=u_E(t)x_E(t)
}
\]

---

## Inhibitory TM

\[
TM_I=
(U_I,\tau_{rec,I},\tau_{fac,I})
\]

\[
\frac{dx_I}{dt}
=
\frac{1-x_I}{\tau_{rec,I}}
-
u_I x_I P(t)
\]

\[
\frac{du_I}{dt}
=
\frac{U_I-u_I}{\tau_{fac,I}}
+
U_I(1-u_I)P(t)
\]

\[
\boxed{
q_I(t)=u_I(t)x_I(t)
}
\]

---

# 17. Representative TM parameters

모든 ROI에서 공통:

\[
\boxed{
U_E,\;
\tau_{rec,E},\;
\tau_{fac,E},
U_I,\;
\tau_{rec,I},\;
\tau_{fac,I}
}
\]

따라서:

- ROI-specific TM lookup 없음
- target-specific q lookup 없음
- rho 없음
- fitted kappa 없음

---

# 18. Spatial weight + TM 결합

최종 effective DBS event rate:

## Excitatory

\[
\boxed{
H^{DBS}_{E,i}(t)
=
w_i f q_E(t)
}
\]

## Inhibitory

\[
\boxed{
H^{DBS}_{I,i}(t)
=
w_i f q_I(t)
}
\]

단위:

\[
w_i: dimensionless
\]

\[
q_E,q_I: dimensionless
\]

\[
f:s^{-1}
\]

따라서:

\[
H^{DBS}:s^{-1}=Hz
\]

---

# 19. Existing E/I RWW gating에 추가

기존 E/I RWW:

\[
\frac{dS_{E,i}}{dt}
=
-\frac{S_{E,i}}{\tau_E}
+
(1-S_{E,i})\gamma_E H_{E,i}
\]

\[
\frac{dS_{I,i}}{dt}
=
-\frac{S_{I,i}}{\tau_I}
+
\gamma_I H_{I,i}
\]

Effective rate:

\[
\boxed{
H_{E,i}^{eff}
=
H_{E,i}
+
w_i f q_E(t)
}
\]

\[
\boxed{
H_{I,i}^{eff}
=
H_{I,i}
+
w_i f q_I(t)
}
\]

최종:

\[
\boxed{
\frac{dS_{E,i}}{dt}
=
-\frac{S_{E,i}}{\tau_E}
+
(1-S_{E,i})
\gamma_E
[
H_{E,i}
+
w_i f q_E(t)
]
}
\]

\[
\boxed{
\frac{dS_{I,i}}{dt}
=
-\frac{S_{I,i}}{\tau_I}
+
\gamma_I
[
H_{I,i}
+
w_i f q_I(t)
]
}
\]

---

# 20. Parameters intentionally not used

최종 Model B baseline에서는 다음 추가 parameter를 사용하지 않는다.

```text
rho          : not used
q lookup     : not used
kappa        : not used
J_DBS        : not used
tau_DBS      : not used
TRK/TCK      : not required in core runtime
ROI-specific TM parameter table : not used
```

---

# 21. Whole-brain propagation

Direct stimulation은 \(w_i>0\)인 ROI에만 직접 적용된다.

그 이후 network propagation은 subject-specific structural connectome에서 발생한다.

SC:

\[
C_{ij}
\]

tract length:

\[
L_{ij}
\]

delay:

\[
\tau_{ij}
=
\frac{L_{ij}}{v}
\]

network term:

\[
I_i^{network}
\propto
G
\sum_j
C_{ij}
S_j(t-\tau_{ij})
\]

역할 분리:

```text
VTA weight w_i → direct spatial stimulation
TM             → frequency-dependent synaptic efficacy
SC             → indirect whole-brain propagation
```

SC를 external DBS stimulation vector에 미리 곱하지 않는다.

---

# 22. BOLD / FC

```text
E/I RWW
↓
neural activity
↓
hemodynamic model
↓
BOLD
↓
FC
```

Simulation:

\[
\Delta FC_{sim}
=
FC_{ON}^{sim}
-
FC_{OFF}^{sim}
\]

Empirical:

\[
\Delta FC_{emp}
=
FC_{ON}^{emp}
-
FC_{OFF}^{emp}
\]

Validation:

\[
\boxed{
\Delta FC_{sim}
\leftrightarrow
\Delta FC_{emp}
}
\]

---

# 23. Optimal DBS search

Optimizer:

\[
\boxed{
\theta=
[target,A,PW,f,contact,waveform]
}
\]

각 iteration:

```text
θ
↓
target centroid
↓
canonical Medtronic electrode
↓
VTA(A, PW, target, contact)
↓
target-normalized spillover weights w_i
↓
pulse events(f)
↓
TM_E / TM_I
↓
q_E(t), q_I(t)
↓
H_DBS,E/I = w_i × f × q_E/I
↓
E/I RWW
↓
TVB(SC, tract length)
↓
BOLD / FC
↓
normalization objective
↓
next stimulation condition
```

---

# 24. Computational caching

Spatial branch can be precomputed:

\[
(target,A,PW,contact)
\rightarrow
VTA
\rightarrow
\mathbf w
\]

Example cache:

```text
STN_R_1.0mA_60us.npy
STN_R_1.5mA_60us.npy
STN_R_2.0mA_60us.npy
GPi_R_2.0mA_60us.npy
...
```

Runtime:

```text
load spatial weight vector w
↓
run TM for frequency
↓
E/I RWW
```

---

# 25. Final parameter roles

## Literature/canonical fixed

```text
Medtronic electrode geometry
GM/WM conductivity
axon diameter assumption
PW-dependent activation threshold

U_E
tau_rec_E
tau_fac_E

U_I
tau_rec_I
tau_fac_I
```

## Already fixed in implementation

```text
DBS pulse implementation
TM integration timestep
RWW integration timestep
```

## Subject-specific twin

```text
E/I RWW parameters
SC weight matrix
tract-length matrix
global coupling / fitted twin parameters
```

## Optimizer variables

```text
target
amplitude
pulse width
frequency
contact
waveform
```

---

# 26. Final core equations

Spatial VTA:

\[
VTA
=
F(A,PW,target,contact)
\]

Target-normalized spillover:

\[
\boxed{
w_i=
\begin{cases}
1,&i=t\\[4pt]
\min\left(
1,
\frac{|VTA\cap ROI_i|}
{|VTA\cap ROI_t|}
\right),
&i\neq t,\ |VTA\cap ROI_i|>0\\[8pt]
0,&otherwise
\end{cases}
}
\]

TM:

\[
\boxed{
q_E(t)=u_E(t)x_E(t)
}
\]

\[
\boxed{
q_I(t)=u_I(t)x_I(t)
}
\]

Effective DBS event rate:

\[
\boxed{
H^{DBS}_{E,i}
=
w_i f q_E(t)
}
\]

\[
\boxed{
H^{DBS}_{I,i}
=
w_i f q_I(t)
}
\]

RWW:

\[
\boxed{
\frac{dS_{E,i}}{dt}
=
-\frac{S_{E,i}}{\tau_E}
+
(1-S_{E,i})
\gamma_E
[
H_{E,i}
+
w_i f q_E(t)
]
}
\]

\[
\boxed{
\frac{dS_{I,i}}{dt}
=
-\frac{S_{I,i}}{\tau_I}
+
\gamma_I
[
H_{I,i}
+
w_i f q_I(t)
]
}
\]

---

# 27. Final summary

최종 Model B:

\[
\boxed{
DBS(A,PW,f,target)
\rightarrow
VTA
\rightarrow
w_i
\rightarrow
TM_E/TM_I
\rightarrow
H^{DBS}_{E/I}
\rightarrow
E/I\ RWW
\rightarrow
SC
\rightarrow
FC
}
\]

핵심 변경:

```text
Primary target ROI:
    w = 1.0

VTA spillover ROI:
    w = 0.xx

Outside VTA:
    w = 0
```

이 방식은 primary target stimulation을 기준값으로 유지하면서
DBS amplitude/PW 증가에 따른 VTA 확장을 주변 ROI spillover로 표현한다.

Spatial stimulation과 frequency-dependent synaptic dynamics를 분리하면서
추가 empirical scaling parameter 없이 RWW neural-mass dynamics에 연결하는 것이
본 Model B의 핵심이다.
