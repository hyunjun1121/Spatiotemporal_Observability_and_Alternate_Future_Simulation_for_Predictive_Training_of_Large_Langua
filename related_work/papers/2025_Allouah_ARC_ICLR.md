
---
**페이지 1**
---

Machine learning models for Si nanoparticle growth in nonthermal plasma Matt
Raymond 1, Paolo Elvati 2, Jacob C. Saldinger 3, Jonathan Lin 1, Xuetao Shi 2
and Angela Violi 1,2,3 1Mechanical Engineering, University of Michigan, Ann
Arbor, 48109-2125, Michigan, USA 2Electrical Engineering and Computer Science,
University of Michigan, Ann Arbor, 48109-2125, Michigan, USA 3Chemical
Engineering, University of Michigan, Ann Arbor, 48109-2125, Michigan, USA
January 3, 2025 Abstract Nanoparticles (NPs) formed in nonthermal plasmas (NTPs)
can have unique properties and applications. However, modeling their growth in
these environments presents significant challenges due to the non-equilibrium
nature of NTPs, making them computationally expensive to describe. In this work,
we address the challenges as- sociated with accelerating the estimation of
parameters needed for these models. Specifically, we explore how different
machine learning models can be tailored to im- prove prediction outcomes. We
apply these methods to reactive classical molecular dynamics data, which capture
the processes associated with colliding silane fragments in NTPs. These
reactions exemplify processes where qual- itative trends are clear, but their
quantification is chal- lenging, hard to generalize, and requires time-consuming
simulations. Our results demonstrate that good predic- tion performance can be
achieved when appropriate loss functions are implemented and correct invariances
are imposed. While the diversity of molecules used in the training set is
critical for accurate prediction, our find- ings indicate that only a fraction
(15-25%) of the energy and temperature sampling is required to achieve high lev-
els of accuracy. This suggests a substantial reduction in computational effort
is possible for similar systems. Keywords: Molecular Dynamics, sticking
coefficient, silane, machine learning, nanoparticle, nonthermal plasma Now at
Low Carbon Pathway Innovation at BP Now at the Dana-Farber Cancer Institute at
Harvard E-mail: avioli@umich.edu 1 Introduction Nonthermal plasmas (NTPs) are
unique environments where low-temperature neutral species and ions coexist with
high-temperature electrons. For this reason, these systems have received
considerable attention, especially for synthesizing particles and nanoparticles
with signifi- cant tunability. This flexibility in the final particle prop-
erties results from an environment with enough localized energy to cross
relatively high free energy barriers while avoiding excessive thermal energy and
discouraging ag- glomeration [1]. As a result, the synthesis of nanoparti- cles
and thin films under these conditions holds potential applications in
biomedicine [2, 3], energy [46], microelec- tronics [7], and catalysis [8].
However, modeling these environments remains a sig- nificant challenge due to
the combined non-equilibrium and multiscale nature. [911] Even when narrowing
the description only to a specific scale or class of processes, such as particle
growth (e.g., nucleation, coagulation, sur- face deposition) [1214], the
accuracy of the methods de- pends on their ability to model a variety of size-
and charge-dependent growth mechanisms. These processes, in turn, depend on the
propensity of species, generally radicals, to form stable bonds upon collision
with other particles or surfaces. Still, these processes have been frequently
estimated using fixed values, independent of the colliding species and energies
[14], primarily due to the complexity of obtaining a more detailed functional
form. Recently [15], we have shown how atomistic simu- lations can capture the
complex reactivity of small neu- trals and provide parameters that can be used
in reac- tor models [16, 17]. While these previous and current works focus on
silane particles, the underlying methodol- ogy is general and adaptable to
various conditions where species internal and translational energy distributions
differ. This flexibility sets our method apart from others, such as the one
recently published by Bal and Neyts [18], 1 arXiv:2501.00003v1 [physics.comp-ph]
31 Oct 2024

---
**페이지 2**
---

which (among other differences) does not make assump- tions about the
translational energy distribution or the specific reaction under investigation.
Due to the large size of some involved species and the resulting (lack of)
separation of vibrational modes, we observed a variety of competing reaction
mechanisms involving a complex in- terplay of physisorption, chemisorption, and
desorption. While very informative, deriving these reacting proba- bilities via
molecular dynamics (MD) simulations remains time-consuming and computationally
burdensome. Even when using classical reactive MD, a timestep of the or- der 10
as is required to guarantee correct numerical inte- gration of the equations of
motion during the reactions. Moreover, due to numerous variables (e.g., impact
param- eter, speed distribution, surface composition) and rele- vant species
present in such reactive systems, the number of simulations required to capture
the collisions experi- ences rapid combinatorial growth. More effective means of
deriving the collision parameters must be considered to scale this approach. In
this work, we focus on machine learning (ML) methods, which offer the potential
to for- mulate a dependency between the system conditions and the final
collision outcome. Data-driven methods do not remove the need for MD simulations
but allow for a dras- tic reduction in computational effort. Recent works have
used ML methods to overcome sim- ilar combinatorial problems associated with the
growth of nanoparticles in reactive gas-phase environments, ac- curately
predicting the aggregation propensities of soot precursors [19]. However, no
existing work addresses the same scientific questions in the context of
nonthermal plasmas. Most studies focus on predicting plasma proper- ties [2022]
and plasma-surface interactions, from surface deposition [11, 2326], plasma
etching [2729], and sur- face modification [30]. In contrast, this work examines
a scale between detailed individual reactions and simplified larger systems,
where detailed chemistry must be approx- imated. Our focus is on particles
approximately 1 nm in diameter colliding with small reactive fragments (SiHy and
Si2Hy). Building upon previous data [15], we demon- strate how easily computable
properties can be used to train ML models to generate predictions for new
species or mitigate the MD computational cost. 2 Methodology 2.1 Molecular
Dynamics Simulations We performed classical reactive molecular dynamics sim-
ulations to study the collisions between disilanes, Si2Hx, and other silane
clusters and molecules using the same procedure described previously [15].
First, we indepen- dently equilibrated both colliding species rotational and
vibrational modes, and then we performed microcanon- ical simulations at a fixed
impact velocity. Between 40 and 100 different collision vectors were imposed to
par- allel the line passing through the center of mass of the two species (i.e.,
impact parameter = 0 or, equivalently, impact angle = π). Simulations were
performed using LAMMPS [31] using the ReaxFF force field [32] in com- bination
with a dynamic charge equilibration model [33] and integrated the equations of
motion every 0.01 fs. To analyze the collision outcome, we monitored the mini-
mum distance between all the atoms or only the Si atoms of each cluster. The
conformations of the colliding species were gen- erated from the canonical
simulations at 300 K, 400 K, 500 K, 600 K, and 900 K. Properties are computed by
reweighting the collisions using a Maxwell-Boltzmann dis- tribution, which
allows labeling each system with a sin- gle temperature. For clarity, we grouped
the colliding species in two sets, labeled clusters and impactors, but there is
no physically meaningful distinction associ- ated with each set. Molecules in
the cluster set are silanes that cover different sizes and H-coverage (i.e.,
Si2H6, Si4, and Si29Hx with x = 18, 27, 31, 36), while as impactors, we
considered different disilanes (i.e., Si2Hx with x [1, 6]) in all possible
hydrogen distribution (e.g., for Si2H4, we simulated both H2Si SiH2 and HSi
SiH3). The re- sults of these simulations were combined with previously computed
data [15] to create a dataset of 390 collision pairs based on approximately 650
000 simulations. 2.2 Machine Learning We compared the predictive performances of
seven stan- dard ML models for predicting sticking probabilities: an unpenalized
linear model, ElasticNet, Kernel Ridge Re- gression (KRR), Support Vector
Regression (SVR), k- nearest neighbors (KNN) [34], DeepSets [35], and Light
Gradient-Boosting Machine (LGBM) [36]. 2.2.1 Input features For model input, we
generated feature vectors using pa- rameters that describe properties likely to
affect the stick- ing probability for silane molecules [15] (i.e., H coverage,
temperature, and molecules size). Specifically, for each cluster and impactor,
we used the number of Si atoms, H atoms, and a vector of the number of unpaired
electrons per Si atom (to differentiate between isomers). For the disilanes,
this two-dimensional vector indicates the num- ber of unpaired electrons on each
Si atom; in contrast, only the total number was used for the larger clusters,
and the second element was always set to 0. For particle a, we denote this
feature vector as f a R4. Each particle interaction has an associated
translational temperature, denoted t (0, ). Thus, we denote the feature vector
for a pair of particles as xa,b .= [f a f b t]R9. These nine features were
selected because they are computed efficiently and were expected to capture much
of the rele- vant chemistry. We normalized each training, validation, and
testing dataset so that the concatenation of the train- ing and validation sets
has a mean of 0 and a standard 2

---
**페이지 3**
---

deviation of 1 for each feature. 2.2.2 Loss functions Each plasma simulation can
be interpreted as a binomial distribution since each outcome is a Bernoulli
trial for some probability p. We use the negative log-likelihood of the binomial
distribution (B-NLL) as a loss function for a given simulation, ℓb(ˆp, m, n) .=
[m log ˆp + (n m) log(1 ˆp)] , (1) where ˆp is the predicted probability, m is
the number of events for the desired outcome (e.g., sticking), n is the total
number of events in a simulation. We implement DeepSets with a sigmoidal
activation σ(u) .= (1 + eu)1 on the output layer where σ: R [0, 1] and directly
optimize the B-NLL loss. Unfortu- nately, many ML libraries do not natively
support bi- nomial loss functions. In such cases, we can rewrite the binomial
loss ℓb in terms of the logistic loss ℓl, ℓb(ˆp, m, n) = [mℓl(ˆp, 1) + (n
m)ℓl(ˆp, 0)] for ℓl(ˆp, b) .= ( log ˆp b = 1 log(1 ˆp) b = 0 , (2) where b
indicates a class label. This can be interpreted as logistic regression that
includes both classes for each set of trials but weights each pseudosample
according to the number of positive and negative events. We use this approach
for Logistic ElasticNet and LGBM. Another perspective is to interpret the event
probabil- ity as a scalar p [0, 1] and perform regression in logit- space. We
cannot perform unconstrained regression di- rectly on probabilities, as the
model may predict unphys- ical values outside [0, 1]. Instead, we apply the
logistic unit (logit) σ1(p) .= log(1/(1p)) where σ1 : [0, 1] R to restrict the
(untransformed) output range. Note that {0, 1} values cannot be predicted with a
finite model out- put when using logits, so we clip the true probabilities at
[ϵ, 1 ϵ] for some small ϵ. We refer to this loss as the Logit MSE (L-MSE). To
further emulate the binomial NLL, we weigh the loss for each simulation
according to the number of trials and refer to it as the Logit-Weighted MSE
(LW-MSE). The LW-MSE penalizes outliers more significantly than Binomial NLL, as
seen in Figure 1. We use the LW-MSE for the unpenalized linear model, Elas-
ticNet, KRR, and SVR, and also evaluate it on DeepSets and LGBM. KNN does not
utilize a loss function and is automat- ically restricted to the range [0, 1] as
predictions are a weighted average of training data points, weighted by dis-
tance. We use a naıve predictor as a baseline, which predicts the mean of the
training probabilities. 2.2.3 Permutation invariance Because our two-particle
systems are permutation invari- ant, we train permutation invariant models using
either 0 1 0 1 Norm. Loss a) 0 1 Predicted Probability b) L-MSE Binomial NLL
Huber 0 1 c) Figure 1: B-NLL, L-MSE, and logit-transformed Huber (L-H) losses
rescaled and aligned for true Psts of a) 0.1, b) 0.5, c) 0.9. data manipulation
or model construction. Some ML model implementations cannot be cus- tomized to
be permutation invariant by construction, so we must adjust the dataset rather
than the model it- self. For linear models such as OLS and ElasticNet, per-
mutation invariance can easily be achieved by defining xa,b .= [(f a + f b )/2
t]R5, which is equivalent to having equal model weights for the same indices of
f a, f b. Crucially, this averaging approach is only valid for linear models and
would reduce the expressiveness of nonlinear models such as LGBM. Instead, for
LGBM, we augment the dataset to contain xa,b and xb,a. Although helpful, this
approach does not guarantee invariance, so we refer to it as pseudo-permutation
invariant. In both cases, nor- malization is applied after transforming the
training and validation feature vectors. Other models can be directly modified
to learn permu- tation invariant functions. KNN, KRR, and SVR all use distance
metrics to express new points as combinations of training data points; KNN
directly uses a distance metric to select and weight neighbors, and KRR and SVR
use the RBF kernel k(xa,b, xc,d) .= exp (γd(xa,b, xc,d)) (3) for some distance
metric d : R9 R9 [0, ) via the representor theorem. For all three methods, we
use the permutation invariant distance metric d(xa,b, xc,d) .= min f a f c f b f
d t 2 2 , f a f d f b f c t 2 2 . (4) This value can be interpreted as the
minimum distance across all particle permutations within xa,b and xc,d. Finally,
we use the DeepSets [35] neural network (NN) architecture, which has the form g
f a f b t .= ϕη ρ (ψθ(f a), ψθ(f b)) t , (5) 3

---
**페이지 4**
---

where ϕη : R4 Rh, ψθ : Rh+1 (0, 1) are neural net- works with a hidden width of
h and parameters η, θ and ρ : R2h Rh is a feature-wise mean or max. This
architecture is permutation invariant by construc- tion and has been used for
similar particle interaction problems [19]. Together, these approaches make our
pre- dictions (pseudo-)permutation invariant, which improves generalization
capabilities. 2.2.4 Cross-validation To estimate the models performance under
different sce- narios, we considered multiple cross-validation (CV) tech-
niques: 5-fold, leave-one-temperature-out, leave-one-im- pactor-out, and
leave-one-cluster-out. Furthermore, we selected the model parameters using a
grid search to reduce human bias. Notably, estimating model perfor- mance and
performing model selection using the same cross-validation splits is known to
overestimate perfor- mance and lead to biased models [37, 38]. To combat such
bias, we estimated the model performance using nested cross-validation. This
approach estimates model perfor- mance using an outer CV loop, splits each
training set into an inner CV loop, and uses the inner loop to select optimal
hyperparameters for each outer fold. Specifically, for each outer fold, we
select the parameters with the low- est average loss for the inner test datasets
and report the loss of the outer test set using these parameters. This pro-
cess, known as nested CV provides an almost unbiased estimate of the true error
[37]. The inner CV loop was conducted similarly to the outer loop, with the
outer training set being split for the in- ner CV loop. For the 5-fold CV, we
perform the inner CV using another random 5-fold CV. We also perform
leave-one-out CV on the inner loop for leave-one-tem- perature-and
leave-one-cluster-out CV. However, because there were so many impactors, the
inner fold was created by partitioning the training impactors into five folds,
each containing multiple impactors. In short, the inner loop for parameter
selection uses the same split criteria as the outer loop; our preliminary tests
show that this approach improves generalization to unseen clusters and
impactors. We use a modified CV approach to provide train, vali- dation, and
test sets. The data is typically split into train- ing and testing sets of
relative size k 1 and 1. However, since the NN and LGBM utilize validation sets,
we split our inner training dataset for these models into k2 folds for training
and one fold for validation. If a model (e.g., linear regression) didnt use a
validation set, we combined the training and validation sets. An algorithmic
representation is shown in the Supple- mental Materials (SI Algorithms 1 and 2),
while tables of the grid-searched parameters and values are included in Appendix
Section C. We estimate the performance stan- dard deviation by performing nested
CV with five ran- dom seeds (random seeds were shared when splitting the dataset
and initializing the model states). All other pa- rameters were set to the
library defaults. 3 Results and Discussion 3.1 Molecular Simulation Previous
work has analyzed the collisions of SiHx as a critical step for the growth of
particles in silane plasma [15]. However, larger silanes also play a role in
chemical growth despite decreasing concentration. Fig- ure 2 shows Pst, the
probability of a chemisorption or sticking event, for the collision of Si2Hx
with three of the simulated clusters. Figure 2: Temperature dependence of the
sticking prob- ability for collisions between different Si2Hy and a) Si4, b)
Si2H6, and c) Si29H36. Lines show the fitted trend, described in (6). Error bars
represent two standard devi- ations. Si2Hy indicate molecules with balanced
hydrogen distribution, while unbalanced fragments are expanded for clarity.
Similarly to previous work [15], the trends of Pst are 4

*이 페이지에 1개의 이미지가 있습니다.*


---
**페이지 5**
---

well approximated by m(T, E, b, c) = (1 c)f(T b, E) + c , (6) where T is the
translational temperature, E is the ki- netic energy, b and c are fitted
constants, and f is the cu- mulative distribution function of the
Maxwell-Boltzmann distribution: f(T, E) = erf r E kBT ! 2 π r E kBT exp E kBT
(7) in which kB is the Boltzmann constant and erf is the error function.
Compared to the results of SiHy collisions, the Pst for Si2Hy, while generally
slightly higher, displays very simi- lar trends. As expected, the sticking
probability decreases at higher temperatures, with a stronger dependency ob-
served for species with a lower number of radical elec- trons due to their high
reactivity. As before, the number of unpaired radicals plays a crucial role in
the reactivity. However, the picture is complicated by the effect of bal- anced
vs. imbalanced hydrogens on the silane impactors. While a hydrogen imbalance
results in greater reactivity, this effect is secondary to the overall hydrogen
coverage. Outliers like Si2H2 do not follow the expected trend, dis- playing a
lower-than-expected propensity to form bonds at higher temperatures. Cluster
size has a relatively small effect, mostly appar- ent when physisorption is
relevant, whether as a step for the chemisorption or as a collision outcome. By
analyz- ing the ratio between collisions that lead to chemisorption and the
chemisorption and physisorption events (see Fig- ure D1 in the Supplementary
Material), we observe that physisorption plays an integral role in the kinetics
for al- most fully saturated species like Si2H5and Si2H4 isomers. These
reactants, which are less reactive than the more unsaturated counterparts or
require as much energy as the fully saturated silanes, are the most likely to
control the kinetics of particle growth. It is worth noting that the lifetime of
a physisorbed pair can vary from a few to tens of ps, even when the outcome is a
chemisorption. As a result, in an experimental setting, other processes can
occur in this timescale that the current model does not capture. Finally, our
simulations also study Si29Hx nanoparticles and hydrogen coverage to provide a
quantitative relation- ship of the effect of hydrogen coverage of larger NPs on
the sticking coefficients. In Figure 3, we compare sticking coefficients with
coverages of 18, 27, 31, and 36 hydrogens. The same trends observed with other
silane fragments are evident here: increasing hydrogen coverage while holding
the temperature, impactor, and number of cluster silicon atoms monotonically
increases the sticking probability. 3.2 Machine learning models As described in
the Methodology, we performed cross- validations using several splits to
determine which envi- Figure 3: Sticking probability vs. temperature for the
collisions between Si2H6 and Si29 cluster with different hydrogen coverages of
Si29Hx. The line represents the fit discussed in (6), and error bars (generally
smaller than the symbols) represent two standard deviations. ronmental settings
our models could and could not gener- alize. We test the overall capabilities of
the model using a 5-fold CV (Figure 4) and out-of-distribution general- ization
by performing leave-one-temperature-, leave-one- impactor-, and
leave-one-impactor-out CV (Figures 6 and 5, and SI Figure A1). The binomial NLL
depends on the number of trials in each simulation and is nonzero for perfect
predictions. Furthermore, this nonzero floor is not constant and de- pends on
the true probability. Thus, comparing the bino- mial NLL between folds may be
misleading, as different numbers of trial runs or distributions of actual
probabil- ities may dominate variations. To improve visualization, for plotting,
we use the adjusted B-NLL: ℓadj(ˆp, m, n) .= 1 n ℓb (ˆp, m, n) ℓb m n , m, n (8)
This loss weights all simulations equally, regardless of the number of trials
run, and subtracts the NLL of a perfect prediction from the NLL of the actual
prediction. For similar reasons, we plot the unweighted L-MSE instead of the
LW-MSE. Unlike root mean squared error, these metrics do not correspond to
intuitive notions of distance but still facilitate a quantitative performance
compari- son between methods. Thus, we also plot the true vs. predicted
probability for the sticking event to provide an intuition of how individual
models perform. In the 5-fold CV setting (Figure 4), all models per- form
significantly better than the naıve model. Indeed, Figures 4 c) and d) show
almost perfect agreement be- tween true and predicted probabilities.
Furthermore, we find that ML models can be highly robust to data sub- sampling
(Figure 4 e)). Indeed, performance does not meaningfully decrease until less
than 25% of the data is used for training, and good performance is achieved by
training on only 15% of the data. In leave-one-cluster-out testing (Figure 5),
the models performed well for most unsampled Si29Hx clusters but 5

*이 페이지에 1개의 이미지가 있습니다.*


---
**페이지 6**
---

0 0.25 0.5 Adj. B-NLL Model 0 5 10 L-MSE 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob.
0 0.5 1 True Prob. 100% 75% 50% 25% 0% Fraction of data used for Training 0 0.25
0.5 Adj. B-NLL LGBM (Var) a) b) c) d) e) Naïve SVR LGBM KNN DS EN KRR Linear
Figure 4: Performance of 5-fold cross-validation. a) and b) show the average
performance of each model trained and evaluated using the adjusted B-NLL and
L-MSE, re- spectively. Black bars indicate the standard deviation across all
five folds and five random seeds. c) and d) show the LGBM predictions for all
folds and seeds using the same loss functions as a) and b), respectively. e)
shows the adjusted B-NLL performance of the permuta- tion invariant (dark blue)
and variant (light blue) LGBM models and the naıve model as the fraction of
training data decreases. The shaded region indicates the standard deviation
across random seeds. not for Si2H6, likely due to the presence only in the large
cluster of low vibrational frequencies that can better ac- commodate the
collision energy. Since our training set has four Si29Hx clusters and only one
Si2Hx and one Si4Hx clusters, it appears that the model is biased towards the
behavior of the Si29Hx clusters. Notably, most models showed an increased error
for Si29H18. The Si29Hx clusters start with the fully satu- rated Si29H36
molecule and become less saturated until we get to Si29H18. Therefore, we expect
the error to be higher for Si29H36 because the model has only unsat- urated
Si29Hx clusters to train from, while for the re- maining Si29Hx clusters, we
expected similar errors. The slightly abnormal behavior of the predictions for
colli- sions involving Si29H18 suggests interactions dominated by different
reaction pathways, possibly related to H iso- merization. The results for the
impactors are similar (see Si2H6 Si4 Si29H18 Si29H27 Si29H31 0 0.5 1 Adj. B-NLL
Si29H36 0 10 20 L-MSE 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. 0 0.5 1 True Prob.
Split a) b) c) d) Naïve SVR LGBM KNN DS EN KRR Linear Figure 5: Performance of
leave-one-cluster-out CV. a) and b) show the average performance of each model
trained and evaluated using the adjusted B-NLL and L- MSE, respectively. Black
bars indicate the standard de- viation across random seeds. c) and d) show the
predic- tions of LGBM for all folds and random seeds using the same loss
functions as a) and b), respectively. Predic- tions for Si29H18 are highlighted
in red. The losses for SVR Si29H31 and Si29H36 in b) are 23.5 and 41.5 but are
truncated for visualization purposes. Appendix Section A). Overall, the models
are most effec- tive when making predictions for similar-sized molecules, at
least when provided with such a limited selection. While the natural conclusion
about the need for a wider variety of cluster sizes is correct, it should also
be con- fronted with the fact that not all atomic arrangements are equally
stable. The lower energy associated with specific structures (e.g., spherical,
truncated polyhedrons) may result in some clustering of the dominant reactive
path- ways, which may complicate even a model trained on a more varied dataset.
A similar analysis for temperature is shown in Figure 6, where we observe that
the model performance is rela- tively consistent except at the lowest
temperature. The naıve loss is highest for 300 K, and most models (except
DeepSets) perform poorly for 300 K and even worse for 900 K. While we could not
determine the reason for this difference beyond the difficulty of extrapolation
compared to interpolation (which should affect the 900 K), it should 6

---
**페이지 7**
---

300K 400K 500K 600K 0 0.25 0.5 Adj. B-NLL 900K 0 5 10 L-MSE 0 0.5 1 True Prob. 0
0.5 1 Pred. Prob. 0 0.5 1 True Prob. Split a) b) c) d) Naïve SVR LGBM KNN DS EN
KRR Linear Figure 6: Performance of leave-one-temperature-out CV. a) and b) show
the average performance of each model trained and evaluated using the adjusted
B-NLL and L- MSE, respectively. c) and d) show the predictions of DeepSets for
all folds and random seeds using the same loss functions as a) and b),
respectively. The predictions for 300 K are highlighted in red. be noted that
physisorption plays a much more significant role at this temperature, hinting
again at the model sen- sitivity to underlying physical and chemical processes.
As shown in Figure 2 and SI Figure D1, Pst is nearly constant between 700 K and
900 K. Because LGBM learns piecewise-constant functions, it also performs con-
stant extrapolations, which is ideal in this setting. How- ever, Pst is hardly
constant between 300 K and 400 K, meaning that constant extrapolations perform
poorly. This is why other methods, such as DeepSets, outperform LGBM when
extrapolating to lower temperatures and why LGBM outperforms all other methods
but DeepSets when extrapolating to higher temperatures. Notably, we find that
permutation-variance signifi- cantly impacts the generalization capabilities of
all mod- els, as shown in Figure 5. This effect is particularly strong for
Si2H6, Si4, Si29H27. We suspect that this is partial because Si4 and Si2H6 are
more similar in size to the impactors than the other clusters. As a result, a
permutation-variant model learns that the larger par- ticles are usually on one
side, which biases the predic- tions. Indeed, comparing Figures 5 and 7, we find
that permutation-variant models achieve surprisingly high and low performance
depending on the particle being held out, indicating a tendency to fit and an
inability to generalize. Additionally, panels c) and f) in Figure 8 show that
permutation-variant models make highly in- consistent predictions for each
permutation of clusters and impactors. Indeed, predictions are inconsistent be-
tween permutations and are inaccurate unless the model is trained and tested on
the same ordering of clusters and impactors. This variability demonstrates the
importance of permutation-invariance for accurate modeling of Pst in nonthermal
plasmas. Si2H6 Si4 Si29H18 Si29H27 Si29H31 0 0.5 1 Adj. B-NLL Si29H36 0 10 20
L-MSE 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. 0 0.5 1 True Prob. Split a) b) c)
d) Naïve SVR (Var) LGBM (Var) KNN (Var) KRR (Var) EN (Var) Figure 7: Performance
of leave-one-cluster-out CV with permutation-variant models. a) and b) show the
average performance of each model trained and evaluated using the adjusted B-NLL
and L-MSE, respectively. Black bars indicate the standard deviation across
random seeds. c) and d) show the LGBM (Var) predictions for each fold and random
seed using the same loss functions as a) and b), respectively. Here, we permute
the particles before applying the model. Predictions for Si2H6 are highlighted
in red. Visual inspection of true vs. predicted probabilities shows that the
B-NLL provides more robust predictions than the L-MSE. Both losses achieve good
empirical per- formance, indicating that the L-MSE may be suitable when a
binomial NLL loss cannot easily be added to a model. However, the B-NLL achieves
more accurate pre- dictions overall due to its less extreme penalization of out-
7

---
**페이지 8**
---

0 0.5 1 Pred. Prob. a) b) c) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. d) 0 0.5 1
True Prob. e) 0 0.5 1 True Prob. f) Figure 8: a)c) predict Pst for
cluster-impactor pairs, while d)f) predict Pst for impactor-cluster pairs. a),
b), d), and e) show pseudo-permutation invariant LGBM, while c) and f) show
standard LGBM. a), c), d), and f) are trained with the Binomial NLL, while b)
and e) are trained using the LW-MSE. Predictions for SiH3 Si and Si2H4 are
highlighted in blue and red, respectively. All random seeds are shown. liers.
For example, compare the predictions for LGBM trained with each loss on
leave-one-impactor-out cross- validation: Although the worst-performing impactor
is equally bad for both B-NLL and L-MSE, the error is lower for
intermediately-performing impactors such as SiH3 SiH and Si2H4 when considering
the B-NLL (SI Figure A1). This is reminiscent of robust regression, where robust
losses (e.g., Huber loss) are chosen because they do not over-emphasize outliers
as significantly as the squared error. Indeed, the B-NLL loss sits between the
L-MSE and logit-transformed Huber Loss (L-H) in Fig- ure 1. Thus, we recommend
using the Binomial NLL when possible, and we suspect that further improvements
are possible using robust binomial or logistic regression approaches [39]. We
find that, in nearly all settings, ML models sig- nificantly outperform the
naıve prediction. However, we note that models generally perform best when
interpo- lating (rather than extrapolating) for temperature and that better
performance may be achieved at higher tem- peratures. Additionally, DeepSets
excels at extrapolating to unseen temperatures, and LGBM is the most consis-
tent at successfully extrapolating to unseen structures. Finally, the loss
function and permutation invariance can significantly affect model performance
and generalization. This result suggests that, given a correctly chosen model
architecture, we can focus our simulations on a small sub- set of important
conditions (e.g., edge temperatures) to derive Pst across many diverse settings
effectively. 4 Conclusions Particle growth in a high-energy gas phase, such as
dur- ing combustion or in nonthermal plasma, is a complex, non-linear process
that requires accurate modeling. Even when narrowing the scope to a specific
system and set of reactions, capturing the rates and mechanism remains
computationally challenging, even using classical approx- imations. While such
detailed descriptions are not always necessary, a more nuanced description of
the reaction rates can benefit several contexts, such as more accurate reaction
rates, hyper-doping, and core-shell nanoparticle production. The subtle
differences between the various Si2Hy species simulated in this paper, as well
as SiHy, are symptomatic of a series of competing phenomena (e.g.,
physisorption, energy redistribution) that cannot be eas- ily generalized and
that can lead to systematic biases when ignored. To address this, we have
focused on training and testing several permutation-invariant ML models to
reduce the computational effort associated with these simulations. Our results
show that nearly 90% of interactions can be predicted using machine learning
without significantly impacting accuracy. Furthermore, we have demonstrated the
importance of principled loss functions, model archi- tectures, and sampling
procedures for deriving accurate and reliable predictions. Figure 4 shows that,
in general, simple system-specific features are descriptive enough to predict
sticking probabilities for silane nanoparticles in nonthermal plasma after
training on only a fraction of simulated interactions. Additionally, Figure 6
shows that our model can extrapolate and interpolate quite well for unsampled
temperatures. However, Figure 5 indicates our specific combination of ML models
and input features has difficulty extrapolating to nanoparticles with different
degrees of saturation. These findings demonstrate that ML methods can sig-
nificantly reduce the computational cost of computing the results of complex
reactions in nonthermal plasma and other difficult-to-model systems. However,
careful selec- tion of model architecture and training data is crucial to ensure
the generalizability of predictions. Based on our results, we conclude that the
most effective accel- eration method involves simulating a subset of particles
across a range of temperatures (especially the upper and lower bounds of the
expected range), maintaining a bal- ance of relevant molecular properties (e.g.,
H-saturation in this case) in the training set, and then training a
permutation-invariant ML model using the binomial neg- ative log-likelihood.
While this study focused on the data collected for a specific system, namely the
sticking probability of silanes computed through classical reactive molecular
dynamics, the overall approach, both molecular dynamics simula- tions and
analysis of ML models, is relatively general. As such, we expect that similar
methods can be readily adapted to generate more computationally efficient and 8

---
**페이지 9**
---

realistic growth parameters for NTPs, thus improving the efficiency and accuracy
of simulations. 5 Reproducibilty Supporting data and code are currently
available through this link: https://gitlab.eecs.umich.edu/
mattrmd-public/ntp-silicon. Data will be provided via a DOI-minting repository
upon acceptance. 6 CRediT authorship contribu- tion statement Matt Raymond:
Formal analysis, investigation, methodology, software, supervision, validation,
visualiza- tion, writing - original draft, writing - review & editing. Paolo
Elvati: Conceptualization, data curation, investi- gation, methodology,
software, supervision, visualization, writing - original draft, writing - review
& editing. Ja- cob Saldinger: Conceptualization, data curation, soft- ware,
supervision, writing - original draft. Jonathan Lin: Formal analysis,
investigation, software, visualiza- tion, writing - original draft. Xuetao Shi:
Data cura- tion, software. Angela Violi: Conceptualization, fund- ing
acquisition, project administration, resources, super- vision, writing - review
& editing. 7 Declaration of Competing In- terests The authors declare that they
have no known compet- ing financial interests or personal relationships that
could have appeared to influence the work reported in this pa- per.
Acknowledgements This work has been supported by the US Army Research Office
MURI Grant No. W911NF-18-1-0240 and by the NSF ECO-CBET No. F059554. 9

---
**페이지 10**
---

References [1] Mangolini L and Kortshagen U 2009 Phys. Rev. E 79(2) 026405 URL
https://doi.org/10.1103/ PhysRevE.79.026405 [2] Gao X, Cui Y, Levenson R M,
Chung L W K and Nie S 2004 Nature Biotechnology 22 969976 ISSN 1546-1696 URL
https://doi.org/10.1038/nbt994 [3] Fujioka K, Hiruoka M, Sato K, Manabe N,
Miyasaka R, Hanada S, Hoshino A, Tilley R D, Manome Y, Hirakuri K and Yamamoto K
2008 Nanotechnol- ogy 19 415102 URL https://doi.org/10.1088/
0957-4484/19/41/415102 [4] Moore D, Krishnamurthy S, Chao Y, Wang Q, Brabazon D
and McNally P J 2011 physica status solidi (a) 208 604607 URL
https://doi.org/10. 1002/pssa.201000381 [5] Saadane O, Longeaud C, Lebib S and
Roca i Cabarrocas P 2003 Thin Solid Films 427 241246 ISSN 0040-6090 proceedings
of Sym- posium K on Thin Film Materials for Large Area Electronics of the
European Materials Re- search Society (E-MRS) 2002 Spring Conference URL
https://www.sciencedirect.com/science/ article/pii/S0040609002011938 [6] Dogan I
and van de Sanden M C M 2016 Plasma Processes and Polymers 13 1953 URL https://
doi.org/10.1002/ppap.201500197 [7] Weis S, Kormer R, Jank M P M, Lemberger M,
Otto M, Ryssel H, Peukert W and Frey L 2011 Small 7 28532857 URL
https://doi.org/10.1002/smll. 201100703 [8] Astruc D, Lu F and Aranzaes J R 2005
Angewandte Chemie International Edition 44 78527872 URL
https://doi.org/10.1002/anie.200500766 [9] Boufendi L and Bouchoule A 1994
Plasma Sources Science and Technology 3 262 URL https://doi.
org/10.1088/0963-0252/3/3/004 [10] Lanham S J, Polito J, Xiong Z, Kortshagen U R
and Kushner M J 2022 Journal of Applied Physics 132 073301 URL
https://doi.org/10.1063/5. 0100380 [11] Kruger F, Gergs T and Trieschmann J 2019
Plasma Sources Science and Technology 28 035002 URL
https://doi.org/10.1088/1361-6595/ab0246 [12] Kortshagen U and Bhandarkar U 1999
Phys. Rev. E 60(1) 887898 URL https://doi.org/10.1103/ PhysRevE.60.887 [13]
Agarwal P and Girshick S L 2014 Plasma Chemistry and Plasma Processing 34 489503
URL https:// doi.org/10.1007/s11090-013-9511-3 [14] Le Picard R, Markosyan A H,
Porter D H, Girshick S L and Kushner M J 2016 Plasma Chemistry and Plasma
Processing 36 941972 ISSN 1572-8986 URL
https://doi.org/10.1007/s11090-016-9721-6 [15] Shi X, Elvati P and Violi A 2021
Journal of Physics D: Applied Physics 54 365203 [16] Lanham S J, Polito J, Shi
X, Elvati P, Violi A and Kushner M J 2021 Journal of Applied Physics 130 ISSN
0021-8979 URL https://doi.org/10.1063/ 5.0062255 [17] Husmann E, Polito J,
Lanham S, Kushner M J and Thimsen E 2023 Plasma Chemistry and Plasma Processing
43 225245 URL https://doi.org/10. 1007/s11090-022-10299-3 [18] Bal K M and Neyts
E C 2021 Journal of Physics D: Applied Physics 54 394004 URL https://doi.org/
10.1088/1361-6463/ac113a [19] Saldinger J C, Raymond M, Elvati P and Violi A
2023 Proceedings of the Combustion Institute URL
https://doi.org/10.1016/j.proci.2022.08.109 [20] Gidon D, Pei X, Bonzanini A D,
Graves D B and Mesbah A 2019 IEEE Transactions on Radiation and Plasma Medical
Sciences 3 597605 URL https:// doi.org/10.1109/TRPMS.2019.2910220 [21] van der
Gaag T, Onishi H and Akatsuka H 2021 Physics of Plasmas 28 033511 ISSN 1070-664X
URL https://doi.org/10.1063/5.0023928 [22] Liang C, Huang D, Lu S and Feng Y
2023 Phys. Rev. Res. 5(3) 033086 URL https://doi.org/10.1103/
PhysRevResearch.5.033086 [23] Han S, Ceiler M, Bidstrup S, Kohl P and May G 1994
IEEE Transactions on Components, Packaging, and Manufacturing Technology: Part A
17 174182 URL https://doi.org/10.1109/95.296398 [24] Guessasma S, Montavon G and
Coddet C 2004 Computational Materials Science 29 315333 ISSN 0927-0256 URL
https: //doi.org/10.1016/j.commatsci.2003.10.007 [25] Pakseresht A H, Ghasali E,
Nejati M, Shirvani- moghaddam K, Javadi A H and Teimouri R 2015 The
International Journal of Advanced Manufactur- ing Technology 76 10311045 ISSN
1433-3015 URL https://doi.org/10.1007/s00170-014-6212-x [26] Gergs T,
Mussenbrock T and Trieschmann J 2023 Journal of Physics D: Applied Physics 56
194001 URL https://doi.org/10.1088/1361-6463/ acc07e 10

---
**페이지 11**
---

[27] Hong S, May G and Park D C 2003 IEEE Trans- actions on Semiconductor
Manufacturing 16 598 608 URL https://doi.org/10.1109/TSM.2003. 818976 [28] Kim B
and May G 1994 IEEE Transactions on Semiconductor Manufacturing 7 1221 URL
https: //doi.org/10.1109/66.286829 [29] Kwon O, Lee N and Kim K 2022 IEEE
Transactions on Semiconductor Manufacturing 35 256265 URL
https://doi.org/10.1109/TSM.2022.3154366 [30] Abd Jelil R, Zeng X, Koehl L and
Perwuelz A 2013 Engineering Applications of Artificial Intelligence 26 18541864
ISSN 0952-1976 URL https://doi.org/ 10.1016/j.engappai.2013.03.015 [31] Plimpton
S 1995 Journal of Computational Physics 117 119 URL
https://doi.org/10.1006/jcph. 1995.1039 [32] Rappe A K and Goddard W A 1991 The
Journal of Physical Chemistry 95 33583363 [33] Nakano A 1997 Computer Physics
Communications 104 5969 URL https://doi.org/10/cmbzsw [34] Pedregosa F,
Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer
P, Weiss R, Dubourg V, Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M
and Duches- nay E 2011 Journal of Machine Learning Research 12 28252830 version:
1.5.1 URL https://dl.acm. org/doi/10.5555/1953048.2078195 [35] Zaheer M, Kottur
S, Ravanbakhsh S, Poczos B, Salakhutdinov R R and Smola A J 2017 Deep sets
Advances in Neural Information Processing Systems vol 30 ed Guyon I, Luxburg U
V, Bengio S, Wallach H, Fergus R, Vishwanathan S and Garnett R (Curran
Associates, Inc.) URL https:// papers.nips.cc/paper_files/paper/2017/hash/
f22e4747da1aa27e363d86d40ff442fe-Abstract. html [36] Ke G, Meng Q, Finley T,
Wang T, Chen W, Ma W, Ye Q and Liu T Y 2017 Ad- vances in neural information
processing sys- tems 30 31463154 URL https://papers.
nips.cc/paper_files/paper/2017/hash/ 6449f44a102fde848669bdd9eb6b76fa-Abstract.
html [37] Varma S and Simon R 2006 BMC Bioinformatics 7 91 ISSN 1471-2105 URL
https://doi.org/10. 1186/1471-2105-7-91 [38] Cawley G C and Talbot N L C 2010
Journal of Ma- chine Learning Research 11 20792107 URL http:
//jmlr.org/papers/v11/cawley10a.html [39] Feng J, Xu H, Mannor S and Yan S 2014
Robust logistic regression and classification Advances in Neural Information
Processing Systems vol 27 ed Ghahramani Z, Welling M, Cortes C, Lawrence N and
Weinberger K (Curran Associates, Inc.) URL
https://papers.nips.cc/paper/2014/hash/
6cdd60ea0045eb7a6ec44c54d29ed402-Abstract. html 11

---
**페이지 12**
---

A Leave Impactor Out Si2H Si2H2 Si2H4 Si2H5 Si2H6 SiH SiH2 SiH2 Si SiH2 SiH SiH3
SiH3 Si SiH3 SiH 0 0.25 0.5 0.75 Adj. B-NLL SiH4 0 10 20 L-MSE Split a) b) Naïve
SVR LGBM KNN DS EN KRR Linear Figure A1: Performance of leave-one-impactor-out
CV. a) and b) show the average performance of each model trained and evaluated
using the adjusted B-NLL and L-MSE, respectively. Black bars indicate the
standard deviation across random seeds. For leave-one-impactor-out (SI Figure
A1), some models perform especially poorly for SiH, SiH2, SiH3 Si, and SiH3 SiH.
There are several potential explanations for this. One is that the naıve
prediction error for these particles is already significantly lower than the
other particles, leaving less room for improvement. For a couple of impactors,
namely SiH4 and SiH3 SiH3, most models had nearly double the error as they did
for other impactors. This may be caused by overfitting due to the large amount
of remaining data for partially saturated molecules, suggesting that optimal
training data would contain a higher percentage of fully saturated molecules.
Alternatively, it could simply mean that the sticking probabilities for this
molecule are outliers. Regardless, DeepSets (and LGBM) significantly outperform
the naıve model in all (most) cases. B CV Method In Supplementary Algorithms 1
and 2, we include the details of the cross-validation methods used in this work.
12

---
**페이지 13**
---

Input : Dataset D, parameter grid G, parametric model fθ,g Output: Performance
metric for each outer testing set // Arrays holding outer test metric µ [];
foreach g G do foreach k [5] do k (k + 1) mod 5; // Outer cross-validation loop
Douter train D split into 5 folds with the k and k-th folds removed; Douter test
k-th fold of D; Douter val k-th fold of D; // Array of losses ηg []; foreach j
[5] do // Inner cross-validation loop Dinner train Douter train split into 5
folds with the j and j-th folds removed; Dinner test j-th fold of Douter train ;
Dinner val j-th fold of Douter train ; θarg minθ fθ,g with training and
validation sets Dinner train and Dinner train ; Append the loss computed on
Dinner test to ηg; end // Select the optimal parameters garg ming P ηg/5; // Use
these parameters to retrain the model θarg minθ fθ,gwith training and validation
sets Douter train and Douter train ; Append the loss computed on Douter test to
µ; end end // Return array of performance metrics on the test sets return µ
Algorithm B1: Grid search with nested 5-fold cross-validation for one seed. 13

---
**페이지 14**
---

Input : Dataset D, parameter grid G, set of groups C, parametric model fθ,g
Output: Performance metric for each outer testing set // Arrays holding outer
test metric µ []; foreach g, G do foreach k [|C|] do k (k + 1) mod |C|; // Outer
cross-validation loop Douter train D split into groups from C with the k and
k-th groups removed; Douter test k-th group of D; Douter val k-th group of D; //
Array of losses ηg []; foreach j [|C \ {k, k} |] do // Inner cross-validation
loop Dinner train Douter train split into groups from C \ {k, k} with the j and
j-th groups removed; Dinner test j-th group of Douter train ; Dinner val j-th
group of Douter train ; θarg minθ fθ,g with training and validation sets Dinner
train and Dinner train ; Append the loss computed on Dinner test to ηg; end //
Select the optimal parameters garg ming P ηg/|C \ {k, k} |; // Use these
parameters to retrain the model θarg minθ fθ,gwith training and validation sets
Douter train and Douter train ; Append the loss computed on Douter test to µ;
end end // Return array of performance metrics on the test sets return µ
Algorithm B2: Grid search with nested leave-x-out cross-validation for one seed.
In our setting, a group may be a cluster, impactor, or temperature. When |C| is
large, we instead let j, j be sets of groups instead of individual groups to
reduce runtime. 14

---
**페이지 15**
---

C Grid Search Parameters used for model selection in the inner 5-fold
cross-validation described in the Methods, Section 3.2. Parameter Name Values
Epochs 100 000 Early stopping epochs 1 000 Activation function relu Batch size
64 Width {64, 128} Depth of each subnetwork {2, 3} Learning rate 0.001
Aggregation function ρ {mean, max} Optimizer adam Table C1: Grid search
parameters for DeepSets Parameter Name Values α (ℓ1, ℓ2 penalty weight) {0.0001,
0.001, 0.01, 0.1, 1, 10} ℓ1 ratio {0, 0.25, 0.5, 0.75, 1} Maximum iterations 10
000 000 Table C2: Grid search parameters for ElasticNet (LW-MSE) Parameter Name
Values C (regularization) {0.1, 1, 10, 100, 1 000} ε (tube) {0.0001, 0.001,
0.01, 0.1, 1} γ (scale for RBF kernel) {0.01, 0.1, 1, 10} Table C3: Grid search
parameters for SVR Parameter Name Values α (regularization) {0.01, 0.1, 1, 10,
100} γ (scale for RBF kernel) {0.01, 0.1, 1, 10, 100} Table C4: Grid search
parameters for KRR Parameter Name Values Neighbors {3, 6, 9} Weighting method
{uniform, distance} p (for the ℓp-norm distance) {1, 2} Table C5: Grid search
parameters for KNN Parameter Name Values Number of estimators {100, 1 000} alpha
(ℓ1 regularization) {0, 0.1, 1} λ (ℓ2 regularization) {0, 0.1, 1.0} Table C6:
Grid search parameters for LGBM 15

---
**페이지 16**
---

Parameter Name Values C (inverse of regularization strength) {0.01, 0.1, 1.0} ℓ1
ratio {0, 0.25, 0.5, 0.75, 1} Solver saga Penalty elasticnet Table C7: Grid
search parameters for ElasticNet (Binomial loss via Logistic Regression) D MD
additional results 300 400 500 600 700 800 900 Temperature, K 0.0 0.2 0.4 0.6
0.8 1.0 Pst Si2H6 Si2H5 Si2H4 SiH3SiH SiH2SiH SiH3Si Si2H2 SiH2Si Si2H Figure
D1: Fraction of chemisorption sticking events between Si29H36 and different
Si2Hx silicon fragments at various temperatures. 16

---
**페이지 17**
---

E True vs. Predicted values Here, we plot each cross-validation procedures true
vs. predicted sticking probability. We plot the test points for every
cross-validation loop that used the selected parameters. 17

---
**페이지 18**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KNN 0 0.5 1
True Prob. KNN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. EN 0 0.5 1 True
Prob. EN (Var) Five-Fold CV, Binomial Figure E1: True vs. predicted
probabilities for each model using nested five-fold cross-validation and the
Binomial loss. 18

---
**페이지 19**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KRR KRR (Var) 0
0.5 1 Pred. Prob. SVR SVR (Var) KNN 0 0.5 1 Pred. Prob. KNN (Var) EN 0 0.5 1
True Prob. EN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. Linear 0 0.5 1 True
Prob. Linear (Var) Five-Fold CV, LW-MSE Figure E2: True vs. predicted
probabilities for each model using nested five-fold cross-validation and the
LW-MSE loss. 19

---
**페이지 20**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KNN 0 0.5 1
True Prob. KNN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. EN 0 0.5 1 True
Prob. EN (Var) Leave Cluster Out, Binomial Figure E3: True vs. predicted
probabilities for each model using nested leave-one-cluster-out cross-validation
and the Binomial loss. 20

---
**페이지 21**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KRR KRR (Var) 0
0.5 1 Pred. Prob. SVR SVR (Var) KNN 0 0.5 1 Pred. Prob. KNN (Var) EN 0 0.5 1
True Prob. EN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. Linear 0 0.5 1 True
Prob. Linear (Var) Leave Cluster Out, LW-MSE Figure E4: True vs. predicted
probabilities for each model using nested leave-one-cluster-out cross-validation
and the LW-MSE loss. 21

---
**페이지 22**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KNN 0 0.5 1
True Prob. KNN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. EN 0 0.5 1 True
Prob. EN (Var) Leave Temperature Out, Binomial Figure E5: True vs. predicted
probabilities for each model using nested leave-one-temperature-out cross-
validation and the Binomial loss. 22

---
**페이지 23**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KRR KRR (Var) 0
0.5 1 Pred. Prob. SVR SVR (Var) KNN 0 0.5 1 Pred. Prob. KNN (Var) EN 0 0.5 1
True Prob. EN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. Linear 0 0.5 1 True
Prob. Linear (Var) Leave Temperature Out, LW-MSE Figure E6: True vs. predicted
probabilities for each model using nested leave-one-temperature-out cross-
validation and the LW-MSE loss. 23

---
**페이지 24**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KNN 0 0.5 1
True Prob. KNN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. EN 0 0.5 1 True
Prob. EN (Var) Leave Impactor Out, Binomial Figure E7: True vs. predicted
probabilities for each model using nested leave-one-impactor-out
cross-validation and the Binomial loss. 24

---
**페이지 25**
---

0 0.5 1 Pred. Prob. Naïve LGBM LGBM (Var) 0 0.5 1 Pred. Prob. DS KRR KRR (Var) 0
0.5 1 Pred. Prob. SVR SVR (Var) KNN 0 0.5 1 Pred. Prob. KNN (Var) EN 0 0.5 1
True Prob. EN (Var) 0 0.5 1 True Prob. 0 0.5 1 Pred. Prob. Linear 0 0.5 1 True
Prob. Linear (Var) Leave Impactor Out, LW-MSE Figure E8: True vs. predicted
probabilities for each model using nested leave-one-impactor-out
cross-validation and the LW-MSE loss. 25

---
**페이지 26**
---

F Example Input and Output Cluster Descriptors Impactor Descriptors
Environmental Descriptors Outputs #Si #h #e1 #e2 #Si #h #e1 #e2 Temperature (K)
Probability 4 0 8 0 1 1 3 0 300 0.979 2 6 0 0 1 4 0 0 600 0.000 ... ... ... ...
... ... ... ... ... ... Table F1: Example inputs and outputs for the machine
learning models (assuming no pre-processing). # Si and # h indicate the number
of silicon and hydrogen atoms, respectively. # e1 and # e2 indicate the first
and second elements in the vector #e describing the number of unpaired electrons
per silicon atom. 26