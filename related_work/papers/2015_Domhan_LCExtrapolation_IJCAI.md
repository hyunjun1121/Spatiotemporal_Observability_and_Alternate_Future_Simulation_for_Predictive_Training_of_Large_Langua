
---
**페이지 1**
---

HU-EP-15/08 Boundary and Interface CFTs from the Conformal Bootstrap Ferdinando
Gliozzi,1 Pedro Liendo,2 Marco Meineri,3 Antonio Rago4 1Dipartimento di Fisica,
Università di Torino, and Istituto Nazionale di Fisica Nucleare - sezione di
Torino, Via P. Giuria 1 I-10125 Torino, Italy 2IMIP, Humboldt-Universität zu
Berlin, IRIS Adelershof, Zum Großen Windkanal 6, 12489 Berlin, Germany 3Scuola
Normale Superiore, Piazza dei Cavalieri 7 I-56126 Pisa, Italy and Istituto
Nazionale di Fisica Nucleare - sezione di Pisa 4Centre form Mathematical
Science, Plymouth University, Drake Circus, Plymouth PL4 8AA, United Kingdom
Abstract: We explore some consequences of the crossing symmetry for defect
conformal ﬁeld theories, focusing on codimension one defects like ﬂat boundaries
or interfaces. We study surface transitions of the 3d Ising and other O(N)
models through numerical solutions to the crossing equations with the method of
determinants. In the extraordinary transition, where the low-lying spectrum of
the surface operators is known, we use the bootstrap equations to obtain
information on the bulk spectrum of the theory. In the ordinary transition the
knowledge of the low-lying bulk spectrum allows to calculate the scale dimension
of the relevant surface operator, which compares well with known results of
two-loop calculations in 3d. Estimates of various OPE coeﬃcients are also
obtained. We also analyze in 4-ϵ dimensions the renormalization group interface
between the O(N) model and the free theory and check numerically the results in
3d. Keywords: Conformal Field Theory, Conformal Bootstrap, Critical Ising Model
arXiv:1502.07217v4 [hep-th] 24 Nov 2021

---
**페이지 2**
---

Contents 1 Introduction and motivations. 1 2 Defect CFTs and the method of
determinants. 4 3 The boundary bootstrap and the 3d Ising and O(N) models. 8 3.1
The ordinary transition. 10 3.2 The extraordinary transition. 12 3.3 The special
transition. 15 4 Renormalization group domain wall for the O(N) model. 17 4.1
The ϵ-expansion and the role of the displacement operator. 18 4.2 Leading order
mixing of primary operators. 23 4.3 The interface bootstrap. 25 5 Conclusions
and outlook. 28 A RG domain wall: details on the ϵ-expansion. 31 A.1 One loop
computations. 31 A.2 Two-point functions across the interface. 34 1 Introduction
and motivations. Conformal ﬁeld theories (CFTs) play in many senses a preeminent
role among quantum and statistical ﬁeld theories. Such a privileged position is
ﬁrst granted by the ﬂow of the renormal- ization group, whose ﬁxed points are
scale invariant theories, which usually show full confor- mal invariance [1, 2].
More generally, approximate scale invariance is a feature of systems in which a
wide separation of scales makes the ﬂow very slow in intermediate regions.
Through the renormalization group, nature realizes the theories possessing the
maximum amount of bosonic symmetry, both in condensed matter and in particle
physics, in appropriate UV and IR regimes. Reversing the argument, one can
understand a generic quantum ﬁeld theory as a CFT deformed by a set of relevant
operators. All perturbative analyses are in fact justiﬁed by the small size of
relevant couplings in the UV limit. One can even pursue non-perturbative
explorations of RG ﬂows using the ultraviolet data as the only input [3] (see
also [4, 5] and references therein). As a consequence, the importance of
conformal invariance exceeds the experimental interest: conformal ﬁeld theories
are among the main actors in formal investi- gations of the space of quantum
ﬁeld theories, which has seen a huge development in recent 1

---
**페이지 3**
---

times. Furthermore, they are an invaluable tool for studying quantum gravity,
through the AdS/CFT correspondence [6]. The most striking feature of a generic
CFT is that, however strongly coupled, it is com- pletely described by two sets
of numbers: the spectrum of scale dimensions of operators of every spin, and the
Operator Product Expansion coeﬃcients. This simpliﬁcation occurs be- cause the
predictive power of the OPE is boosted by the conformal symmetry. On one hand,
irreducible representations of the conformal group gather inﬁnitely many
operators, and the contribution to the OPE of every conformal family is labeled
by the dimension and spin of the highest weight and is ﬁxed up to a single
coeﬃcient. On the other hand, the OPE converges inside correlation functions
[7], and can be repeatedly used to reduce all of them to a sum over functions of
the kinematic variables, the so called conformal blocks, one for each conformal
family. This pairwise reduction can be carried out fusing operators in various
diﬀerent orders, so that sums over diﬀerent blocks need to be equal. The
crossing equations obtained this way provide constraints on the possible CFT
data [8], and after the seminal paper [9], a wealth of new results on the space
of conformal ﬁeld theories in dimensions greater than two were found by
exploiting these constraints [1028]. The method proposed in [12, 16] which we
refer to as the linear functional method, relies on unitarity to ﬁnd forbidden
regions in the space of the CFT data, by considering particular channels in the
conformal block decomposition of a four- point function. At the boundary of
these regions a spectrum which is crossing symmetric up to some maximum scale
dimension can be extracted numerically [16]. It is not diﬃcult to show that the
four-point functions of local operators on the vacuum encode all of the
constraints coming from crossing symmetry: however, one needs in principle all
of them, and therefore the trial spectrum extracted from a speciﬁc correlator is
not guaranteed to correspond to a unitary CFT. Sometimes it does, though [13],
or maybe a set of minimal hypotheses on the spectrum can be put in place to
lower the bound disregarding uninteresting solutions which stand in the way
[18]. Another possibility is to consider more than one four-point function, so
that further requirements on the spectrum can be made: for instance, internal
symmetries diﬀerentiate the set of primaries appearing in diﬀerent OPEs. This
strategy was applied to the 3d Ising model in [24], providing strong evidence
that the presence of Z2 symmetry and two relevant primaries deﬁnes only one
theory. The reader is referred to the aforementioned papers for a detailed
explanation of the linear functional method. Here we shall employ a diﬀerent
technique, introduced in [17], which we review in section 2. The method of
determinants is based on the choice of a truncation of the spectrum, and
directly provides an approximate solution to the crossing equation. It is
independent from unitarity and can be applied to any correlator. On the other
hand, it is not yet completely automated, and this makes it diﬃcult in practice
to deal with truncations involving many primaries. As a consequence, estimating
the size of the systematic error is a delicate matter. We shall comment on this
issue along the way. The aim of this paper is to apply the conformal bootstrap
program to some examples of defect conformal ﬁeld theories. These are theories
in which the conformal group is broken down to the stabilizer of some
hypersurface. We shall be concerned only with the case of a 2

---
**페이지 4**
---

codimension one hyperplane, alias a ﬂat interface, but the considerations in
section 2 apply to generic ﬂat conformal defects. Motivations for studying
conformal defects are again both phenomenological and abstract. For instance,
conformal defects describe modiﬁcations of a d dimensional QFT localized near a
p dimensional plane, with p d, in the infrared limit, provided these
modiﬁcations are not swept away by coarse graining, and scale invariance is
enhanced to invariance under the conformal group SO(p + 1, 1). The simplest
example is of course a conformal boundary - that is, an interface between a
non-trivial and the trivial CFT. Lower dimensional defects may correspond to
magnetic-like impurities in a spin system, see for instance [29], or to
dispersionless fermions, acting as a source for the order parameter of some
bosonic system [30], or to vortices in holographic superﬂuids and
superconductors [31], etc. On the more abstract side, extended defects are
probes of a system, and may be used to constrain properties of the bulk CFT. We
shall in fact see this happening in the present study. Moreover, interfaces are
a natural way to compare two theories, and may provide information on the
geometric structure of the space of CFTs [32]. The conformal bootstrap was ﬁrst
applied to the boundary setup in [14], while the twist line defect deﬁned in
[29] was tackled in [18]. Both papers are concerned with the 3d Ising model, and
both used the linear functional method. In the latter, four-point functions of
defect operator s were considered, while the former focused on two-point
functions of bulk operators. Correlators of defect operators are blind to
bulk-to-defect couplings, but correlators of bulk primaries do not satisfy in
general the positivity constraints required by the linear functional method, and
ad hoc assumptions were made in [14], motivated by computations in 2d and in
ϵ-expansion. Here we concentrate on the two-point function of bulk scalar
primaries, using the method of determinants, which can be safely applied to this
case. Since our main interest is again the 3d Ising model, we compare our
results for the special and the extraordinary transitions with those of [14]. We
also ﬁnd approximate solutions to the crossing equations corresponding to the
ordinary transition, which cannot be studied with the linear functional. In the
latter case we extended the analysis to the O(N) models with N = 0, 2, 3, where
a comparison can be made with two-loop calculations. The main results are
summarized in the tables 1 and 2. In the end, we initiate the study of an
example of RG domain wall, an interface between two CFTs connected by the
renormalization group, which is obtained by turning on a relevant deformation on
half of the space and ﬂowing to the IR. Speciﬁcally, we study the ﬂow triggered
by the (φ2)2 coupling in a bosonic theory. We give a ﬁrst order description in
ϵ-expansion which applies to models with O(N) symmetry and can be easily
generalized to other perturbation interfaces. We then focus on the Ising model
when looking for a numerical solution to the crossing equations in 3d. The
structure of the paper is as follows. In section 2 we review the general
features of conformal ﬁeld theories in the presence of defects, and we explain
the method of determinants. Section 3 is devoted to the study of the boundary
CFTs associated to the 3d Ising and other spin systems. We deﬁne and study the
domain wall in section 4. Finally, we draw our conclusions in section 5.
Appendix A contains some details of the ϵ-expansion computations. 3

---
**페이지 5**
---

## 2 Defect CFTs and the method of determinants. The constraints imposed by conformal symmetry on correlation functions near a boundary were analyzed in [33] (see also [34]), and the boundary bootstrap was set up in [14], from which we borrow the notation. Here we review the necessary material, and then introduce the method of determinants. A general p-dimensional defect diﬀers from the codimension one case for the residual SO(d p) symmetry generated by rotations around the defect. This is just a ﬂavor symmetry for the defect operators, but induces some diﬀerences when it comes to bulk-to-defect couplings. Although most of what we shall say applies to a generic ﬂat defect, in this paper we shall be concerned with the codimension one case. Therefore, further reference to the general case are limited to some side comments. Correlation functions of excitations living at the defect are the same as in an ordinary (d 1)-dimensional CFT, and are completely characterized by the spectrum of scale dimen- sions (bl) and the coeﬃcients of three-point functions ( bλlmn). We shall later need one more piece of information. While no conserved stress-tensor is expected to exist on the defect, a protected scalar operator of dimension d or p+1 in the general case is always present: the displacement operator, which we call D(xa), measures the breaking of translational invariance, and is deﬁned by the Ward identity for the stress-tensor: µT µd(x) = D(xa) δ(xd). (2.1) Here we denoted by latin indices the directions along the defect, which is placed at xd = 0, while Greek letters run from 1 to d. Similarly, for every bulk current whose conservation is violated by the defect, a protected defect operator exists. In the bulk, there is of course the usual OPE. For scalar primaries, O1(x)O2(y) = δ12 (x y)21 + X k λ12kC[x y, y]Ok(y) , (2.2) where C[x y, y] are determined by conformal invariance, and we isolated the contribution of the identity. One can also fuse a local operator with the defect. The bulk operator is thus turned into a sum over defect primaries. The bulk-to-defect OPE for a scalar primary can be written O1(x) = a1 |2xd|1 + X l µ1lD[xd, a] bOl(xb) , (2.3) where we denoted defect operators with a hat. Again, the diﬀerential operators D[xd, a] are ﬁxed by conformal invariance. Similar OPEs can be written for bulk tensors. The λ12ks in eq. (2.2) are the coeﬃcients of three-point functions without the defect, while µl is the coeﬃcient of the correlator O(x) bOl(ya), otherwise ﬁxed by conformal symmetry. Even if, for the sake of simplicity, some abuse of notation is present1, in this paper all OPE coeﬃcients refer to canonically normalized operators, with one exception: the normalization of the displacement 1For instance, the coeﬃcient µφ2D in free theory appears in the two point function  φ2  2N D.  4

---
**페이지 6**
---

operator is ﬁxed by eq. (2.1). Taking the expectation value of both sides in eq.
(2.3) one sees that a scalar acquires a one-point function proportional to aO,
the coeﬃcient of the identity in the bulk-to-defect OPE. It is not diﬃcult to
prove that tensors do not acquire an expectation value in the presence of a
codimension one defect. They do, instead, if they are even spin representations
and the defect is lower dimensional. Let us now derive the easiest crossing
equation involving the OPEs (2.2) and (2.3). Con- sider the two-point function
O1(x)O2(x). One can decompose it into the bulk channel by plugging in eq. (2.2):
a sum over one-point functions is obtained, that is, a sum over the coeﬃcients
λ12kak multiplying some known functions of the kinematic variables. Or, one can
substitute both operators with their Defect OPE, and in this case the sum
involves the quanti- ties µ1lµ2l. In order to write explicitly the equality of
the two conformal block decompositions, let us introduce the conformal invariant
combination ξ = (x x)2 4xdxd . (2.4) This cross-ratio is conveniently positive
when both points are chosen in the half-plane xd 0. This is not the case when
considering bulk operators on opposite sides of an interface. Moreover, in this
setup the bulk OPE is not deﬁned. The issue is solved by folding the system and
treating it as a boundary CFT: the folding trick provides us with a trivial OPE,
ﬁxed by the absence of local interactions between the two primaries. We shall
have more to say on this point in section 4. For now, we just point out that the
natural cross-ratio is the one constructed from a point and the mirror image of
the second one, and it is again positive. We assume ξ 0 in the rest of this
section. Conformal symmetry justiﬁes the following parametrization: O1(x)O2(x)=
1 (2xd)1(2xd)2 ξ(1+2)/2G12(ξ). (2.5) Then the crossing equation can be written
as a double decomposition of the function G12(ξ): G12(ξ) = δ12 + X k λ12k ak
fbulk(12, k; ξ) = ξ(1+2)/2 a1a2 + X l µ1l µ2l fbdy(bl; ξ) ! , (2.6) where [33]
fbulk(12, , ξ) = ξ/2 2F1 1 2(1 2 + ), 1 2(2 1 + ); + 1 d 2, ξ , (2.7a) fbdy(, ξ)
= ξ2F1 , + 1 d 2; 2+ 2 d; 1 ξ . (2.7b) It is worth noticing that the conformal
blocks of the boundary channel in d = 3 can be expressed as elementary algebraic
functions, namely, fbdy(, ξ)|d=3 = 1 2ξ 4 1 + ξ 1 2 " 1 + s ξ 1 + ξ #2(1) .
(2.8) 5

---
**페이지 7**
---

This is of course of great help in numerical calculations. Before describing how
to extract information from eq. (2.6), we make some side remarks. The set {bl,
bλlmn, i, λijk, ai, µl} is in fact redundant: by repeatedly applying the
bulk-to- defect OPE one can reduce all correlators to correlators of defect
operators, therefore the λijk are in principle unnecessary to solve the theory.
However, it is easy to realize that all crossing equations constraining the
bulk-to-defect couplings µl also involve the bulk three- point function
coeﬃcients. One is naturally led to the following question: what is the minimal
set of correlators encoding all the crossing symmetry constraints of a Defect
CFT? All the four-point functions of defect operators are surely in the number,
the proof being the usual one (see for instance [35]). A similar argument shows
that all the other crossing equations of a generic correlator of bulk and defect
primaries are automatically satisﬁed once the three-point functions O1O2 bOare
crossing symmetric. In the rest of this paper we explore the case bO = 1,
leaving for future work the general case. Let us now turn our attention back to
eq. (2.6) that we rewrite in the following form X k λ12k ak fbulk(12, k; ξ) +
ξ(1+2)/2 a1a2 + X l µ1l µ2l fbdy(bl; ξ) ! = δ12 . (2.9) In most situations, an
inﬁnite number of operators contributes to both channels, which makes the
crossing constraint diﬃcult to exploit. The strategy described in [17] can be
summarized in the following way. First, we trade one functional equation for
inﬁnitely many linear equations: one for each coeﬃcient of the Taylor expansion
around, say, ξ = 1. Then we truncate both the Taylor expansions, keeping only
the ﬁrst M derivatives, and the spectrum, keeping the ﬁrst N operators in total
from the two channels. The bulk identity is excluded from the count. We denote
this truncation with a triple (nbulk, nbdy, s), the three numbers counting
respectively bulk and boundary operators of non vanishing dimension, and the
presence (s = 1) or absence (s = 0) of the boundary identity. We obtain this way
a ﬁnite system, at the price of introducing a systematic error, coming from the
disregarded higher order derivatives and heavier operators: nbulk X k pk fk bulk
ξ=1 + nbdy X l ql fl bdy ξ=1 + a1a2 = δ12, nbulk + nbdy + s = N , nbulk X k pk n
ξ fk bulk ξ=1 + nbdy X l ql n ξ ξ(1+2)/2fl bdy ξ=1 + a1a2 n ξ ξ(1+2)/2 ξ=1 = 0,
n = 1, . . . , M , (2.10) where we used a shorthand notation for the OPE
coeﬃcients pk = λ12kak, ql = µ1lµ2l . Let us focus for deﬁniteness on the case
of two identical external scalars, δ12 = 1. The pks, qls and a2 1 are the
unknowns of a linear system whose coeﬃcients depend nonlinearly on the bulk and
defect spectra. Choosing M N, the homogeneous system, i.e. the second line in
(2.10), admits a non-trivial solution if and only if all the M N minors of the
system vanish. This condition provides a set of non-linear equations in the N
unknown scale dimensions. When 6

---
**페이지 8**
---

this set admits a (numerical) solution we say that the the two-point function
under study is truncable. In such a case, inserting the obtained (approximate)
spectrum in the complete linear system (2.10), we get the OPE coeﬃcients. Notice
that every consistent CFT data is in particular a solution to this crossing
equation. Therefore, some input has to be provided: here we are implicitly
assuming that the external dimensions are known, and in fact this is going to be
the strategy when we try to isolate the 3d Ising model. One does not expect to
ﬁnd an exact solution for a generic truncation: heavier defect and bulk
operators become more and more important when moving respectively towards the
bulk (ξ 0) or the defect (ξ ), therefore we expect a good truncation to require
N to grow with M. In practice, in this work we usually choose M = N + 1, and we
ﬁnd that the space of solutions to the system of nonlinear equations has in
general non-zero dimension. By ﬁxing the free parameters with the best known
values of the lowest lying bulk primaries, we give predictions for the low lying
defect spectrum and for heavier primaries. As a general rule, a ﬁnite truncation
of the crossing symmetry equations is a good ap- proximation of a given CFT if
the missing operators can be consistently put at = or at zero coupling. When a
trial spectrum has been found, one can check its stability by adding one
operator and one derivative. It turns out in most cases that the scaling
dimension of the new operator acts as a free parameter which can vary in a ﬁxed
range. We use the solution for predictions only if it does not depend very
strongly on this parameter. This gives a way of controlling the systematic
error, albeit not an algorithmic one. Let us also observe that the general
agreement with the results of the epsilon expansion suggests that the error is
rather small, at least for what concerns the boundary case. Another important
check comes from the Ward identity associated with the displacement operator,
which, as we shall see, yields non- trivial relations among the CFT data. These
relations are perfectly veriﬁed by the numerical solutions, as described in the
next section. Another parameter to be considered in order to check the quality
of a given truncation is the spread of the solutions. As soon as the number M of
equations exceeds the number of unknowns, the system is over-determined and can
be split in consistent subsystems, each of them giving in principle a diﬀerent
solution. The spread of these solution gives a rough estimate of the error. In
the cases where the exact solution is known the narrower is the spread the
closer is the solution to its exact value. This is the case for instance of the
four- point function of the free scalar massless theory in any dimension [17].
On the contrary large spreads are associated to large systematic errors due to
too rough approximations of the crossing equations. A clear illustration of this
behavior can be found in the ordinary transition of the 2d Ising model, where
the exact two-point function is known [36]. Assuming we already know the bulk
spectrum, we can start considering the truncation (2,1,0) to evaluate the scale
dimensions of the ﬁrst surface operator. We have to look at the zeros of 3 3
determinants. Taking for instance 8 derivatives we have 56 equations whose
solutions are plotted in the histogram of of ﬁg. 1. Their large spread is
associated with a rather rough approximation of the sum rule (2.9) as ﬁg. 2
shows. The same ﬁgure points out also that the truncation (4,3,0) is much
better. In this case the unknowns are the dimensions of the three surface
operators. 7

---
**페이지 9**
---

0 20 40 60 0 20 40 60 D 0.47 0.48 0.49 0.5 0.51 0.52 0.50090 0.50095 0.50100 D 5
10 15 20 Figure 1. Top panel: paired histograms of the solutions of two diﬀerent
truncations of the crossing equations for the ordinary transition of the 2d
Ising model. Left: histogram for the scale dimensions of the ﬁrst boundary
operator in the (2,1,0) truncation. The exact result is at b= 1 2. Right: the
corresponding histogram for the (4,3,0) truncation. Bottom panel: a more
detailed view of the latter histogram. The consistent subsystems are made of
sets of three 7 7 determinants. With 8 derivatives we have again 56 possible
solutions. Their spread is drastically reduced and the mean value is closer to
the exact one, as ﬁg. 1 shows. We anticipate that all the solutions considered
in the next section have a microscopic spread (see e.g. ﬁg. 3 and ﬁg. 5). 3 The
boundary bootstrap and the 3d Ising and O(N) models. In this Section we shall
consider the boundary conformal ﬁeld theories (BCFTs) associated with the Ising
model and other magnetic systems. Speciﬁcally, the IR properties of the surface
8

---
**페이지 10**
---

0.7 0.8 0.9 1.0 1.1 1.2 Ξ 1.00000 1.00002 1.00004 1.00006 Truncations Figure 2.
The left-hand-side of the sum rule (2.9) for various truncations (nbulk, nbdy,
0) of the two- point function of the 2d Ising model in the ordinary transition.
Only in the nbulk , nbdy limit the sum rule is saturated. transitions in these
systems are controlled by RG ﬁxed points, which of course are described by just
as many Defect CFTs. We denote with σ(x) the scalar ﬁeld (i.e. the order
parameter of the theory) and with bσ the corresponding surface operator. The
surface Hamiltonian associated with a ﬂat d 1 dimensional boundary of a
semi-inﬁnite system can be written in terms of the three relevant surface
operators (see for instance [37]) H = Z dd1x cbσ2 + h1bσ + h2zbσ . (3.1) Here z
xd is the coordinate orthogonal to the boundary. This Hamiltonian has three ﬁxed
points O : h1 = h2 = 0, c = +; (3.2) E : h1 = h2 = 0, c = ; (3.3) S : h1 = h2 =
c = 0 . (3.4) Near the ﬁrst ﬁxed point the conﬁgurations with bσ = 0 are
exponentially suppressed, then bσ = 0 (i.e. Dirichlet boundary condition). This
ﬁxed point controls the ordinary transition. The only relevant surface operator
in this phase is zbσ. The ﬁxed point with c = favors the conﬁgurations with bσ =
0: it is associated with the extraordinary transition, where the Z2 symmetry is
broken and no relevant surface operator can couple with it; the lowest
dimensional surface operator, besides the identity, is the displacement, whose
scaling dimension is d. The ﬁxed point with c = 0 controls the special
transition, a multicritical phase with two relevant primaries. The even operator
bσ2 is responsible for the ﬂow of c to or according to the initial sign, while
the odd one, bσ, is the symmetry breaking operator 9

---
**페이지 11**
---

of this phase, characterized by the Neumann boundary condition zbσ = 0. We
omitted a classically marginal coupling, zbσ2, because it vanishes with both
Neumann and Dirichlet boundary conditions, and it cannot be turned on in the
extraordinary transition, where there is no local odd relevant excitation. We
shall come back to this operator when considering the RG domain wall. One
important question to address within a BCFT is how to ﬁnd the scale dimensions
of the surface operators and their OPE coeﬃcients in terms of the bulk data.
This problem has been completely solved in 2d [38] thanks to the modular
invariance. In d 2 useful information can be extracted by the epsilon expansion
and other perturbative methods. Recently the conformal bootstrap approach has
been shown to be very promising [14]. Here we face this problem with the method
of determinants. We study the 2-point function σ(x)σ(y). The general criterion
we use to classify the surface transition associated with a speciﬁc truncation
(nbulk, nbdy, s) of the crossing symmetry equations (2.10) is based on three
steps. First, we verify that the solution is compatible with a unitary theory by
requiring the positivity of all the non-vanishing couplings µ2 a (a = 1, 2, . .
. , nbdy). Then we look at the sign of the couplings to the bulk blocks akλσσk
(k = 1, . . . , nbulk). As in [14], we will assume that the ordinary transition
is signaled by the presence of at least one negative coupling in the bulk
channel. On the other hand, positivity of the couplings indicates the
extraordinary or the special transition, depending on the presence or absence of
the surface identity. We should point out that these assumptions have not been
proven. However, the results of this work seem to conﬁrm them, serving as a
consistency check on the whole setup. 3.1 The ordinary transition. We start by
considering what is perhaps the simplest successful truncation of eq. (2.10),
corresponding to the fusion rules σ σ 1 + ε + ε, bulk channel, σ bO, boundary
channel. (3.5) This truncation is denoted by the triple (2,1,0). The system
(2.10) admits a solution if and only if the 3 3 determinants made with the
derivatives of the conformal blocks associated with ε, ε, bO vanish. We assume
that the scale dimensions of σ, ε and ε are known (σ = 1 2 + η 2; ε = 3 1/ν; ε =
3 + ω, see table 1) and in this particular case the only unknown scale dimension
is b O. Fig. 3 shows the values of few determinants of this kind. Clearly they
all apparently vanish at the same point. In fact there is a microscopic spread
of the solutions and we ﬁnd b O = 1.276(2). The solution of the complete linear
system yields a negative aελσσε, thus, according to the above criterion, we are
faced with the ordinary transition of the 3d Ising model. Hence, bO has to be
identiﬁed with zbσ. A two-loop calculation in the 3d φ4 model yields [39] zbσ
1.26 in good agreement with our result. This solution admits a straightforward
generalization to any 3d O(N) model by simply replacing the critical indices
with the appropriate values. Table 1 shows our results for N = 0 10

---
**페이지 12**
---

N η ν ω 0 0.0314(32) 0.5874(2) 0.812(16) 1 0.03627(10) 0.63002(10) 0.832(6) 2
0.0380(4) 0.67155(27) 0.789(11) 3 0.0364(6) 0.7112(5) 0.782(13) zbσ N 2-loop
Monte Carlo Bootstrap 0 1.33 1.332(6) 1 1.26 1.2751(6) 1.276(2) 2 1.211 1.219(2)
1.2342(9) 3 1.169 1.187(2) 1.198(1) N aελσσε aελσσε µ2 b 0 0.8447(34) 0.0366(17)
0.692(1) 1 0.789(3) 0.042(1) 0.755(13) 2 0.747(1) 0.0488(4) 0.80022(5) 3
0.710(1) 0.0509(6) 0.8395(6) Table 1. The ﬁrst table collects the input
parameters. The second one is a comparison between two-loop calculations [39],
Monte Carlo simulations (reference [40] for N = 1 and reference [41] for N 1)
and our bootstrap results for the scaling dimension of the surface operator zbσ
in the ordinary transition of 3d O(N) models. The last three columns collect our
results for the OPE coeﬃcients. The critical indices η and ν for N = 0, 1, 2, 3
are taken respectively from references [42], [43], [44] and [45]. Those for ω
from [46]. 1.1 1.2 1.3 1.4 D -0.15 -0.10 -0.05 0.05 0.10 0.15 Det Figure 3. Plot
of the 10 3 3 minors made with the ﬁrst 5 derivatives of the conformal blocks
associated with ε, ε and bO as functions of b O. They all vanish approximately
at he same point, selecting the allowed value of b O. (the non-unitary
self-avoiding walk model), N = 1 (Ising), N = 2 (XY model) and N = 3 (Heisenberg
model), where we can compare our results with the two-loop calculation of [39].
11

---
**페이지 13**
---

## 3.2 The extraordinary transition. Such a transition is characterized by the non-vanishing contribution of the boundary identity to the two-point functions of Z2 odd operators. In this case the boundary surface is in an ordered phase, therefore the degrees of freedom described by Z2 odd operators are frozen. The ﬁrst non-vanishing surface operator, besides the identity, is the displacement D with D = 3. As a consequence, the most relevant contribution to the boundary channel is known and the crossing equations can be exploited to obtain information on the bulk channel. Actually adding the boundary identity to the truncation requires adding more bulk op- erators as well. We found a ﬁrst stable solution of the type (4,1,1). This time the scaling dimensions of the two needed bulk scalars ε and ε cannot be used as input parameters because, once ﬁxed σ, ε and ε2, we get a solution only if N = 1 , ε = 7.316(14) , ε = 13.05(4)3. (3.6) The other parameters of the solution are aελσσε = 6.914(6), aελσσε = 2.261(2), aελσσε = 0.187(1), aελσσε = 0.0046(1), a2 σ = 6.757(4) , µ2 σD/CD = 0.06282(3); (3.7) where we denoted with CD the Zamolodchikov norm of the displacement operator. In this case, we probed the stability of the solution by adding a new conformal block in the boundary channel. It turns out that the truncation (4,2,1) deﬁnes a one-dimensional family of the solutions, where the free parameter is the dimension of the added surface operator, which can vary in the range 0  b. In the limit bwe recover, as expected in a stable solution, the truncation (4,1,1). The dimensions of the two bulk operators ε and ε vary as functions of bin a narrow range: the net eﬀect of the unknown parameter is to reduce a bit the scaling dimensions of these bulk operators. Eliminating bwe obtain the plot in ﬁg. 4. The uncertainty on the actual value of bforces us to enlarge the errors in the bulk dimensions. Fig. 4 roughly suggests ε = 7.27[5] , ε = 12.90[15], (3.8) which supersede eq. (3.6). We used square brackets to indicate that this is not a statistical error, but a sum of the uncertainties. Unfortunately one can ﬁnd in literature a wide range of proposed values for ε and ε which strongly depend on the method employed (see for instance table 3 of [19]). What is especially disturbing for us is that the method of determinants applied to the four-point function gave very diﬀerent values for these quantities [20], so we decided to reanalyze the 2Here and in the rest of this section we use as input parameters of the Ising model the values σ = 0.518154(15), ε = 1.41267(13) and ε = 3.8303(18) taken from [19]. 3In the entire paper the estimate of the statistical error due to the uncertainty on the input parameters is obtained by means of a statistical bootstrapping procedure.  12

---
**페이지 14**
---

1 1 1 1 1 1 1 11 1 1 1 1 1 1 1 1 1 11 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 11 1 1 1
1 1 1 11 111 7.22 7.24 7.26 7.28 7.30 7.32 7.34 DΕ 12.7 12.8 12.9 13.0 DΕ Figure
4. Parametric plot of the scaling dimensions of ε and ε generated by the unknown
parameter bin the (4,2,1) truncation. Here we see the eﬀect of the statistical
errors on the input data, namely σ, ε and ε as well as the eﬀect of the spread
of the solutions. Some of these data are presented in table 3. bootstrap
equations for the four-point function on the bulk in order to see whether there
is also a solution compatible with the spectrum suggested by the boundary
bootstrap. Out of this study we can conﬁrm the existence of a scalar of
dimension 7.2 with a positive coupling. We were unable to ﬁnd a proper solution
for the scalar at 13, all solutions being characterized by a coupling that is
very small, negative and nearly always compatible with zero. The quoted
dimensions of these two scalars found with the linear functional method [19] are
respectively 7 and 10.5. Another interesting two-point function to be studied in
the extraordinary transition of the Ising model is the spin-energy correlator
σ(x)ε(y)which is diﬀerent form zero only in this phase, being the only surface
transition where the Z2 symmetry of the model is broken. The fusion rule of the
bulk sector contains odd operators only: σ ε σ + σ + σ + . . . , (3.9) while in
the boundary sector the ﬁrst primary operator contributing, besides the
identity, is the displacement operator: σ 1 + D + . . . , ε 1 + D + . . . (3.10)
The ﬁrst stable solution corresponds to the truncation (3,1,1) deﬁned by the
above fusion rules. It is associated with the (apparently) common intersection
of the zeros of the 5 5 determinants made with the derivatives of the 5
conformal blocks involved (see ﬁg. 5): 13

---
**페이지 15**
---

4.5 5.0 5.5 6.0 6.5 8 9 10 11 12 13 DΣ DΣ Figure 5. Plot of the zeros of some 5
5 determinants associated with the fusion rules (3.9) and (3.10). σ 5.66 ; σ
10.89 ; (3.11) aσλσεσ 0.148 κ ; aεaσ 0.927 κ ; µσDµεD/CD 0.0196 κ . (3.12) The
parameter κ arises because now the bootstrap equations are homogeneous, that is,
they do not contain the information about the normalization of the external
operators. The nor- malization of the order parameter is contained in the
correlator σσ, while the normalization of the energy follows from assuming
symmetry of the OPE coeﬃcient λσσε = λσεσ. There- fore, combining (3.12) with
the analogous couplings in (3.7), we can compute the unknowns aε, aσ, µσD/CD,
µεD/CD, κ, λσεσ. In order to probe the stability of the solution and to evaluate
the errors we upgraded the solution to (5,1,1), which corresponds to a
one-parameter family of solutions. We used as a free parameter the heaviest bulk
scalar σ4. A solution exists for 18 σ4 28. As expected for a stable solution,
this parameter has no visible eﬀect on the OPE coeﬃcients and only slightly
aﬀects the scale dimensions of the two scalar σ and σ. The results of this
analysis can be found in table 2 It turns out that σ is nicely close to the
bound σ 5.41(1) found in [24]. Notice also that the resulting OPE coeﬃcient λσσε
is in perfect agreement with the estimate of a recent 14

---
**페이지 16**
---

ε ε σ σ σ λσσε 7.27[5] 12.90[15] 5.49(1) 10.6[3] 16[1] 1.046(1) aε µεD/CD aσ
µσD/CD 6.607(7) 1.742(6) 2.599(1) 0.25064(6) Table 2. The main results of the
combined analysis of σσand σεin the extraordinary transition are split in two
parts. The top table refers to data of the bulk channel, while the bottom table
contains OPE coeﬃcients speciﬁc to the boundary channel of the extraordinary
transition. Errors in square brackets refer to data whose uncertainties depend
on an unknown parameter; the other errors simply reﬂect the statistical errors
of the input data, namely, σ, ε and ε. Monte Carlo calculation [48] which gives
λσσε = 1.07(3) and the value (λσσε)2 = 1.10636(9) found in [19] through the
study of the four-point function with the linear functional method. There is
another very impressive check of these results. The Ward identity associated
with the displacement operator tells us that the quantity xO = O aO µOD CD does
not depend on the speciﬁc bulk operator O but only on the surface transition, as
described in section 4. The above results yield xσ = 5.3727(27) ; xε = 5.358(15)
, (3.13) showing, within the errors, a reassuring fulﬁllment of the Ward
identities. Note added, November 2021 A previous version of this paper contained
results about the extraordinary transition for N 1. However, the ﬁrst operator
in the boundary channel was incorrectly assumed to be the displacement. Instead,
a protected boundary operator of dimension b= 2 arises from the breaking of the
continuous O(N) symmetry. We refer to [63] for a conformal bootstrap study of
this boundary condition for N 1. 3.3 The special transition. According to our
discussion at the beginning of this section, solutions ascribed to the special
transition are associated with truncations of the form (m, n, 0) in which all
the OPE coeﬃcients are non-negative. By consistency with the results of the
previous subsection we have to use the same bulk spectrum determined in the
extraordinary transition. We found solutions of the form (3,3,0) and (4,3,0)
with similar properties. Here we only discuss the latter. Instead of an isolated
solution, in this case we ﬁnd a one-parameter family in the three- dimensional
space of the boundary scale dimensions (b1 b2 b3). The lowest-dimensional
operator has to be identiﬁed with bσ and according with the two-loop calculation
of [39] we expect bbσ 0.42. In our case a unitary solution exists only for 0.34
b1 0.45. Below 0.34 the solution disappears abruptly; above 0.45 it becomes
non-unitary. Using b3 as a free parameter, we obtain the plot of ﬁg. 6, which is
superimposed to the unitarity upper bound found in [14]. As expected, the
transition to the non-unitary region 15

---
**페이지 17**
---

0.40 0.45 0.50 D1 1.5 2.0 2.5 D2 Figure 6. Plot of the one-parameter family of
the truncation (4,3,0) in the plane (b1, b2), superim- posed to the upper
unitarity bound found in[14]. The blue and green dots correspond respectively to
the minimal and the maximal choice of the pair (ε, ε), as determined in ﬁg. 4.
These dots are replaced by ones respectively magenta and yellow when some OPE
coeﬃcient become negative. For the black dots on the unitarity bound see
explanation in the text. coincides with the unitarity boundary found by the
linear functional method. Consistency requires that the spectrum of our solution
at the intersection should agree with the one extracted from the zeros of the
linear functional[16] calculated at the same point. In fact, the ﬁrst zero of
the linear functional at the intersection point, in the bulk sector, is (see ﬁg.
7) around 6.7, which is consistent with our result for ε. Similarly, the zero of
the extremal functional for the boundary sector (besides b1 and b2) is perfectly
consistent with the value b3 4.44 at the crossing point. 2 4 6 8 DΕ² L 3 4 5 6 D
ï 3 L Figure 7. Linear functionals for the bulk and boundary channels in the
special transition. Such a boundary required by unitarity could also be seen as
the locus were one or more OPE coeﬃcients change sign. Our solution leads us to
conjecture that the couplings vanishing at the unitarity bound are λσσε and
λσσε. In the construction of the upper unitarity bound in [14] it is assumed
that the ﬁrst bulk primary is the Ising energy ε and it follows that the 16

---
**페이지 18**
---

subsequent primary has scale dimension larger than ε, as suggested by our
conjecture. The knowledge of the linear functional leading to the bound of ﬁg. 6
suggests another interesting cross-check of the two methods: given a value of b1
we insert in the (4,3,0) truncation the ﬁrst four zeros of the linear functional
on the bulk channel and evaluate with the method of determinants the
corresponding boundary values b2 and b3. It turns out that in the plane b1, b2
such a solution lies on the unitarity bound, as consistency requires (see black
dots in ﬁg. 6). 4 Renormalization group domain wall for the O(N) model. Before
starting the exploration of a speciﬁc conformal interface, let us recall the
relevant CFT data that one needs to collect in order to completely describe the
generic system. Conformal interfaces are closely related to boundaries. In fact,
as we mentioned in section 2, an interface between a CFT1 and a CFT2 can be
mapped to a boundary problem using the folding trick. One turns the original
setup into a boundary for the theory CFT1 CFT2, where the bar means that a
reﬂection xd xd has been applied to one of the theories. We see that the natural
bulk CFT data is given by the value of the two point functions of operators
placed in mirroring points with respect to the interface: they are mapped to
expectation values of operators in the folded CFT. This also identiﬁes the
needed operators as primaries of the folded theory, which in particular include
all bulk primaries of the two CFTs. The latter are not suﬃcient, though, because
they do not play any role as building blocks of correlators across the
interface. Another way of understanding this circumstance is provided by the
north-south pole quantization, or equivalently by conformally mapping the theory
to a d-dimensional sphere. Local operators at the north or south pole create a
state belonging to the Hilbert space of either CFT. The interface is a linear
map between the Hilbert spaces, and the correlators of operators placed in
mirroring points - that is, at the north and south poles - are the matrix
elements of this map. Analogous considerations are valid for the bulk-to-defect
couplings. Let us now turn to the speciﬁc interface we shall study in this
paper. The Renormalization group domain walls are interfaces between two CFTs
which lie at the top and at the bottom of an RG ﬂow. More precisely, there is an
easy operational deﬁnition: start with a CFT on the whole space, and modify the
action by integrating a relevant operator over half of the space. Far away in
this region, the long distance physics will be dominated by the CFT at the
bottom of the ﬂow triggered by the perturbation. This deﬁnition can be employed
literally when the coupling is only mildly relevant, and perturbation theory
makes sense. In order to single out a unique gluing condition, it is also
necessary to specify which defect deformations are turned on along with the bulk
ﬂow. In the case of interest for us, we shall argue that no marginal
deformations exist on the defect, and so we just choose to ﬁne tune
perturbatively the relevant defect couplings. As usual, near the interface the
critical behaviour is modiﬁed with respect to both the UV and the IR homogeneous
ﬁxed points, with new critical exponents arising. RG domain walls have been
mainly studied in two dimensions [4953]. In a general non perturbative setting,
the determination of the defect spectrum and 17

---
**페이지 19**
---

the computation of correlators is a very diﬃcult task. In some limiting cases,
however, some of the answers might be found with little eﬀort. For instance, a
relevant operator may force the bulk to ﬂow towards a trivial theory. In this
case, the RG interface is reduced to a boundary condition for the ultraviolet
CFT. As an example, consider giving a mass to a free boson on half of the space,
in any dimension greater than two. Correlators on the perturbed side are
exponentially damped, and at large distances the theory is empty. From an RG
point of view, the coupling grows in the IR, and the conﬁgurations of non-zero
ﬁeld on the perturbed side are suppressed in the partition function. As a
consequence, a Dirichlet boundary condition is imposed to the massless free
boson on the other side. A more interesting case is the RG domain wall
corresponding to the Wilson-Fisher ﬁxed point of the O(N) model with (φ2)2
interaction. This interface is captured by the following bare action: S = Z ddx
1 2Sd(d 2) µφi µφi + θ(xd) g 4!(φiφi) 2, (4.1) where θ(xd) is the Heaviside
function, Sd = 2πd/2/ Γ(d/2) and we chose to normalize the elementary ﬁeld so
that it has a canonical two-point function in free theory. As we pointed out, a
question that needs to be answered concerns the stability of this interface. One
needs to know how many relevant operators must be ﬁne-tuned, and if marginal
deformations exist. The interface possesses a weakly coupled description in 4 ϵ
dimensions, and, at the classical level, the only relevant defect primary in the
singlet sector is bφ2. Once we tune it to zero, unlike the situation in the
special transition, we do not impose Neumann boundary conditions, but only
continuity of z bφi on the interface. Hence, the classically marginal operator z
bφ2 does not vanish, and should be taken into account. We shall show that this
operator becomes irrelevant at one loop. Therefore, the RG interface appears to
be isolated in perturbation theory. In the following, we characterize the
correlations of scalar primaries in the presence of the domain wall at lowest
order in ϵ-expansion. Along the way, we point out that correlations across the
interface encode at this order the mixing induced by the RG ﬂow among nearly
degenerate operators [51]. This is true in the larger class of perturbation
interfaces constructed by means of a nearly marginal deformation. We then focus
on the RG domain wall between the three dimensional free theory and the Ising
model, and study the two-point function of the ﬁeld σ using the method of
determinants. We also provide some non-perturbative information on generic
conformal interfaces involving the free theory, by noticing that some of the
crossing constraints can be solved analytically. 4.1 The ϵ-expansion and the
role of the displacement operator. Since the UV side of this RG interface is a
free theory, the interface itself is not captured by mean-ﬁeld theory: the CFT
data related to it is O(ϵ) in perturbation theory. One can easily obtain general
results at leading order by exploiting the Ward identity eq. (2.1), which deﬁnes
the displacement operator. The identity tells us that we can move the interface
in the orthogonal direction by integrating the displacement in the action. Its
insertion in a 18

---
**페이지 20**
---

correlation function is therefore equivalent to a derivative with respect to the
position of the interface, that is, Z dd1y D(ya) O1(x1) . . . On(xn)= n X i=1 xd
i O1(x1) . . . On(xn). (4.2) Since the violation of translational invariance
happens at order g - see eq. (4.7) - the relation (4.2) rephrases some
information about an n-point function of order gL in terms of the integral of a
(n + 1)- point function of order gL1. In general, knowledge of the variation
with respect to the position of the interface is obviously insuﬃcient for
reconstructing the full correlator. However, all conﬁgurations of two points are
conformally equivalent to the one in which the points are aligned on a line
perpendicular to the defect. Therefore a two-point function can be traded for
the integrated three-point function on the l.h.s. of eq. (4.2). The advantage is
that the integral does not generate additional divergences: one only needs to
renormalize the theory at order gL1. On the other hand, it is still necessary to
determine a primitive of the l.h.s. of eq. (4.2) as a function of the position
of the interface. We shall see that this is possible at lowest order: the tree
level 2-point correlator, which is just the homogeneous one, can be used to
compute the one loop correction in the presence of the interface. It is simple
to derive from (4.2) a new scaling relation. As pointed out, when two operators
are placed in mirroring points, in which case ξ = 1, their correlator is
equivalent, through the folding trick, to a one-point function: OL(x) OR(Rx)= aL
R |2xd|L+R , Rx = xa, xd . (4.3) Here we think of OL and OR as scalars belonging
respectively to the UV and IR spectrum. Similarly, the three-point function OL
OR Dis ﬁxed up to a number: OL(x) OR(Rx) D(ya)= µL R D |2xd|L+Rd |x y|2d (4.4)
Using the fact that in this geometry ξ is stationary with respect to orthogonal
displacements of the interface, it is easy to derive the following relation
between these pieces of CFT data (R L)aL R Sd = µL R D. (4.5) In the particular
case where one of the bulk operators is the identity, one recovers a relation
which was ﬁrst noticed in the case of a boundary by Cardy [54] (see also [33]):
kak Sd = µkD, (4.6) where the plus/minus sign is valid for the interacting/free
side respectively. We start by using eq. (4.6) to determine the aks. The answer
at order ϵ is quite simple: only one operator acquires expectation value, on
both sides of the interface. To see this, let us identify the 19

---
**페이지 21**
---

displacement. Looking at the action (4.1), we see that the interface is
displaced at leading order by integrating the bare operator g(φ2)2/4!, that is4
D = g 4!(φ2)2 + O(g2) = 1 8(N + 8)π2 ϵ (φ2)2 + O(ϵ2), (4.7) where we plugged the
ﬁxed point value of the coupling at order ϵ: g= 3 (N + 8)π2 ϵ. (4.8) Now, since
(φ2)2 is a primary of the free theory, and no other primary mixes with it at
order one, its correlation function with any other primary is zero at leading
order. This means that all coeﬃcients µOD = O(ϵ2), but for the case O = (φ2)2.
Using the relation (4.6), we conclude that the only non vanishing expectation
value at this order is (φ2)2. We can then obtain the number aφ4 at order ϵ from
a tree level computation. Indeed, the relevant bulk-to-defect coupling is given
at leading order by µφ4D = |x|8 (φ2)2(x) p 8N(N + 2) D(0)= p 2N(N + 2) 4(N +
8)π2 ϵ. (4.9) Therefore aIR φ4 = aUV φ4 = p 2N(N + 2) 8(N + 8) ϵ. (4.10) Let us
make a comment. It was obvious from the start that only a small class of
operators could exhibit a one-point function at ﬁrst order in the coupling: four
powers of the elementary ﬁeld are needed to contract a single vertex, and of
course the operator must be in the singlet of O(N). However, inﬁnitely many
scalar primaries can be constructed in free theory which fulﬁll these
requirements, involving an increasing number of derivatives of the ﬁelds5. The
simplest use of eq. (4.10) is the determination of the most general two-point
function of operators lying on the same side of the interface at order ϵ.
Sticking for simplicity to the case of external scalars, one simply writes
O1(x)O2(x) = 1 (2xd)1(2x d)2 ξ(1+2)/2 δ12 + λ12φ4aφ4fd=4 bulk(12, = 4, ξ) +
O(ϵ2). (4.11) 4Notice that at higher orders the interacting stress-tensor needs
to be improved to be kept ﬁnite and traceless [55]. The improvement is
proportional to (µν δµν2)φ2, so that the displacement receives a contribution
from the operator aaφ2. 5That these primaries must exist can be seen
independently from their expression in terms of elementary ﬁelds, for instance
from the asymptotics of the two point function of φ2 in a free theory with a
boundary. The presence of the identity in the boundary channel can only be
balanced by an inﬁnite number of conformal blocks in the bulk channel. Only one
primary can be built with two powers of the ﬁelds, so the rest are the ones we
are interested in. The explicit conformal block decomposition for this case can
be found in [14]. It is also amusing to notice that, analogously to the case at
hand, this tower of operators does not contribute at order ϵ to the two-point
function of φ with Dirichlet or Neumann boundary conditions. As noticed in [14],
in that case the OPE coeﬃcients λφφ 2kφ4 are the vanishing quantities at order
ϵ. 20

---
**페이지 22**
---

Notice that λ12φ4 is guaranteed to belong to the 4d free theory only when O1 and
O2 are on the UV side. Indeed, primaries on the interacting side are in general
a mixture of classi- cally degenerate renormalized operator, and when the mixing
happens at leading order λ12φ4 becomes a linear combination of UV OPE
coeﬃcients. For completeness, we compare this derivation with some direct one
loop computations in appendix A. As pointed out in the introduction to this
section, in order to capture correlations across the interface we would need all
the one-point functions of the folded theory. This set encom- passes the aL R
deﬁned in (4.3), and is much bigger. It is in fact more viable to reach for the
two-point functions of primaries directly through the integrated Ward identity
eq. (4.2), speciﬁed to the case of interest: Z dd1y OL(x) OR(x)D(y)= xd + xd
OL(x) OR(x). (4.12) We pick for the left hand side the three-point function of
primaries in the translational invariant theory, and we get the one-loop
two-point function by integrating over the position of the displacement. Notice
that in doing so we disregard the mixing of primaries with descendants. In the
cases in which this happens at order one, on the left hand side of eq. (4.12)
additional terms needs to be taken into account, which have the form of a
three-point function involving derivatives of a primary operator. Consider ﬁrst
two operators which are degenerate in the free theory. In other words, LR L R =
O(ϵ). (4.13) In this case eq. (4.12) can only be used to determine the one loop
correlator up to a constant. Indeed, since both µL R D and L R are of order ϵ,
one needs the one loop three-point function to determine aL R from eq. (4.5).
This is the familiar eﬀect of degeneracies in perturbative computations, and is
related to the mixing of operators along the RG ﬂow (see section 4.2).
Integration of (4.12) is straightforward, and one gets OL(x) OR(x)= aL R
|2xd|L(2xd)R (ξ)L 1 + LR 2 log(ξ) , LR = O(ϵ). (4.14) Comparing with the form
(2.5) we can write at this order GL R(ξ) = aL R LR = O(ϵ). (4.15) A comment is
in order. The presence of a logarithmic singularity compatible with exponen-
tiation is somewhat natural, since turning the coupling oﬀone recovers the short
distance power low divergence proper of the homogeneous theory. However, there
is no reason for this to happen when considering the OPE limits in the Euclidean
defect CFT. The exponentiation agrees in the large ξ limit with the defect OPE,
as it is easy to verify using the formulae given in subsection 4.2. On the other
hand, no small ξ limit exists for primaries on opposite sides of the domain
wall, and in fact the folded cross-ratio is ξfolded = (1 + ξ), which vanishes
when the operators are placed in mirroring points. We decide to keep using the
form (2.5), 21

---
**페이지 23**
---

and notice that it might be fruitful to look for a justiﬁcation in Lorentzian
signature, where the small ξ limit corresponds to light-like separated
operators. In the case of operators with dimension diﬀering in the UV limit, the
two-point functions at one loop can be ﬁxed completely. Due to O(N) and
rotational symmetry, LR is an even integer in d = 4, which provides a
simpliﬁcation. The computation is slightly more involved than in a previous
case, and we give some details in appendix A. The result in the case |LR| = 2 is
diﬀerent from all the others: GL R(ξ) = π2 2 µL R D sign(LR) (ξ 1), |LR| = 2,
(4.16) while GL R(ξ) = π2Γ(2k + 3) (k 1)k2Γ(k + 2)2 µL R D sign(LR) (ξ)k+1 (4k +
2)ξ2 + 3(k + 1)ξ + 1 2F1 k 1, k, 2(k + 1); 1 ξ (4k + 2)ξ2 + (k + 2)ξ 2F1 k 1, k
1, 2(k + 1); 1 ξ , |LR| 2k 2. (4.17) As one might have expected, the
hypergeometric functions in eq. (4.17) are in fact polynomials. These results
complete the analysis of bulk correlations at order ϵ, if knowledge of the λ123
is assumed: n-point functions of bulk operators are determined by taking
successive OPEs on the two sides until one is left with a one-point function or
a two-point function across the interface. We shall content ourselves of this
leading order solution, but we would like to comment on the possibility of
generalizing the procedure. Unfortunately, the number of non vanishing one-point
functions is inﬁnite already at next to leading order6. Therefore, once the
displacement has been correctly normalized, one has to compute the relevant
three-point functions at one loop and integrate them to ﬁnd the two loop
two-point functions. Let us now consider the defect spectrum at order ϵ. The
dimensions of the operators can be extracted through the defect OPE
decomposition of eq. (4.11). When nearly degenerate operators are present in the
UV theory, also the defect operators mix, and the spectrum is given by the
eigenvalues of the matrix of anomalous dimensions. We shall deal with this more
general case in the next subsection. Here we comment on some features of the
spectrum focusing for simplicity on the non-mixing operators. The lightest
defect scalar in the OPE of a bulk operator O has dimension bO = UV O 2λOOφ4 aUV
φ4 + O(ϵ2) = 1 2(UV O + IR O ) + O(ϵ2). (4.18) 6This statement again follows
immediately from the fact that the operator φ2 acquires an expectation value at
order ϵ2. 22

---
**페이지 24**
---

The second equality in eq. (4.18), which agrees with ﬁrst order conformal
perturbation the- ory, says that the defect primary stands half way between the
corresponding infrared and ultraviolet operators in the bulk. Let us make some
more speciﬁc comments. bφ4 = 4 ϵ is the protected dimension of the displacement
operator. This is expected, even if there are degenerate operators in free
theory. Two primaries exist with dimension near to four, but both of them are
protected, the second one being the displacement of the folded theory. The
second interesting scale dimension is obtained by going one step further in the
defect OPE of φ2. We encounter the operator z bφ2, and since no other scalars
exist which could mix with it, we can safely read oﬀhis dimension from the
boundary block decomposition: bφ2 = 3 N+14 2(N+8)ϵ. We see that this scalar is
irrelevant at the Wilson-Fisher ﬁxed point, so that the stability of the
interface is not altered by its presence. A third remark concerns the odd
spectrum. Since the anomalous dimension of φi starts at two loops, or
equivalently the bulk OPE does not contain (φ2)2 on either side of the
interface, the dimensions of bφi and z bφi remain classical. Moreover, at this
order all ﬁelds of the kind k z bφi can be converted to descendants of bφi and z
bφi by means of the tree level equations of motion. Hence, the latters are the
only primaries appearing with an OPE coeﬃcient of order one. The interesting
fact is that bφ and bφ do not receive loop corrections at all, as we review in
subsection 4.3. A last comment on the one-loop odd spectrum is in order. The
two-point function of φ2φi should obey eq. (4.11) only on the free side, where
the operator is a primary. This two-point function contains a tower of defect
operators which we might identify with d φ2φi and its transverse derivatives.
The dimension of d φ2φi is consistently half-way between φ2φi and its image
under RG ﬂow, that is, φi, and turns out to be marginal at this order. Since we
could not devise a mecha- nism to protect this operator from quantum
corrections, we believe this feature will disappear from the spectrum at higher
orders. The fact that d φ2φi is independent from the conformal families of bφi
and 2 z bφi is naturally justiﬁed by deﬁning the defect ﬁelds as the limit of
the free bulk ﬁelds approaching the interface. Notice that this happens
automatically in a hard-core regularization, where all integrals are cut-oﬀat a
small distance from the interface. The considerations leading to eq. (4.11)
apply in fact to the leading order in conformal perturbation theory of any
interface obtained by a nearly marginal bulk perturbation. Indeed, the key point
is that the Zamolodchikov norm of the displacement operator equals the square of
the coupling at leading order. We turn now to this more general setting in order
to discuss the leading order mixing of bulk and defect primaries. On the
contrary, notice that eqs. (4.16) and (4.17) do not generalize trivially,
because we used the fact that UV scale dimensions are (nearly) even-integer
separated: formulae get a bit more messy in the general case. 4.2 Leading order
mixing of primary operators. Consider a conformal ﬁeld theory in any number of
dimensions d, whose spectrum includes one7 mildly relevant operator ϕ, that is ϵ
= d ϕ is a small positive number. The interface 7We consider for simplicity the
case of a one parameter RG ﬂow. The general case proceeds along the same lines.
23

---
**페이지 25**
---

constructed by integrating gϕ on one half of the space has an infrared ﬁxed
point in which g = gO(ϵ). The two-point functions of operators on the same side
of the interface obey the obvious generalization of eq. (4.11): O1(x)O2(x) = 1
(2xd)1(2x d)2 ξ(1+2)/2 δ12 + λ12ϕaϕfd bulk(12, = d, ξ) + O(ϵ2). (4.19) Here aϕ
is of order ϵ and at this order aIR = aUV = gSd d , (4.20) as dictated by eq.
(4.6). We would like to study the eﬀect of the mixing of bulk primaries on the
defect operators. Let us choose a set of UV scalar primaries OUV i which are
degenerate up to terms of order ϵ. Their defect OPE, restricted to the lowest
lying primaries, is encoded in the fusion rule OUV i µij bOj + . . . (4.21)
These defect operators are connected by the RG ﬂow to the UV operators
themselves, that is there exists a family of renormalized operators bOi(g) such
that OUV i = bOi(0) and bOi = bPij bOj(g). The matrix bP ij depends on the
deﬁnition of the renormalized operators, that is on the regu- larization scheme.
However, in what follows we shall only need the fact that bP ij is orthogonal at
order one. Comparing with eq. (4.21) we see that µij = bP ji + O(ϵ). (4.22) The
relevant part of the defect OPE decomposition of the correlator OUV i OUV j is
determined by the following asymptotic behavior for large ξ: fd bulk(12 = 0, =
d, ξ) d 2 log ξ + γ ψ(d/2) + O(ξ1). (4.23) Comparing this with the large ξ and
small ϵ limit of the boundary blocks, we get X k µikµjk bk UV i + UV j 2 ! = d 2
λUV ijϕ aUV ϕ . (4.24) Since the quantity in parenthesis is of order ϵ, we can
make the substitution µ bP. The latter matrix was deﬁned to be the orthonormal
change of basis which diagonalizes the matrix of anomalous dimensions bγij of
the boundary operators bOj(g), so that we get bγij = UV i δij d 2 λUV ijϕ aUV ϕ
= UV i δij + Sd 2 λUV ijϕ g. (4.25) One may proceed order by order in the large
ξ expansion. The resulting defect spectrum includes in general nearly degenerate
scalars with dimension close to + k, being the scale 24

---
**페이지 26**
---

dimension of a bulk primary. A primary of level k of course originates from
linear combinations of transverse and parallel derivatives of a UV primary. But
when nearly integer separated bulk primaries exist, further mixing is expected
to take place. To complete the analysis, we would like to show that by matching
the defect spectrum with the IR bulk primaries, we get back the known mixing
matrix between UV and IR operators of the homogeneous theory [56]. We restrict
ourselves to the case in which the mixing only involves primary operators. We
consider the set of IR primaries OIR i which are related to the OUV i through a
matrix P ij whose deﬁnition is analogous to the one we gave for bP. The leading
part of the defect fusion rule is OIR i νij bOj + . . . (4.26) where we required
that the defect spectrum coincides with the one of the UV counterparts. This
time we have νij = Pik bP jk + O(ϵ). (4.27) The same steps as before now lead to
a relation identical to eq. (4.24), up to the substitutions µ ν and UV IR. The
combination of eqs. (4.20), (4.25), (4.27) with the statement λIR ijϕ =
PimPjnλUV mnϕ + O(ϵ), (4.28) leads to IR i δij = PimPjn UV m δmn + Sd λUV mnϕ g
. (4.29) Since the matrix P diagonalizes by hypothesis the matrix of bulk
anomalous dimensions, we recover the formula γij = UV i δij + Sd λUV ijϕ g.
(4.30) Notice that the anomalous part of the defect mixing matrix is one half of
the bulk one. As a last comment, by means of eq. (4.5), we can verify that the
pairing of UV and IR primaries matches the matrix P at leading order [51]: aji =
Pij + O(ϵ). (4.31) Indeed, eq. (4.31) is immediately obtained starting from the
equality (IR i UV j )aji = Sd PikλUV jkϕ g, (4.32) which is valid at leading
order, and using the deﬁnition (4.30) of the mixing matrix. 4.3 The interface
bootstrap. In order to single out a solution to the crossing equation which
corresponds to our interface, we shall again concentrate on the 3d Ising model,
and in particular on the two-point functions involving the lowest lying odd
primaries φ and σ, on the free and interacting side respectively. The bootstrap
constraints involving φ can be in fact completely solved in any number of 25

---
**페이지 27**
---

dimensions by requiring the correlation functions to be annihilated by the
Laplace operator. Therefore, we start by collecting some general facts about
free bosonic theories in the presence of codimension one conformal defects. Let
us ﬁrst of all consider the two-point function φφ. As it is well known, one can
prove by applying the equations of motion to the φφ OPE that it contains only
twist two operators, and in particular: φ φ 1 + φ2 + (primaries with zero
expectation value). (4.33) The same method can be applied for establishing that
only two primaries appear in the defect OPE of the ﬁeld (this was ﬁrst noticed
in [57]). Indeed, when the Laplace operator is applied to the r.h.s. of the
defect OPE, the parallel derivatives give descendants and we can disregard them.
The derivative orthogonal to the defect imposes a constraint on the scale
dimension of allowed primaries: 0 = φ(x, xd) X b O (b O φ) (b O φ 1) bO(x)
(xd)φb O+2 + descendants (4.34) Hence, there are only two primaries, the
limiting value of the ﬁeld bφ and of its derivative c φ. These primaries have
protected dimensions bφ = d 2 1 and c φ = d 2. We see that the most general
defect CFT featuring the free theory on half of the space, bounded by any
codimension one defect, satisﬁes the following crossing equation: 1 + λφφφ2
aφ2fd bulk(φ2, ξ) = ξφ µ2 φbφ fd bdy(bφ, ξ) + µ2 φc φ fd bdy(c φ, ξ) . (4.35)
All conformal blocks reduce to elementary function: fd bdy d 2 2 , ξ = 1 2ξφ 1 +
ξ ξ + 1 φ! (4.36) fd bdy d 2, ξ = 2 d 2ξφ 1 ξ ξ + 1 φ! , (4.37) so the crossing
equation is equivalent to the following: 1 2 µ2 φbφ + 2 d 2 µ2 φc φ = 1, (4.38a)
1 2 µ2 φbφ 2 d 2 µ2 φc φ = λφφφ2 aφ2. (4.38b) The solution is parametrized by an
angle: µφbφ = 2 cos α, µφc φ = r d 2 2 sin α, λφφφ2 aφ2 = cos 2α. (4.39) The
solution of this particular crossing equation is only a necessary condition for
the existence of a full ﬂedged defect CFT, therefore the question arises whether
for any value of α such a 26

---
**페이지 28**
---

theory exists. Vice versa, a given value of α might be realized in more than one
defect CFT, which diﬀer elsewhere. We can restrict α to take values in the
interval [0, π/2], since sending the defect ﬁelds bφ and c φ to minus themselves
does not spoil their canonical normalization. At the extrema of this interval
one ﬁnds Neumann (α = 0) and Dirichlet (α = π/2) boundary conditions, and at the
center (α = π/4) the trivial interface between the free theory and itself. The
RG interface with the O(N) model with φ4 interaction lies perturbatively near to
the no-interface value, in ϵ-expansion, and ﬁlls an interval if N is allowed to
take value over the reals. Since any two-point function involving the ﬁeld φ has
to contain only the same two blocks in the defect channel, one can generalize
the previous procedure to any correlator of this kind. The general fusion rule
with a primary O with dimension is φ OO+ O+ + (spinning primaries), = φ, + = +
φ. (4.40) Notice that degenerate primaries may exist with the right dimensions
to enter the r.h.s. of eq. (4.40), as it happens in the O(N) model for N 1.
Denoting λ+ = λφOO+ and λ= λφOO, the solution to the bootstrap equation is µφbφ
µObφ = λa+ λ+ a+, 4 d 2 µφc φ µOc φ = λaλ+ a+. (4.41) This includes the system
(4.38), in particular. The relations (4.41) also apply when the operator O is a
primary on the interacting side of the interface. In this case, the OPE happens
in the folded picture, and turns out to be a simple way to choose the solution
of the Laplace equation with the appropriate asymptotics. Speciﬁcally, no
singularities should arise when the operators are placed in mirroring points,
and this prompts us to eliminate Ofrom the r.h.s. of eq. (4.40). In other words,
φ O :φO: , (4.42) and the two-point function is simply φ(x)O(x)= aφO
(2|xd|)φ(2xd)2F1 (φ, , , ξfolded) = aφO (2xd)φ(x x)2φ , (4.43) where ξfolded is
just obtained by replacing xd with minus itself. The relation (4.41) reduces to
aφO = µφbφ µObφ = 4 d 2µφc φ µOc φ. (4.44) This relation is potentially useful
in bootstrapping the interacting side of the interface. Indeed, the defect OPE
of every operator which couples with φ contains bφ and c φ, and the ratio
µObφ/µOc φ = 2 tan α/ d 2 does not depend on the operator, and may be used to
match solutions for diﬀerent external primaries. From eq. (4.38), we see that
this ratio among 27

---
**페이지 29**
---

coeﬃcients of the interacting theory is determined by the expectation value of
φ2 on the free side. In particular, as we pointed out, this one-point function
deviates from zero only at order ϵ2 in the case we are interested in. We compute
the leading order value in appendix A for generic N, and ﬁnd α = π 4 3 1024π6 N
+ 2 (N + 8)2 ϵ2. (4.45) In sum, the signature of the RG domain wall in the
conformal block decomposition of σσ is the presence of two protected defect
operators, with a ratio of OPE coeﬃcients near to the free theory value. In
fact, we found in 3d a numerical solution for a (4,4,0) truncation of σσwhich
has the expected features. The defect channel is formed by the two operators bσ
and d zσ of protected dimensions 1 2 and 3 2 and two unprotected operators bO3
and bO4 of dimensions c 3 3.11 and c 4 6.17. The precise value of these
quantities as well as the estimates of the relative OPE coeﬃcients depend on the
choice of the bulk spectrum. For the sake of consistency we put in the same bulk
spectrum obtained in the (4,2,1) solution of the extraordinary transition. The
values of ε and ε depend on the scale dimension ˆof a surface operator which
acts as a free parameter. Therefore, our interface solution also depends on it,
though the dependence is very mild, as a stable solution requires (see the
discussion on the stability of the solutions on section 2). Table 3 shows the
relevant data of such a solution. Note that the ratio µσbσ/µσd zσ follows the
trend suggested by the ϵ expansion. Let us make some ﬁnal remarks. When the bulk
OPE coeﬃcients and the scale dimensions are exactly known on one side of an
interface, one may extract the one-point functions from the crossing equations
involving operators placed on this side. The same data enter various
correlators, and the interplay between diﬀerent solutions to the crossing
equations may be used to detect systematics, or to reduce the unknowns. We leave
this for future work. For now, we notice that the even spectrum on the free side
of our interface is made by an increasing number of degenerate primaries of
integer dimension, so it is foreseeable that a reliable truncation would require
the inclusion of many bulk primaries. Furthermore, since the parameter N only
enters the determinants through the unknown defect spectrum, one expects to ﬁnd
a one-parameter family of solutions. Studying two-point functions of free even
primaries is important in particular if one is interested in the Zamolodchikov
norm of the displacement operator. Indeed, two defect primaries exist with
dimension d, one of which might be identiﬁed with the displacement of the folded
theory. Given two primaries OL and OR with non-vanishing one-point function, it
is not diﬃcult to see that, in order to isolate the displacement, one needs to
know OLOL, ORORand OLOR. Unfortunately, we have not been able to identify a
solution for ϵ ϵwhich satisfactorily reproduces the domain wall. 5 Conclusions
and outlook. In this paper we explored some consequences of crossing symmetry
for defect CFTs. We focused our study on the cases where the defect is a
codimension one hyperplane, i.e. a ﬂat 28

---
**페이지 30**
---

ˆ ϵ ϵ µ2 σbσ µ2 σd zσ 3.9 7.235(6)(3) 12.736(7)(4) 1.00612(11)(5) 0.27138(5)(2)
7.9 7.274(10)(2) 12.843(17)(4) 1.00644(15)(4) 0.27123(7)(1) 11.1 7.287(11)(4)
12.892(22)(8) 1.00657(17)(6) 0.27117(7)(2) 15.1 7.297(11)(2) 12.932(23)(4)
1.00668(16)(3) 0.27112(7)(2) 19.9 7.298(11)(1) 12.948(24)(2) 1.00667(16)(2)
0.271127(68)(5) 25.5 7.302(11)(2) 12.968(25)(5) 1.00672(16)(3) 0.27110(7)(2)
31.9 7.303(11)(3) 12.980(25)(7) 1.00674(16)(5) 0.27110(7)(1) 39.1 7.307(12)(4)
12.995(28)(8) 1.00679(17)(5) 0.27108(7)(2) ˆ b3 b4 µ2 σ b O3 µ2 σ b O4 3.9
3.1190(8)(4) 6.1816(9)(4) 0.002555(5)(2) 0.00002387(4)(2) 7.9 3.1151(12)(3)
6.1757(15)(4) 0.002572(7)(2) 0.00002408(6)(2) 11.1 3.1136(14)(5) 6.1734(17)(6)
0.002579(7)(3) 0.00002417(7)(3) 15.1 3.1123(14)(3) 6.1715(17)(3) 0.002584(7)(2)
0.00002424(7)(2) 19.9 3.1121(14)(1) 6.1710(17)(2) 0.0025850(74)(10)
0.00002426(7)(1) 25.5 3.1115(14)(3) 6.1701(18)(4) 0.002588(7)(1)
0.00002429(7)(1) 31.9 3.1113(14)(4) 6.1697(17)(5) 0.002589(7)(2)
0.00002431(7)(3) 39.1 3.1108(15)(5) 6.1689(19)(6) 0.002591(8)(2)
0.00002433(8)(2) Table 3. Data of the (4,4,0) solution of the 3d Ising interface
with the free UV theory. The ﬁrst column is the free parameter of the solution
which is the scale dimension of a surface operator contributing to the
extraordinary transition discussed in sec. 3. The data are aﬀected by two kinds
of errors. The ﬁrst parenthesis reﬂects the statistical error of the input data
(namely σ and ε), while the second parenthesis indicates the spread of the
solutions. interface or a boundary. In the latter case our main results concern
the surface transitions of 3d Ising model. The numerical solutions to the
bootstrap equations with the method of determinants turn out to be particularly
eﬀective in the ordinary transition, where it suﬃces to know the scale
dimensions of the ﬁrst few bulk primaries to obtain the dimension of the
relevant surface operator of this transition as well as its OPE coeﬃcient. This
analysis has been extended to the O(N) models with N = 0, 1, 2, 3 where a
comparison can be made with the results of a two-loop calculation [39], ﬁnding a
perfect agreement (see table 1). In the extraordinary transition the
contribution of the boundary channel is dominated by the ﬁrst two low-lying
operators, namely the identity and the displacement, thus we used this fact to
extract more information on the even and odd spectrum contributing to the bulk
channel. We obtained in this way also an accurate determination of the OPE
coeﬃcient λσσε which compares well with other estimates based on a recent Monte
Carlo calculation [48] or on conformal bootstrap [19]. We also obtained some OPE
coeﬃcients of one-point and two-point 29

---
**페이지 31**
---

functions (see table 2) which allow to verify the impressive fulﬁllment of the
Ward identities associated with the displacement operator. The solution
corresponding to the special transition contains a free parameter, hence we dont
get precise numerical results. This case is still very useful for an accurate
cross-check of the consistency of the method of determinants with the linear
functional method. Together with the just mentioned Ward identities, this check
provides evidence for the fact that the systematic error is rather small when a
truncation is stable. In this paper we investigated the stability of the
truncations through the sensitivity to the addition of heavier operators. It
would be important to establish more rigorous bounds on the systematic error,
maybe along the lines of [15]. The next example of a codimension one defect
studied in this paper is an interface be- tween the O(N) model and the free
theory. We tackled the problem both in 4ϵ and in three dimensions. The weak
coupling analysis of the two-point functions was carried out in a way which is
trivially adapted to general perturbation interfaces. A preeminent role is
played by the displacement operator, whose small Zamolodchikov norm signals the
transparency of the interface, in the sense that operators with nearly
degenerate dimension are allowed to couple at order one across the interface,
while the opposite is true for primaries well separated in the spectrum. This
intuition can be made precise in 2d, where the norm of the displacement
coincides up to a normalization with the reﬂection coeﬃcient deﬁned in [58].8 It
is certainly interesting to look for a similar interpretation of the
displacement in higher dimensions, possi- bly in relation to the correlators of
polarized stress-tensors. However, it is worth emphasizing that while in 2d the
reﬂection coeﬃcient of a boundary is unity, in dimensions greater than two the
norm of the displacement depends on the boundary conditions. The results of the
perturbative analysis also conﬁrm that this kind of interfaces encode
information about the RG ﬂow that links the theories on the two sides:
speciﬁcally, the coupling of UV and IR pri- maries reproduces the leading order
mixing of operators, as does the one-dimensional domain wall constructed
non-perturbatively in [51]. On the numerical side, we found a solution to the
crossing equation consistent with the features of the two-point function of σ in
three dimen- sions. The analysis can be extended in various directions. It would
be interesting go to second order in perturbation theory [59], or to study the
setting at large N, and see whether the displacement operator still provides
important simpliﬁcations. We already pointed out that it is viable to bootstrap
correlators on the free side, and it would be important in particular to give a
prediction for the norm of the displacement in 3d, to compare it with the
estimates for the boundary transitions. We would also like to emphasize that the
interface can be realized on the lattice, for instance as a Gaussian model with
the addition of a quartic potential on one-half of the lattice. As we mentioned
in the introduction, a complete description of the CFT data cannot be reached,
even in principle, only through the study of bulk two-point functions. Four- 8In
particular, it is not diﬃcult to prove unitarity bounds for reﬂection and
transmission in function of the central charges, just by diagonalizing the
defect spectrum. 30

---
**페이지 32**
---

point functions of defect operators should be studied, and in this case both the
method of determinants and the linear functional might be employed. Along the
same lines, in both the boundary and the interface setups one may study the
crossing constraints coming from correlators of the kind O1O2 bO, or two-point
functions of tensors. The necessary tools for the latters were developed in
[14]. It is of course viable to use the method of determinants for the study of
generic defects, and in particular it would be nice to complement the bootstrap
analysis carried out in [18] for the twist line in the Ising model.
Acknowledgements We thank Leonardo Rastelli for pointing out an error in the
ﬁrst version of this paper. Pre- liminary results of this paper were ﬁrst
presented at the workshop Back to the Bootstrap IV at Porto University, June
30-July 11, 2014. FG would like to thank the organizers and the participants for
the stimulating atmosphere; he also thanks John Cardy and Slava Rychkov for
fruitful discussions. PL would like to thank Leonardo Rastelli and Balt van Rees
for discussions during the early stages of this work. MM would like to thank
Michele Caselle, Dalimil Mazac, Enrico Trincherini and Ettore Vicari for useful
discussions, and especially Davide Gaiotto, for suggesting the study of the RG
interface and for many illuminating dis- cussions. He also thanks Perimeter
Institute for Theoretical Physics for the hospitality during the preparation of
this paper. PL is supported by SFB 647 Raum-Zeit-Materie. Analytische und
Geometrische Strukturen. AR is supported by the Leverhulme Trust (grant
RPG-2014- 118) and STFC (grant ST/L000350/1). Research at Perimeter Institute is
supported by the Government of Canada through Industry Canada and by the
Province of Ontario through the Ministry of Research and Innovation. A RG domain
wall: details on the ϵ-expansion. A.1 One loop computations. Two regularization
procedures have been preferred in the literature, in dealing with the φ4 model
in the presence of a defect of co-dimension one. Dimensional regularization has
been especially used for the systematic renormalization of the Lagrangian and
for extracting the critical exponents [6062]. More recently, fully real space
computations were carried out in [33, 34], with a short distance cutoﬀ. Both
series of works were concerned with the φ4 theory in the presence of a plain
boundary. We follow the latter technique. We start by checking eq. (4.11)
through the two-point function φ2φ2on the free side of the domain wall. At one
loop, the only diagram contributing is shown in ﬁg. 8. Since the correlator
depends only on one cross-ratio, it is suﬃcient [34] to compute the two-point
function in the collinear geometry of ﬁg. 8, for which ξ (y y)2 4yy . (A.1) 31

---
**페이지 33**
---

The corresponding integral is φ2(x)φ2(x)one-loop = 1 3N(N + 2)g Z 0 dz Z dd1x 1
{(x2 + (z + y)2) (x2 + (z + y)2)}d2 . (A.2) Notice that we chose y, y 0. The
integral does not diverge in the UV. This is expected, since the coupling
constant renormalizes at O(ϵ2), and the lowest lying interface operator that
might be needed as a counterterm is φ4, which however - barring mixing which
appears at higher orders - equals the displacement operator and is therefore
irrelevant. Since the ﬁxed point coupling constant gis of order ϵ, we can plug d
= 4 in the integral to obtain the leading order correction, which is easily
computed. The result is φ2(x)φ2(x)one-loop = N(N + 2) 3 g π2 (y y)4 ξ ξ + 1
log(1 + ξ) + O(ϵ2). (A.3) Plugging into this expression the ﬁxed point value for
g(4.8) and adding the tree level contribution, one obtains the correlator at
ﬁrst order in ϵ-expansion: φ2(x)φ2(x)= 2N s2(d2) 1 + 1 2 N + 2 N + 8 ϵ ξ ξ + 1
log(1 + ξ) . (A.4) Notice that the one point function of φ2 is O(ϵ2), therefore
this is the full correlator - not just the connected part - at order ϵ.
Comparing the result with the form of the conformal block of φ4, evaluated in d
= 4 at this order: fd=4 bulk(φ4; ξ) = 2 ξ ξ + 1 log(1 + ξ) , (A.5) and using
that in free theory λφ2φ2φ4 = r 2(N + 2) N , (A.6) ( ,-y) 0 ( ,-y') 0 ( ,z) x (
,-y) 0 ( , y') 0 ( ,z) x Figure 8. One loop contributions to φ2(x)φ2(x)and to
φ2φi(x)φj(x). The free side is the left one, and y, y, z 0. 32

---
**페이지 34**
---

we see agreement with the general result (4.11) and with the one-point function
in eq. (4.10). Let us compare also the general formula (4.16) with an explicit
one-loop example. We focus on the correlator between the ﬁeld φi on the
interacting side and the free primary φ2φi. The one-loop contribution is encoded
in the diagram on the right in ﬁg. 8, which is UV ﬁnite. Including the
combinatorics, the result is φ2φi(x) p 2(N + 2) φj(x)= δij (2y)3(2y) ξ2 N + 2 2
2(N + 8) ϵ (ξ 1). (A.7) It is easy to compute the tree level three-point
function needed to ﬁx µL RD, and see that eq. (A.7) matches eq. (4.16). Next, we
compute the ﬁrst non-trivial contribution to the two-point function of φi on the
free side, which departs from its free theory value at order ϵ2. The only
diagram contributing is the sunset (ﬁg. 9). As explained in subsection 4.3, we
actually only need to know aφ2, which amounts to colliding the two external
operators in the diagram. The computation only slightly simpliﬁes at this order,
but the statement is valid at any loop (and of course, for any interface
involving the free theory). The bulk conformal block of the operator φ2 of
dimension φ2 = d 2 is fbulk(φ2; ξ) = ξ 1 + ξ d2 . (A.8) Therefore we ﬁnd
φi(x)φj(x)= δij sd2 1 + λφφφ2aφ2 ξ ξ + 1 d2! . (A.9) The integral to be
evaluated is the following: I = Z 0 dz Z 0 dz Z d3x Z d3x 1 x2 + (y + z)2x2 + (y
+ z)2 1 (x x)2 + (z z)23 . (A.10) ( ,-y') 0 ( ,-y) 0 ( ,z) x ( ,z') x' Figure 9.
Two loops contribution to φi(x)φj(x). Again, y, y, z, z 0. 33

---
**페이지 35**
---

Along the computation, which is straightforward, we encounter two divergences. A
bulk di- vergence requires a mass counterterm, and a second divergence arises
when the interaction vertices hit the interface. This is compensated by
integrating bφ2 along the interface. Relevant operators are required because our
cut-oﬀbreaks scale invariance. Their renormalized cou- plings, however, must be
ﬁne-tuned in order to reach the critical point. Hence, requiring scale
invariance of the one-point function is suﬃcient to ﬁx the subtraction
unambiguously. After renormalization, one ﬁnds I = 3π4 16 1 y2 . (A.11) Taking
the combinatorics into account, the expectation value at leading order is φ2(y)
2N aφ2 (2y)2 = 3 512π6 r N 2 N + 2 (N + 8)2 ϵ2 1 (2y)2 . (A.12) Substituting
back in (A.9), and using λφφφ2 = r 2 N , (A.13) we ﬁnd at this order φi(x)φj(x)=
δij sd2 1 + 3 512π6 N + 2 (N + 8)2 ϵ2 ξ ξ + 1 2! . (A.14) One can now extract
some CFT data. By using the relations (4.38) one ﬁnds the defect OPE coeﬃcients
µφ bφ = 1 + 3 1024π6 N + 2 (N + 8)2 ϵ2, µφ c φ = r d 2 4 1 3 1024π6 N + 2 (N +
8)2 ϵ2 . (A.15) We also obtain a piece of information about the defect OPE of
any primary on the interacting side which couples with φi, through the
equalities (4.44): µO bφ µO c φ = 4 d 2 µφ c φ µφ bφ = 2 d 2 1 3 512π6 N + 2 (N
+ 8)2 ϵ2 . (A.16) We use this result in subsection 4.3 as a check of the
solutions to the approximate crossing equation for σσ. A.2 Two-point functions
across the interface. We give some details on the formulae (4.14), (4.16) and
(4.17). Let us call xd = yi the position of the interface. We choose again the
collinear geometry for the two operators and we place one on either side of the
interface, at the points x = (x, yL yi) and x = (x, yR yi). 34

---
**페이지 36**
---

After plugging the free theory three-point function in eq. (4.12), we shall ﬁnd
the two-point function by solving the following equation: µL R D (yR yL)L+R4 Z
d3z z2 + (yL yi)24+LR 2 z2 + (yR yi)24LR 2 = d dyi OL(x)OR(x). (A.17) First of
all, we brieﬂy comment on (4.14), that is, on the case L R = O(ϵ). Since µL R is
also at least of order ϵ, we can plug L = R in (A.17). The integrals are easily
evaluated and we get OL(x)OR(x)= π2µL R D |yL yR|2L log yR yi yL yi + c(yR, yL).
(A.18) The constant of integration c(yR, yL) does not depend on the position of
the interface. One way to ﬁx it is to require that when the interface stands
half-way between the points the correlator takes the form (4.3): c(yR, yL) = aL
R |yL yR|L+R . (A.19) By asking for conformal invariance of this result, one
gets back at ﬁrst order the scaling relation (4.5). Eq. (4.14) is then obtained
by reconstructing the correlator for generic choice of the two points through
conformal invariance. Let us now tackle the case of external dimensions diﬀering
at order one. The integra- tion in the translational invariant directions is
easily recast as the Euler representation of a hypergeometric function: Z d3z z2
+ (yL yi)24+LR 2 z2 + (yR yi)24LR 2 = π2 8 |yL yi|1LR|yR yi|4+LR 2F1 3 2, 2 LR 2
; 4; 1 yL yi yR yi 2 (A.20) Internal and spacetime symmetries allow to restrict
ourselves to the case L R = 2k, for integer k, at this order. Furthermore, there
is a clear symmetry for the exchange L R, so we only consider the case k 0.
Since for k 1 the hypergeometric function is a polynomial, we treat separately
the case k = 1. Eq. (4.16) is obtained integrating the position of the interface
and again ﬁxing the integration constant in accordance with conformal
invariance. When k = 2, 3, . . . one can write eq, (A.17) as OL(x)OR(x)= µL R D
(yR yL)L+R Z y0 (yL+yR)/2 dyi 3π3/2Γ k 1 2 2Γ(k + 2) (yR yL)4 (yi yL)5 2F1 5 2,
2 k; 3 2 k; (yR yi)2 (yL yi)2 + aL R (yR yL)L+R . (A.21) 35

---
**페이지 37**
---

One can exploit the fact that the hypergeometric function is a polynomial and
integrate addend by addend the second line of (A.21). In particular, we can
choose to put the interface in y0 = 0. Some simpliﬁcations occur because of the
following observation. As already pointed out, the value of aL R is ﬁxed by the
requirement of conformal invariance. On the other hand, any constant piece in
the integration has the only eﬀect of shifting aL R. Therefore, we disregard
such pieces, and ﬁx the constant in the end. All together, introducing the scale
invariant variable r = yL/yR, we ﬁnd OL(x)OR(x)= µL R D (yR yL)L+R (1)kπ5/2 (k
1)k2Γ(k + 2)Γ(k 1/2) (r 1)2 4r 2k(r2 r + 1) + (r + 1)2 2F1 1 2, k; k 1 2; 1 r2
2k(r2 r) + (1 + r)2 2F1 3 2, k; k 1 2; 1 r2 + a (yR yL)L+R . (A.22) We only need
to enforce invariance under inversions, which amounts to sending yR 1/yR and yL
1/yL. With the help of standard hypergeometric identities one can check that the
ﬁrst three lines in (A.22) are invariant, therefore a = 0. (A.23) Alternatively,
one may simply verify that with this choice the relation (4.5) is fulﬁlled. The
result is not yet explicitly a function of the cross-ratio. The ﬁnal form eq.
(4.17) can be obtained at the price of some more massage. References [1] J.
Polchinski, Scale and Conformal Invariance in Quantum Field Theory, Nucl.Phys.
B303 (1988) 226. [2] A. Dymarsky, Z. Komargodski, A. Schwimmer, and S. Theisen,
On Scale and Conformal Invariance in Four Dimensions, arXiv:1309.2921 [hep-th].
[3] V. Yurov and A. Zamolodchikov, Truncated conformal space approach to scaling
Lee-Yang model, Int.J.Mod.Phys. A5 (1990) 32213246. [4] M. Hogervorst, S.
Rychkov and B. C. van Rees, Truncated conformal space approach in d dimensions:
A cheap alternative to lattice ﬁeld theory?, Phys. Rev. D 91 (2015) 025005,
arXiv:1409.1581 [hep-th]. [5] A. Coser, M. Beria, G. P. Brandino, R. M. Konik,
and G. Mussardo, Truncated Conformal Space Approach for 2D Landau-Ginzburg
Theories, arXiv:1409.1494 [hep-th]. [6] J. M. Maldacena, The Large N limit of
superconformal ﬁeld theories and supergravity, Int.J.Theor.Phys. 38 (1999)
11131133, arXiv:hep-th/9711200 [hep-th]. 36

---
**페이지 38**
---

[7] G. Mack, Convergence of Operator Product Expansions on the Vacuum in
Conformal Invariant Quantum Field Theory, Commun.Math.Phys. 53 (1977) 155. [8]
S. Ferrara, A. Grillo, and R. Gatto, Tensor representations of conformal algebra
and conformally covariant operator product expansion, Annals Phys. 76 (1973)
161188. [9] R. Rattazzi, V. S. Rychkov, E. Tonni, and A. Vichi, Bounding scalar
operator dimensions in 4D CFT, JHEP 0812 (2008) 031, arXiv:0807.0004 [hep-th].
[10] V. S. Rychkov and A. Vichi, Universal Constraints on Conformal Operator
Dimensions, Phys.Rev. D80 (2009) 045006, arXiv:0905.2211 [hep-th]. [11] R.
Rattazzi, S. Rychkov, and A. Vichi, Central Charge Bounds in 4D Conformal Field
Theory, Phys.Rev. D83 (2011) 046011, arXiv:1009.2725 [hep-th]. [12] D. Poland
and D. Simmons-Duﬃn, Bounds on 4D Conformal and Superconformal Field Theories,
JHEP 1105 (2011) 017, arXiv:1009.2087 [hep-th]. [13] S. El-Showk, M. F. Paulos,
D. Poland, S. Rychkov, D. Simmons-Duﬃn, A. Vichi, Solving the 3D Ising Model
with the Conformal Bootstrap, Phys.Rev. D86 (2012) 025022, arXiv:1203.6064
[hep-th]. [14] P. Liendo, L. Rastelli, and B. C. van Rees, The Bootstrap Program
for Boundary CFTd, JHEP 1307 (2013) 113, arXiv:1210.4258 [hep-th]. [15] D.
Pappadopulo, S. Rychkov, J. Espin, and R. Rattazzi, OPE Convergence in Conformal
Field Theory, Phys.Rev. D86 (2012) 105043, arXiv:1208.6449 [hep-th]. [16] S.
El-Showk and M. F. Paulos, Bootstrapping Conformal Field Theories with the
Extremal Functional Method, Phys.Rev.Lett. 111 (2013) no. 24, 241601,
arXiv:1211.2810 [hep-th]. [17] F. Gliozzi, More constraining conformal
bootstrap, Phys.Rev.Lett. 111 (2013) 161602, arXiv:1307.3111. [18] D. Gaiotto,
D. Mazac, and M. F. Paulos, Bootstrapping the 3d Ising twist defect, JHEP 1403
(2014) 100, arXiv:1310.5078 [hep-th]. [19] S. El-Showk, M. F. Paulos, D. Poland,
S. Rychkov, D. Simmons-Duﬃn, A. Vichi, Solving the 3d Ising Model with the
Conformal Bootstrap II. c-Minimization and Precise Critical Exponents,
J.Stat.Phys. 157 (2014) 869, arXiv:1403.4545 [hep-th]. [20] F. Gliozzi and A.
Rago, Critical exponents of the 3d Ising and related models from Conformal
Bootstrap, JHEP 1410 (2014) 42, arXiv:1403.6003 [hep-th]. [21] C. Beem, L.
Rastelli, and B. C. van Rees, The N = 4 Superconformal Bootstrap, Phys.Rev.Lett.
111 (2013) 071601, arXiv:1304.1803 [hep-th]. [22] Y. Nakayama and T. Ohtsuki,
Five dimensional O(N)-symmetric CFTs from conformal bootstrap, Phys.Lett. B734
(2014) 193197, arXiv:1404.5201 [hep-th]. [23] S. M. Chester, J. Lee, S. S. Pufu,
and R. Yacoby, The N = 8 superconformal bootstrap in three dimensions, JHEP 1409
(2014) 143, arXiv:1406.4814 [hep-th]. [24] F. Kos, D. Poland, and D.
Simmons-Duﬃn, Bootstrapping Mixed Correlators in the 3D Ising Model,
arXiv:1406.4858 [hep-th]. 37

---
**페이지 39**
---

[25] S. M. Chester, S. S. Pufu, and R. Yacoby, Bootstrapping O(N) Vector Models
in 4 d 6, arXiv:1412.7746 [hep-th]. [26] C. Beem, M. Lemos, P. Liendo, L.
Rastelli, and B. C. van Rees, The N = 2 superconformal bootstrap,
arXiv:1412.7541 [hep-th]. [27] D. Simmons-Duﬃn, A Semideﬁnite Program Solver for
the Conformal Bootstrap, arXiv:1502.02033 [hep-th]. [28] N. Bobev, S. El-Showk,
D. Mazac, and M. F. Paulos, Bootstrapping the Three-Dimensional Supersymmetric
Ising Model, arXiv:1502.04124 [hep-th]. [29] M. Billó, M. Caselle, D. Gaiotto,
F. Gliozzi, M. Meineri, R. Pellegrini, Line defects in the 3d Ising model, JHEP
1307 (2013) 055, arXiv:1304.4110 [hep-th]. [30] A. Allais and S. Sachdev,
Spectral function of a localized fermion coupled to the Wilson-Fisher conformal
ﬁeld theory, Phys.Rev. B90 (2014) 035131, arXiv:1406.3022 [cond-mat.str-el].
[31] Dias, Òscar J.C. and Horowitz, Gary T. and Iqbal, Nabil and Santos, Jorge
E., Vortices in holographic superﬂuids and superconductors as conformal defects,
JHEP 1404 (2014) 096, arXiv:1311.3673 [hep-th]. [32] M. R. Douglas, Spaces of
Quantum Field Theories, J.Phys.Conf.Ser. 462 (2013) no. 1, 012011,
arXiv:1005.2779 [hep-th]. [33] D. McAvity and H. Osborn, Conformal ﬁeld theories
near a boundary in general dimensions, Nucl.Phys. B455 (1995) 522576,
arXiv:cond-mat/9505127 [cond-mat]. [34] D. McAvity and H. Osborn, Energy
momentum tensor in conformal ﬁeld theories near a boundary, Nucl.Phys. B406
(1993) 655680, arXiv:hep-th/9302068 [hep-th]. [35] V. S. Rychkov, Conformal
invariance in D 3, Lectures given at EPFL (2012) . [36] P. Di Francesco, P.
Mathieu, and D. Senechal, Conformal ﬁeld theory. Springer, 1997. [37] J. L.
Cardy, Scaling and Renormalization in Statistical Physics. Cambridge Lecture
Notes in Physics, 1996. [38] J. L. Cardy and D. C. Lewellen, Bulk and boundary
operators in conformal ﬁeld theory, Phys.Lett. B259 (1991) 274278. [39] H. Diehl
and M. Shpot, Massive ﬁeld theory approach to surface critical behavior in
three-dimensional systems, Nucl.Phys. B528 (1998) 595647, arXiv:cond-mat/9804083
[cond-mat]. [40] M. Hasenbusch, The thermodynamic Casimir force: A Monte Carlo
study of the crossover between the ordinary and the normal surface universality
class, Phys. Rev. B83 (2011) 134425, arXiv:1012.4986 [cond-mat.stat-mech]. [41]
Youjin Deng, Henk W. J. Blöte, and M. P. Nightingale, Surface and bulk
transitions in three-dimensional O(n) models, Phys. Rev. E72 (2005) 016128,
arXiv:cond-mat/0504173. [42] T. Prellberg, Scaling of Self-Avoiding Walks and
Self-Avoiding Trails in Tree Dimensions, J. Phys. A: Math. Gen. 34 (2001) L599,
arXiv:cond-mat/0108538 [hep-th]. [43] M. Hasenbusch, Finite size scaling study
of lattice models in the three-dimensional Ising universality class, Phys.Rev.
B82 (2010) 174433, arXiv:1004.4486. 38

---
**페이지 40**
---

[44] M. Campostrini, M. Hasenbusch, A. Pelissetto, P. Rossi, and E. Vicari,
Critical behavior of the three-dimensional xy universality class, Phys.Rev. B63
(2001) 214503, arXiv:cond-mat/0010360 [cond-mat]. [45] M. Campostrini, M.
Hasenbusch, A. Pelissetto, P. Rossi, and E. Vicari, Critical exponents and
equation of state of the three-dimensional Heisenberg universality class,
Phys.Rev. B65 (2002) 144520, arXiv:cond-mat/0110336 [cond-mat]. [46] R. Guida
and J. Zinn-Justin, Critical exponents of the N vector model, J.Phys. A31 (1998)
81038121, arXiv:cond-mat/9803240 [cond-mat]. [47] D. F. Litim, Critical
exponents from optimized renormalization group ﬂows, Nucl. Phys. B 631 (2002)
128, arXiv:hep-th/0203006 [hep-th]. [48] M. Caselle, G. Costagliola, and N.
Magnoli, Numerical determination of OPE coeﬃcients in the 3D Ising model from
oﬀ-critical correlators, arXiv:1501.04065 [hep-th]. [49] S. Fredenhagen and T.
Quella, Generalised permutation branes, JHEP 0511 (2005) 004,
arXiv:hep-th/0509153 [hep-th]. [50] I. Brunner and D. Roggenkamp, Defects and
bulk perturbations of boundary Landau-Ginzburg orbifolds, JHEP 0804 (2008) 001,
arXiv:0712.0188 [hep-th]. [51] D. Gaiotto, Domain Walls for Two-Dimensional
Renormalization Group Flows, JHEP 1212 (2012) 103, arXiv:1201.0767 [hep-th].
[52] A. Konechny and C. Schmidt-Colinet, Entropy of conformal perturbation
defects, J.Phys. A47 (2014) no. 48, 485401, arXiv:1407.6444 [hep-th]. [53] G.
Poghosyan and H. Poghosyan, RG domain wall for the N=1 minimal superconformal
models, arXiv:1412.6710 [hep-th]. [54] E. Eisenriegler, Polymers Near Surfaces.
World Scientiﬁc, Singapore, 1993. [55] L. S. Brown, Dimensional Regularization
of Composite Operators in Scalar Field Theory, Annals Phys. 126 (1980) 135. [56]
A. Zamolodchikov, Renormalization Group and Perturbation Theory Near Fixed
Points in Two-Dimensional Field Theory, Sov.J.Nucl.Phys. 46 (1987) 1090. [57] T.
Dimofte and D. Gaiotto, An E7 Surprise, JHEP 1210 (2012) 129, arXiv:1209.1404
[hep-th]. [58] T. Quella, I. Runkel, and G. M. Watts, Reﬂection and transmission
for conformal defects, JHEP 0704 (2007) 095, arXiv:hep-th/0611296 [hep-th]. [59]
M. R. Gaberdiel, A. Konechny, and C. Schmidt-Colinet, Conformal perturbation
theory beyond the leading order, J.Phys. A42 (2009) 105402, arXiv:0811.3149
[hep-th]. [60] H. Diehl and S. Dietrich, Field-theoretical approach to
multicritical behavior near free surfaces, Phys.Rev. B24 (1981) 28782880. [61]
H. Diehl and S. Dietrich, Multicritical behaviour at surfaces, Zeit. f. Phys B
50 (1983) 117. [62] H. Diehl, S. Dietrich, and E. Eisenriegler, Universality,
irrelevant surface operators, and corrections to scaling in systems with free
surfaces and defect planes, Phys.Rev. B27 (1983) 29372954. 39

---
**페이지 41**
---

[63] J. Padayasi, A. Krishnan, M. A. Metlitski, I. A. Gruzberg, and M. Meineri,
The extraordinary boundary transition in the 3d O(N) model via conformal
bootstrap, arXiv:2111.03071. 40