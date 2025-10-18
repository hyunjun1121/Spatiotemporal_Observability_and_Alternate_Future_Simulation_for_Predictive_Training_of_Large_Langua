
---
**페이지 1**
---

Mathematical modelling of flow and adsorption in a gas chromatograph A.
Cabrera-Codonya, A. Valverdeb, K. Bornc, O.A.I. Noreldind, T.G. Myers1e aLEQUIA,
Institute of the Environment, Universitat de Girona, Spain bDepartment of
Chemical Engineering, Universitat Polit`ecnica de Catalunya, Spain cSchool of
Computer Science and Applied Mathematics, University of the Witwatersrand,
Johannesburg, South Africa d Department of Mathematical Sciences, University of
Zululand, South Africa eCentre de Recerca Matem`atica, Barcelona, Spain Abstract
In this paper, a mathematical model is developed to describe the evolution of
the concentration of compounds through a gas chromatography column. The model
couples mass balances and kinetic equations for all components. Both single and
multiple-component cases are considered with constant or variable velocity.
Non-dimensionalisation indicates the small effect of diffusion. The system where
diffusion is neglected is analysed using Laplace transforms. In the
multiple-component case, it is demonstrated that the competition between the
compounds is negligible and the equations may be decoupled. This reduces the
problem to solving a single integral equation to determine the concentration
profile for all components (since they are scaled versions of each other). For a
given analyte, we then only two parameters need to be fitted to the data. To
verify this approach, the full governing equations are also solved numerically
using the finite difference method and a global adaptive quadrature method to
integrate the Laplace transformation. Comparison with the Laplace solution
verifies the high degree of accuracy of the simpler Laplace form. The Laplace
solution is then verified against experimental data from BTEX chromatography.
This novel method, which involves solving a single equation and fitting Preprint
submitted to Partial Differential Equations in Applied Math. January 3, 2025
arXiv:2501.00001v1 [cs.CE] 7 Oct 2024

---
**페이지 2**
---

parameters in pairs for individual components, is highly efficient. It is
significantly faster and simpler than the full numerical solution and avoids the
computationally expensive methods that would normally be used to fit all curves
at the same time. Keywords: Gas chromatography, Column Adsorption,
Advection-Diffusion equation, Compounds Separation 1. Introduction Gas
chromatography (GC) is a widely used technique for identifying and analysing
volatile compounds. It has a broad range of applications, such as detecting and
quantifying pollutants, pesticides, and environmental contaminants in air,
water, and soil samples [1, 2]. In food and beverage analysis, it is used to
determine the presence and concentrations of flavour compounds or additives. In
the pharmaceutical industry, GC is employed for analysing drugs, including their
purity, and the quantification of active ingredients. Additionally, clinical and
medical laboratories use GC for analysing blood, urine, and other biological
samples [3, 4]. GC operates through a series of precise steps to separate and
analyse volatile compounds within a sample. Initially, a small quantity of the
sample is injected into the chromatograph. A carrier gas, typically helium or
nitrogen, transports the vaporised sample through a chromatographic column. The
column may be lined with a thin layer of liquid or packed with adsorbent
material. The flowing gas is termed the mobile phase, while the lining or packed
solid adsorbent is called the stationary phase- nowadays most GC systems involve
a polymeric liquid stationary phase. Typical dimensions for a column are of the
order 5-50m long with an inner diameter of 100300 microns. The processs
temperature is controlled by placing the column inside an oven. The separation
process is caused by the interaction between the sample compounds and the
stationary phase. As the sample travels through the column, the compounds
molecules are repeatedly adsorbed and desorbed by the stationary phase. The
rates of attachment differ for different molecules (depending on, for example,
their molecular size, polarity, and volatility) and different types of
stationary phases. Compounds that interact strongly with the stationary phase
spend more time in the column 2

---
**페이지 3**
---

and move slower, while those with weaker interactions move through quicker. Oven
temperature programming is often employed, where the column temperature is
gradually increased during the process. This helps achieve efficient separation
by progressively reducing the compounds affinity for the stationary phase,
allowing them to elute based on their boiling points and interactions with the
stationary phase. Upon exiting the column, the mixture passes through a detector
which identifies the separate compounds. Common types of detectors used in GC
include flame ionisation detectors, electron capture detectors, and mass
spectrometers. The signals produced by the detector are then recorded and
analysed to create a chromatogram. The chromatogram is a graphical
representation that shows the concentration of compounds over time. Typically it
takes the form of a series of Gaussian-like and asymmetric peaks [5], which
appear at different times. The shape of the elution peak and retention time are
affected by operating parameters and materials. Various authors have carried out
theoretical and experimental studies on liquid chromatography, as documented in
[5, 6, 7, 8, 9]. The importance of the retention time and peak width in gas
chromatography columns has been addressed in studies by Dose [10] and Rodrıguez
et al. [11]. Laplace transforms have proved effective in the study of single
components, under a number of restrictions. Guiochon and Lin [12] present a
solution for a pulse inlet condition after neglecting diffusion. Noting that the
inverse transform is complicated they reduce this condition to a delta function.
They also discuss the case with diffusion but are forced to apply a constant
concentration at the inlet. Their solution follows the work of Lapidus and
Amundson [13] which is valid for a semi-infinite column. In all cases, the
velocity remains constant throughout the column and only a single component is
analysed. Mathematical models for chromatography are separated based on the type
of adsorption and operating parameters. Linear equilibrium elution in GC with
capillary columns has been investigated by Aris [14] and Golay[15]. Packed
column GC is well studied across linear/nonlinear equilibrium/nonequilibrium
elutions [16, 17, 18, 19, 20, 21] and similarly with open columns [22, 23]. More
recently the work done by Guiochon and Lin [12] and Asnin et al. [24] in which
nonlinear isotherms were considered, takes into account both the analyte
concentrations in the liquid phase and sites of adsorption in the stationary
phase. Rodrıguez et al. [11] proposed a mathematical model based on the
advection-diffusion equation and Langmuir kinetic equation to analyse the 3

---
**페이지 4**
---

transport of BTEX (benzene, toluene, ethylbenzene, and xylene isomers) molecules
inside a capillary chromatography column. This model was solved numerically
using a finite volume method, and the results were validated with experimental
data on BTEX compounds. This study highlighted the importance of considering
proper volatile organic compounds (VOCs) separation for accurate quantification
and monitoring of pollutants. A similar system but with a linear kinetic
equation was employed to investigate liquid chromatography in a packed fixed
cylindrical column [25, 26]. In both studies, the finite volume method was
utilised to obtain numerically solutions, analyse the performance of the
underlying process, investigate retention behavior, and identify the optimal
parameter values in the liquid chromatography process. Perveen et. al. [26]
employed a linear kinetic equation to describe the rate of change of
concentration. For the computations they employed the so-called bi-Langmuir
isotherms, which are inconsistent with the kinetic equation (the isotherm comes
from the steady-state of the kinetic equation). Mathematical models for GC are
analogous to the capture of contaminants in a packed columm. The theory in this
field for the single contaminant case is well established and has been recently
advanced in [27, 28, 29]. These papers involve the coupling of an
advection-diffusion equation describing the mass balance with linear and
nonlinear kinetic equations. Analytical solutions are obtained through
identifying dominant terms and applying a travelling wave substitution.
Comparison with numerical solutions and experimental data confirm their
accuracy, in particular showing significant improvements over standard models
(many of which are presented in the review of [30]). Aguareles et al. [31]
extend the results to include chemical bonding, providing a family of solutions
for different reactions. The difference between such models and previously
accepted standard ones is clarified in [32]. In particular it is explained how
the travelling wave solutions maintain accuracy over a wide range of operating
conditions while earlier models tend to be accurate for a single experiment but
fail when conditions change. The objective of the present study is to develop
and analyse a mathematical model for GC. The model development will follow that
of [29, 32] for column adsorption. The purpose of the study being to gain a
deeper understanding of the behaviour of materials whilst passing through a
column and so, once developed and verified, the model may be used as a tool for
analysing and optimising the separation process in gas chromatography. 4

---
**페이지 5**
---

2. Mathematical model Here we present two basic models for the flow, adsorption
and desorption of molecules in a long thin column. Assuming that the sample is
introduced in a short burst at time t = 0 we first consider the case where the
velocity is constant. This is based on the fact that after a sufficiently large
distance into the column, we expect the sample to have mixed with the carrier
fluid, since the carrier fluid occupies a large volume compared to the sample
the removal of small quantities should have a negligible effect on the flow. In
the second case, we allow for density variation due to the pressure drop along
the column such that the flow varies. In both cases, we assume that the
stationary phase is always far from being fully loaded. This is in contrast to
the packed column models of [27, 28, 29], where the intention is to retain as
much contaminant as possible. With GC a small sample is introduced to a very
long column and the desorption is at a similar rate to adsorption, consequently
only partial loading occurs. 2.1. Constant velocity model Under the assumption
of a constant fluid velocity, u, the evolution of the cross-sectional average
concentration in the gas mixture and the amount captured by the stationary phase
may be expressed by c t + u c x = D 2c x2 αq t , (1) q t = kac kdq . (2)
Equation (1) is a mass balance for the concentration density c, D represents the
diffusion coefficient, and q is the amount attached to the stationary phase. The
coefficient α = 2δ/R is known as the phase ratio and represents the difference
between the volume of stationary and mobile phases assuming the stationary phase
coats a circular cylindrical tube (δ is the thickness of the stationary phase
layer). The derivation of (1) is provided in Appendix A. Equation (2) represents
the attachment/detachment process, where ka, kd are the adsorption and
desorption coefficients respectively. It may be viewed as a reduced form of
Langmuir kinetic equation where the amount attached at any moment is
significantly lower than the attachment capacity (this follows from the
assumption that the loading of the stationary phase is low in GC). 5

---
**페이지 6**
---

To determine the relative strength of the terms in the equations we non-
dimensionalise the variables ˆx = x L ˆt = t τ ˆc = c c0 ˆq = q qe where qe =
Kc0 L = uR 2δka τ = qe kac0 . The scale qe is the adsorbed quantity in
equilibrium. It is defined from the steady-state of (2), where K = ka/kd and c0
the concentration at the inlet. The time-scale τ indicates the order of
magnitude of the time taken for the attachment process, that is we work on the
attachment time-scale rather than the faster flow time-scale. The length-scale L
is then the distance travelled by the fluid over the attachment time scale. The
governing equations may now be written as Daˆc ˆt + ˆc ˆx = Pe1 2ˆc ˆx2 ˆq ˆt
(3) ˆq ˆt = ˆc ˆq (4) where Da = L/(uτ) is the Damkohler number and Pe1 = D/(uL)
is the inverse Peclet number. With hydrogen as the carrier gas, moving at 2cm/s
in a column with dimension 100µm the Reynolds number is of the order 102 and the
flow is clearly laminar, diffusion then is purely the result of Brownian motion.
Noting that Brownian diffusion is generally negligible in comparison to
advection the Peclet number term may be neglected [31, 29]. The Damkohler number
Da is also expected to be small however close to the start of the process there
will be a time boundary layer where this term may be important, particularly for
the numerical scheme which requires the term to be retained in order to apply
the initial condition. Consequently, for a single compound we define the base
system Daˆc ˆt + ˆc ˆx = ˆq ˆt (5) ˆq ˆt = ˆc ˆq . (6) Assuming the column is
initially free of the sample, which is introduced at the inlet over a short
period, we apply ˆc(ˆx, 0) = q(ˆx, 0) = 0 ˆc(0, ˆt) = H(ˆt) H(ˆt ˆt1), (7) 6

---
**페이지 7**
---

where H represents the Heaviside function and the sample is injected for 0 ˆt
ˆt1. Since the stationary phase is far from equilibrium we can assume that there
is no competition for attachment sites (since many are available) and then the
extension to an arbitrary number of components is trivial Daˆci ˆt + ˆci ˆx = βi
ˆqi ˆt , ˆqi ˆt = Kaiˆci Kdiˆqi (8) with i = 1, ..., N. The concentrations and
adsorbed fractions are scaled with the inlet and equilibrium values of each
component, ˆci = ci/c0,i, ˆqi = qi/qe,i and qe,i = Kic0,i , L = u0R 2δka,1 , τ =
qe,1 ka,1c0,1 , where Ki = ka,i/kd,i and u0 = u is the inlet velocity. The
additional parameters arise due to the choice of time and length scales being
based on component i = 1 (which must then be chosen as a dominant component),
such that βi = qe,ic0,1 qe,1c0,i , Ka,i = ka,ic0,iqe,1 ka,1c0,1qe,i , Kd,i =
kd,iqe,1 ka,1c0,1 , and β1 = Ka,1 = Kd,1 = 1. In the absence of competition
between compounds this multi-component model effectively reduces to a set of
identical single component equations. In which case, it is sufficient to solve
only a single pair of equations and then the appropriate solution for each
component appears due to the different non-dimensionalisation. 2.2. Variable
velocity model Chromatography columns are very long compared to their inner
diameter. To drive the flow then requires a significant pressure drop which may
affect the gas density and thus the velocity field of the flow. This results in
a modification to (1), such that c t + x(uc) = x D c x αq t . (9) Equation (2)
remains unchanged. As noted in the studies of packed columns of [27, 33] with
large mass removal we must also track the motion of the 7

---
**페이지 8**
---

carrier fluid. Anticipating the extension to multi-components, we denote the
concentration of carrier gas molecules as cN and then u = R2 8µ p x , p =
p0cN/c0,N , (10) where p0, c0,N denote the inlet values and µ the dynamic
viscosity. The carrier gas concentration also satisfies a mass balance of the
form (9), but with zero adsorption. Boundary and initial conditions follow those
of the previous section. The full derivation of the variable velocity model is
detailed in Appendix B. A similar variable flow model has previously been
considered by Rodrıguez et al. [11]. If we distinguish the individual components
by subsript i, where i [1, N] represents different analytes and i = N the
carrier gas then the extension of the mass and momentum balance to multiple
components is ci t + ci u x + uci x = x Di ci x αqi t for i = 1, ..., N , (11)
qi t = ka,ici kd,iqi , (12) and qN = 0. The velocity and pressure for the
carrier fluid are given by equations (10). The initial and boundary conditions
are ci(x, 0) = qi(x, 0) = 0 , (13) uci Di ci x x=0+ = u0c0,i (H(t) H(t t1)) for
i = 1, ..., N 1 , (14) ucN DN cN x x=0+ = u0c0,N , (15) ci x x=L = 0 , p(L, t) =
pL , (16) where u0 = u(0, t) is the fluid velocity just before the inlet, it may
be calculated from the inlet mass flow, and pL is the pressure just before the
outlet x = L. The variables are nondimensionalised with ˆx = x L , ˆt = t τ ,
ˆci = ci c0,i , ˆqi = qi qe,i , ˆu = u u0 , ˆp = p p0 , 8

---
**페이지 9**
---

where the scales match those of the previous section. Again assuming Pe1 i =
Di/(u0L) 1, we may write the reduced dimensionless model Daci ˆt + ˆci ˆu ˆx +
ˆuˆci ˆx = βi ˆqi ˆt for i = 1, ..., N 1 , (17) ˆcN = 1/ˆu , (18) ˆqi ˆt =
Ka,iˆci Kd,iˆqi , (19) ˆp = q 1 (1 ˆp2 L) ˆx/ˆL , (20) ˆu = 1/ˆp , (21) with Da
= Rc0,1 2δqe,1 , ˆpL = pL p0 , (22) Ka,i = ka,ic0,iqe,1 ka,1c0,1qe,i , Kd,i =
kd,iqe,1 ka,1c0,1 , β = qe,ic0,i qe,1c0,1 . (23) The mass balance for the
carrier fluid has been reduced to (18), which in turn leads to the expressions
for pressure and velocity (20, 21), details are given in Appendix B: Reduced
model. The initial and boundary conditions are ˆci(ˆx, 0) = ˆqi(ˆx, 0) = 0 ,
(24) ˆci(0, ˆt) = H(ˆt) H(ˆt ˆt1) for i = 1, ..., N 1 , (25) ˆcN(0) = 1 , (26)
the last condition (26) stems from ˆu(0) = 1. As before the mass balance and
kinetic equations are uncoupled for each component and so only the solution for
a single pair is required. The velocity and pressure throughout the column only
depend on x, not on time nor any analyte concentration which simplifies the
solution. 3. Solution methods We note that Laplace transforms have been applied
previously in the literature. Guiochon and Lin [12] discuss a system of the form
(5, 6) with a 9

---
**페이지 10**
---

pulse inlet condition. Noting that the inverse transform is complicated they
reduce the inlet condition to a delta function. They go on to discuss the case
with diffusion but are forced to apply a constant concentration at the inlet.
Their solution is taken from the work of Lapidus and Amundson [13] which is
valid for a semi-infinite column. In all cases the velocity remains constant
throughout the column and only a single component is analysed. 3.1. Laplace
transform solution We now consider the system defined by equations (8) subject
to ˆci(x, 0) = 0, ˆqi(x, 0) = 0, x 0, i (27) ˆci(0, t) = H(ˆt) H(ˆt ˆt1), t 0,
i. (28) 3.1.1. Solution for the constant velocity model Taking the Laplace
transform of (8) where L{c(ˆx, ˆt)} = c(ˆx, s) and applying the initial
conditions gives, sDaci(ˆx, s) + ci ˆx + sβiqi(ˆx, s) = 0, (29) sqi(ˆx, s) =
Ka,ici Kd,iqi , (30) subject to the boundary condition ci(0, s) = 1 exp ˆt1s s .
(31) Equation (30) provides the relation, qi(ˆx, s) = Ka,i s + Kd,i ci(ˆx, s) .
(32) Substituting this into (29) gives a first order differential equation for
c, ci ˆx + s Da + βiKa,i s + Kd,i ci = 0. (33) Integrating and applying the
boundary condition determines ci(ˆx, s) = (34) exp (Daˆxs) exp (ˆt1 + Daˆx)s s
exp βiKa,iKd,iˆx s + Kdi βiKa,iˆx . 10

---
**페이지 11**
---

The inverse transform, to return to the t domain, requires use of the
convolution theorem and the following results L1 exp βiKaiKdiˆx s + Kdi = (35)
exp(Kdiˆt) "r βiKaiKdiˆx ˆt I1(2 q βiKaiKdiˆxˆt) + δ(ˆt) # , L1 ( exp (Daˆxs)
exp (ˆt1 + Daˆx)s s ) =H(ˆt Daˆx) H(ˆt Daˆx ˆt1). (36) Equation (34) may then be
transformed to ˆci(ˆx, ˆt) = exp(βiKaiˆx) Z ˆt 0 {H(ˆt T Daˆx) H(ˆt T Daˆx ˆt1)}
exp(KdiT ) " δ(T ) + r βiKaiKdiˆx T I1(2 p βiKaiKdiˆxT ) # dT . (37) The
Heaviside functions restrict the limits of integration resulting in ˆci(ˆx, t) =
Z max{0,ˆtDaˆx} max{0,ˆtDaˆxˆt1} eβiKaiˆxKdiT (38) " δ(T ) + r βiKaiKdiˆx T I1(2
p βiKaiKdiˆxT ) # dT , where I1(z) is a modified Bessels function of the first
kind. The problem of solving a coupled partial/ordinary differential equation
system has therefore been reduced to a single numerical integration. This is
easily achieved numerically. 3.1.2. Solution for the variable velocity model In
this case we take the Laplace transform of (17) and (19) and apply the initial
conditions to find sDaci(ˆx, s) + ci ˆu ˆx + ˆuci ˆx + sβiqi(ˆx, s) = 0, (39)
sqi(ˆx, s) = Ka,ici Kd,iqi. (40) 11

---
**페이지 12**
---

Equation (40) relates q to c which we then substitute into (39) ci ˆx + 1 ˆu ˆu
ˆx + s ˆu Da + βiKa,i s + Kd,i ci = 0 , (41) where ˆu is determined by equations
(21, 20). The solution is ci(s, ˆx) = B(s) ˆu exp s 3ˆuˆx Da + βiKa,i s + Kd,i ,
(42) where ˆuˆx = ˆu ˆx = 1 ˆp2 L 2ˆLˆp3 , (43) B(s) = 1 exp ˆt1s s exp s
3ˆuˆx,0 Da + βiKa,i s + Kd,i , (44) with ˆp(ˆx) given by equation (20) and
ˆuˆx,0 = (1 ˆp2 L)/(2ˆL) is the derivative of ˆu evaluated at ˆx = 0. We thus
arrive at the solution, ci(ˆx, s) = exp (Daˆωs) exp (ˆt1 + Daˆω)s ˆus exp
βiKa,iKd,iˆω s + Kdi βiKa,iˆω , (45) where ˆω(ˆx) = 1 3 1 ˆuˆx,0 1 ˆuˆx = 2ˆL (1
ˆp3) 3 (1 ˆp2 L) . (46) Again the back transform requires use of the convolution
theorem with the following results L1 exp βiKaiKdiˆω s + Kdi = exp(Kdiˆt) "r
βiKaiKdiˆω ˆt I1(2 q βiKaiKdiˆωˆt) + δ(ˆt) # , (47) L1 ( exp (Daˆωs) exp (ˆt1 +
Daˆω)s s ) = H(ˆt Daˆω) H(ˆt Daˆω ˆt1). (48) This leads to ˆci(ˆx, ˆt) = Z
max{0,ˆtDaˆω} max{0,ˆtDaˆωˆt1} eβiKaiˆωKdiT ˆu (49) " δ(T ) + r βiKaiKdiˆω T
I1(2 p βiKaiKdiˆωT ) # dT . 12

---
**페이지 13**
---

In the limit of small pressure drop the pressure may be expressed as ˆp 1 (1 ˆp2
L)ˆx/(2ˆL) and ˆω 2ˆL(1 ˆp)/(1 ˆp2 L), so ˆω ˆx. Equation (49) then reduces to
(38) and the constant velocity model is retrieved. The Bessel function I1
increases rapidly with increasing argument which can cause problems with the
numerical integration. To avoid this we note that I1(z) = z π Z π 0 exp(z cos ξ)
sin2 ξ dξ , (50) see [34, Eq. 9.6.18]. Taking the δ function outside of the
integral and replacing the Bessel function we obtain the following expression
which avoids the large function values ˆci(ˆx, ˆt) = exp (βiKaiˆω) ˆu H max{0,
ˆt Daˆω ˆt1} H max{0, ˆt Daˆω} + 2βiKaiKdiˆω πˆu Z max{0,ˆtDaˆω}
max{0,ˆtDaˆωˆt1} Z π 0 e(βiKaiˆωKdiT +2βiKaiKdiˆωT cos ξ) sin2 ξ dξdT . (51) The
solution of the PDE/ODE system is now reduced to a single integration if using
(49) or a double integration if using the more stable version (51). 3.2.
Numerical solution The fixed velocity model is specified by equations (8), for
the concentration and amount adsorbed, subject to the boundary and initial
conditions (27)-(28). The variable velocity model is specified by equations
(10)(12), subject to boundary and initial conditions (13)(16). In order to
simplify the numerical solution of the variable velocity model, we note that the
accumulation term Da1 while, if the quantity of other components is much less
than that of the carrier fluid then the diffusion of carrier fluid is negligible
(compared to advection). Thus, we use the analytical expressions for pressure
and velocity in equations (20) and (21), so only the concentration and amount
adsorbed of the test compounds must be determined numerically. Consequently, the
same scheme can be applied to both fixed and variable velocity models. Here we
employ the method of lines with finite differences for spatial discretisation.
Further, since the analytes do not interact, it is sufficient to solve for a
single pair of equations, the solutions will then differ once
re-dimensionalised. Therefore we drop the subscript i notation for this section.
13

---
**페이지 14**
---

The domain of the problem is discretised into Nx + 1 spatial nodes ({ˆxj}Nx j=0)
and Nt + 1 temporal nodes ({ˆti}Nt i=0). Let ˆci j = ˆc(ˆxj, ˆti), ˆqi j =
ˆq(ˆxj, ˆti). Then from the initial conditions we obtain, ˆc0 j = 0, ˆq0 j = 0,
j = 0, (52) and from the boundary conditions we obtain, ˆci 0 = 2ˆx H(ˆti) H(ˆti
ˆt1) + Pe1 (4ˆci 1 ci 2) 2ˆxˆu + 3Pe1 , i (53) ˆci Nx = 1 3 4ˆci Nx1 ˆci Nx2 , i
. (54) where the first-order spatial derivatives at the inlet and the outlet
have been approximated as ˆc ˆx(ˆti, ˆx0) = 3ˆci 0 + 4ˆci 1 ˆci 2 2ˆx , (55) ˆc
ˆx(ˆti, ˆxNx) = 3ˆci Nx 4ˆci Nx1 + ˆci Nx2 2ˆx . (56) For the spatial
coordinates j = 1, ..., Nx 1, we approximate the time derivatives with forward
differences and the first-order and second-order spatial derivatives with a
central difference, ˆc ˆt(ˆti, ˆxj) = ˆci+1 j ˆci j ˆt , ˆq ˆt (ˆti, ˆxj) =
ˆqi+1 j ˆqi j ˆt , (57) ˆc ˆx(ˆti, ˆxj) = ˆci j+1 ˆci j1 2ˆx , 2ˆc ˆx2(ˆti, ˆxj)
= ˆci j+1 2ˆci j + ˆci j1 ˆx2 , (58) This provides the scheme, Ri j =Kaiˆci j
Kdiˆqi j, ˆqi+1 j = ˆqi j + ˆtRi j, (59) ˆci+1 j =ˆci j ˆt Da ˆuˆx,jˆci j + ˆuj
+ Pe1Dˆx,j ˆci j+1 ˆci j1 2ˆx + (60) Pe1Dj ˆci j+1 2ˆci j + ˆci j1 ˆx2 + βRi j !
, (61) where ˆuj is the value of ˆu at ˆxj, and ˆuˆx,j is the value of the
derivative ˆuˆx = ˆu/ˆx (which can be obtained analytically) at ˆxj. Since the
diffusion 14

---
**페이지 15**
---

coefficient depends on pressure, the parameter D(ˆx) = D(ˆx)/D0 is defined as
the ratio between the diffusion coefficient at pressure ˆp(ˆx) and the diffusion
coefficient at the inlet pressure (D0). Since the expression for ˆp(ˆx) is
analytical (20), we can obtain an analytical expression for D(x) using the
equation reported in the Appendix B of the work by Rodrıguez et al. [11] (which
they attribute to Ferziger and Kaper [35]). The equation indicates that the
diffusion coefficient is inversely proportional to pressure, so if temperature
is constant, we can write D(ˆx) = 1/ˆp(ˆx). (62) Thus, Dˆx,j is the value of the
derivative Dˆx = D/ˆx at ˆxj. Note that for fixed velocity we have ˆu = 1 and D
= 1 for any value of ˆx, so equation (61) reduces to ˆci+1 j = ˆci j ˆt Da ˆci
j+1 ˆci j1 2ˆx + Pe1 ˆci j+1 2ˆci j + ˆci j1 ˆx2 + βRi j . (63) 4. Results 4.1.
Verification We begin by analysing the predictions of the two solution forms:
the Laplace solution and the numerical solution. To ensure that the models
operate within a physically realistic parameter regime, we use the parameter
values from the experiments discussed in 4.2. These correspond to the
experiments of [11, 36] involving five analytes where operating conditions are
provided in Tables 1, 3. Since the experiments involve a high-pressure drop
along the column, we have used the variable velocity solution, as given in
equation (51). 15

---
**페이지 16**
---

ˆt = 70 ˆt = 1500 ˆt = 4000 ˆt = 6500 Figure 1: Simulation using Laplace
solution (51) for the concentration of each compound throughout the column at
different times. From left to right and top to bottom: dimensionless times ˆt =
70, 1500, 4000 and 6500 (corresponding to dimensional t = 5, 108, 290 and 470
s). Parameter values are provided in Tables 1 and 3. The outlet is located at ˆL
= 10683. In Figure 1 we show the evolution of five compound concentrations along
the GC column. At ˆt = 70 (5s) the components are all gathered around the column
inlet, however by ˆt = 1500 (108s) the separation is clear: benzene travels the
fastest and is close to the column outlet while xylene is the slowest. By ˆt =
4000 (290s) both benzene and toluene have escaped, by ˆt = 6500 (470s) only o-,
p-, m-xylene remains. Note that the vertical axis decreases with time,
reflecting the fact that signals spread out and decrease in height as they move
down the column. In Figure 2 we compare solutions through the chromatogram (the
concentration measured at the outlet). In the top figure the solid lines
correspond to the Laplace solutions shown in Figure 1, and the dashed lines
correspond to the numerical solution of 3.2. As may be observed from the figure
benzene escapes the column first and so shows up at the earliest time in the
chromatogram while o-xylene only appears at around ˆt 7500. The longer the
residence time the more the signals spread out and so decrease in height. These
two sets of results clearly demonstrate the accuracy of the Laplace solution, in
comparison to the numerical one. The numerical solution includes the effects of
diffusion where Di obtained from Rodrıguez et al. [11] (leading to Pe1 i =
0.0027 0.015). The Laplace solution is an 16

*이 페이지에 4개의 이미지가 있습니다.*


---
**페이지 17**
---

approximation in that it neglects diffusion, based on the observation that the
inverse Pecl`et number is small (indicating diffusion is negligible in
comparison to advection). It has previously been shown that in adsorption
columns, which have a higher Pe1 value due to dispersion, the diffusion term may
be neglected without losing accuracy [27, 28, 29]. The close correspondence
between results verifies this approximation in GC. Consequently, henceforth we
will employ only the variable velocity Laplace solution, which is significantly
faster to compute. The bottom figure shows a comparison between the constant and
variable velocity Laplace solution, as given in the equations (38, 51). The
difference in both position and peak size is apparent. The constant velocity
model assumes that the velocity of the fluid remains constant throughout the
entire column. In reality the velocity increases by a factor of 4 due to the
pressure drop from 4 to 1 bar. The decrease of pressure at the outlet also leads
to lower concentrations, which can be observed through the smaller peak areas of
the variable velocity model. This demonstrates that for a practical system, at
least that of [11, 36], the constant velocity model may be highly inaccurate.
However, it may be appropriate for systems with a small pressure drop or much
shorter columns (for example those used in contaminant capture). 17

---
**페이지 18**
---

Figure 2: Comparison of the chromatograms obtained using different simulations.
Top: simulation using Laplace solution (51) (solid line) against numerical
solution of the full PDE system in (10) to (12) (circles). Bottom: simulation
using the Laplace solution of the variable velocity model (51) (solid line) and
the constant velocity model (38) (dashed line). The evolution of velocity and
pressure throughout the column is presented in Figure 3. The large variation in
velocity and pressure along the column clarifies the size of the error obtained
when applying the constant velocity model to the data of [11, 36]. 18

*이 페이지에 2개의 이미지가 있습니다.*


---
**페이지 19**
---

Figure 3: Evolution of velocity (left) and pressure (right) throughout the
column. Results obtained using the values in Tables 1 and 3. 4.2. Comparison
with experiment Having verified the accuracy of the variable velocity Laplace
solution against the numerical solution, we now verify the model against
experimental data. We follow the experimental studies of Nasreddine et al. [36]
and Rodrıguez et al. [11]. Their work involved five analytes: o-xylene (1);
p-xylene and m-xylene (2); ethylbenzene (3), toluene (4); benzene (5). The
carrier fluid was nitrogen and the inlet concentration was the same for all
compounds - however since p- and m-xylene act as a single component in the GC
this composite compound is assigned double the inlet concentration. Operating
conditions are provided in Table 1 for the experiment of 36, Fig. 6 (the curve
labelled 20ppb). 19

*이 페이지에 2개의 이미지가 있습니다.*


---
**페이지 20**
---

Table 1: Operating conditions of lab-scale chromatography experiments reported
by [36, 11]. Diffusion coefficients, at the inlet pressure, taken from [11].
Name Symbol Units Value Temperature T K 353.15 Inlet pressure p0 Pa 4.01105
Outlet pressure pL Pa 1.013105 Column length L m 20 Inner radius R m 9105 Wall
thickness δ m 106 Inlet flow rate Q0 m3/s 108 Inlet velocity u0 m/s 0.41 Inlet
concentration c0,i mol/m3 2.732106 (20 ppb) Injection time t1 s 4 Fluid
viscosity µ Pas 2.3105 Diffusion coef. D0,i m2/s 2.061062.89106 Certain elements
of Table 1 require clarification. The flow rate indicated in [36] is 2.5 mL/min.
This was recorded at the outlet, at atmospheric pressure. In the Table we
provide the value adjusted for the inlet which is at a much higher pressure,
0.625 mL/min. In [37] the same research group reported that the injection time
in their previous work [36] was 20 s. However, in Rodrıguez et al. [11] a more
detailed description of the injection mechanism is provided, indicating that
almost all the BTEX concentration is injected during the first 4 s. This is then
the injection time taken in our calculations. Before comparing the results with
the experimental data, we need to convert the GC units. The experimental
chromatogram of [36] has intensity units a.u. (arbitrary units). To convert
these to concentration units (mol/m3) we need the calibration of the gas
chromatograph for each component. This requires the relation between the peak
area of the chromatogram and the inlet concentration. Although Nasreddine et al.
[36] already report the calibration curves, we have calculated the
area-concentration ratio for the experiment with the conditions from Table 1.
Once this is known, we can calculate the factor between a.u. and mol/m3 by
assuming that the molecules injected are equal to the molecules eluted, namely
fic0it1u0 = uL Z tf 0 Ii(t)dt , fi = Aip0 c0,it1pL , (64) where fi
(a.u./(mol/m3)) is the conversion factor for each component i, uL the velocity
at the outlet, tf the final time of experiment (s), Ii the 20

---
**페이지 21**
---

intensity of component i in the chromatogram (a.u.), and Ai the area of the peak
of component i (a.u.s). Note that we have used the relation u0/uL = pL/p0, which
stems from equation (21). The calibration in a.u.s/ppb and the conversion factor
f are presented in Table 2. Table 2: Calibration value (peak area/inlet
concentration) and conversion factor f for the chromatography experiments
reported by Nasreddine et al. [36] with the conditions in Table 1. The analytes
with their component number are o-xylene (1), p-xylene & m-xylene (2),
ethylbenzene (3), toluene (4), benzene (5). Parameter Units Compound 1 2 3 4 5
Calibration a.u.s/ppb 25.82 35.28 28.10 46.36 57.97 f (108) a.u./(mol/m3) 1.8708
2.5557 2.0356 3.3584 4.1997 The calibration and the conversion factor is the
same for p and m-xylene (component 2), since they contribute together to the
area of the peak. Note that the values of the calibrations are very similar to
the slope of the calibration curves reported by Nasreddine et al. [36]. Once the
conversion factors are determined, we can fit the chromatogram obtained with the
model to the experimental data. Rodrıguez et al. [11] develop a numerical
solution for GC which was verified using the data of [36]. Their mathematical
model takes a similar form to the system (11, 10) but with the key difference
that the adsorption term for each component involves a summation which shows
that all compounds compete for the same adsorption sites. This term makes it
impossible to uncouple the equations and so their system must be solved
simultaneously for all compounds. Further, they require a value for the maximum
number of available sites - this is approximated using data from similar
experiments of previous authors. They note that the value obtained is several
orders of magnitude higher than the number of molecules that we are injecting in
the column and consequently any errors, even of one or two orders of magnitude
it would make no difference in the overall results. The scheme uses finite
volumes to integrate the PDE system and then an iterative scheme is applied to
fit each ka,i, kd,i to the experimental data of all components. The process of
solving simultaneously the PDE system for all five components and then iterating
for ten unknowns involves a high computational cost. In our approach we have
decoupled the equations, based on the assumption there are sufficient available
adsorption sites at all times and so the 21

---
**페이지 22**
---

competition is negligible. This is clearly the case in the latter stages of the
process when the components have separated. However, the findings of [11], that
the number of sites is orders of magnitude greater than the number of injected
molecules, vindicates this assumption and makes it clear that the competition
term is negligible. Our solution, equation (51), involves a single integration
to determine the concentration profile for all components (since they are scaled
versions of each other). Substituting xL then determines the concentration at
the outlet, which defines the chromatogram. For a given analyte we then only
need to fit two parameters to the data: there is no need to calculate the
maximum number of adsorption sites. This results in a highly efficient solution
method. In Figure 4 we show the result of matching the Laplace solution to the
experimental data [36]. Note that the experimental data has a non-zero baseline.
This background is approximately constant, so the mean value (3935.4 a.u.) has
been calculated from the flat region between component 3 and 4 (between 300 and
400 s approximately). This value has then been added to the simulation results
when fitting the experimental data. Equation (51) was integrated using a global
adaptive quadrature method. Then, the Matlab global optimisation function
GlobalSearch was used to fit the model to the experimental data. The local
solver coupled to GlobalSearch uses interior-point algorithm to minimise the sum
of squared errors (SSE). The chromatogram shows an overlap between component 2
(ethylbenzene) and 3 (toluene). However, the overlap region is small, we
neglected this small section in the data fitting. 22

---
**페이지 23**
---

Figure 4: Fitting of the variable velocity model (solid line) to the
experimental chromatograph (striped line) reported by Nasreddine et al. [36].
Results obtained using the values in Tables 1 and 3. As shown in Figure 4, the
agreement between the analytical solution, equation (51), and the experimental
data is very good. The fitting parameters to achieve this, ka,i, kd,i, are
presented in Table 3. Also shown are a number of related parameters, which may
be subsequently calculated with the aid of values provided in Table 1. With
regard to the discussion concerning goodness of fit we note that the R2 values
provided in the table are all high. The main deviations between the solutions
and data are due to the peak tailing. The causes of the tailing are discussed in
[11], where it is suggested that the culprit is the flow path disruption at the
inlet of the column. This leads to some molecules entering the column after the
initial four seconds. 23

*이 페이지에 1개의 이미지가 있습니다.*


---
**페이지 24**
---

Table 3: Physical parameters obtained from fitting the model to the experimental
data by Nasreddine et al. [36]. The only fitting parameters are ka,i and kd,i.
The rest may be calculated from ka,i, kd,i and parameters provided in Table 1.
The analytes with their component number are o-xylene (1), p-xylene & m-xylene
(2), ethylbenzene (3), toluene (4), benzene (5). Parameter Units Compound 1 2 3
4 5 ka,i (104) s1 1.0168 0.6483 0.9203 0.2953 0.0579 kd,i s1 14.291 11.057
16.940 11.502 5.299 Ki - 711.55 586.33 543.31 256.73 109.28 qe,i (103) mol/m3
1.9436 3.2032 1.4841 0.7013 0.2985 βi - 1.0000 0.8242 0.7635 0.3608 0.1530 Pe1 i
(103) - 2.6926 2.6926 3.1069 3.3968 3.7696 Da - 0.0633 L (103) m 1.8722 τ s
0.0723 SSE (103) a.u.2 0.0323 0.7044 0.1232 1.3718 6.0291 R-squared - 0.9715
0.9425 0.9642 0.9402 0.8585 5. Conclusions This works presents a novel solution
to simulate and predict gas chromatography column processes. Two different
scenarios have been considered: a first situation where pressure drop is
negligible, so no effects on the velocity field are considered, and a second
situation where pressure and velocity vary throughout the column. Although the
system study is analogous to others previously recorded in the literature the
solution technique is novel. Standard solutions use numerical techniques to
integrate the PDE system and then an iterative scheme to fit adsorption and
desorption coefficients to the experimental data of all components. The process
of solving simultaneously the PDE system for all components and then iterating
for all unknowns involves a high computational cost. In our approach the
equations are decoupled and the solution involves a single integration which
determines the concentration profile for all components (since they are scaled
versions of each other). For a given analyte we then only need to fit two
parameters to the data. This results in a highly efficient solution method. 24

---
**페이지 25**
---

Comparison of our solution against numerics and experimental data verified the
accuracy of the new approach. A comparison between the constant and variable
velocity models demonstrated the possible inaccuracy of neglecting pressure
effects. The model outputs include the adsorption coefficient, which is related
to the variance of the peaks in the chromatogram, and the equilibrium constant,
which is related to the retention time. These two are key in optimising the
chromatography process. Whilst the new approach is efficient and provides
excellent agreement with experimental data it must be viewed as a starting
point. Many chromatography processes involve heating systems, which is not
considered in our isothermal model. The impact of higher concentrations or other
types of eluents, analytes and adsorbents are also possible future extensions.
Including these aspects would further increase the value and applicability of
the methods presented in this paper. Acknowledgements First of all we would like
to thank Prof. David Mason for his tireless dedication toward the South African
Mathematics in Industry Study Groups. Many inspiring problems have arisen in
these meetings. The work described in this paper being just one of them! T.
Myers, A. Cabrera-Codony were funded by MCIN/AEI/ 10.13039/50110 0 011033/ and
by ERDF A way of making Europe, grant number PID2020- 115023RB-I00. TM
acknowledges the CERCA Programme of the Generalitat de Catalunya and the Spanish
State Research Agency (AEI), through the Severo Ochoa and Maria de Maeztu
Program for Centres and Units of Excellence in R&D (CEX2020-001084-M). ACC
acknowledges AEI for Juan de la Cierva Incorporacion fellowship
(IJC2019-038874-I). A. Valverde acknowledges support from the Margarita Salas
UPC postdoctoral grants funded by the Spanish Ministry of Universities with
European Union funds - NextGenerationEU (UNI/551/2021 UP2021-034) O. A. I.
Noreldin acknowledges financial support from the University of Zululand. 25

---
**페이지 26**
---

Appendix A. Cross-sectional averaging for constant velocity model The standard
mass balance for flow down the column is c t + u c = D2c, (A.1) where we have
assumed constant diffusivity D and velocity u = M/(πR2ρ), where M is the mass
flux. We define a cross-sectional average concentration in the mobile phase πR2c
= 2π Z R 0 rcdr. (A.2) Integrating the concentration equation over the
cross-section 2π Z R 0 c t + u c x rdr = 2πD Z R 0 2c x2 + 1 r r r c r rdr πR2 c
t + u c x = D πR2 2c x2 + 2π r c r R 0 ! . (A.3) The final term of (A.3)
represents the contribution of the diffusive flux. Due to symmetry it is zero at
r = 0, at r = R it represents the mass flux onto the stationary phase. Assuming
mass is evenly distributed within the thin stationary phase 2π RD c r R = π (R +
δ)2 R2 q t 2πRδq t (A.4) where q is the average concentration of the attached
molecules and neglecting δ/R 1. Replacing this in equation (A.3) leads to c t +
u c x = D 2c x2 2δ R q t . (A.5) Defining α = 2δ/R we arrive at Eq. (1).
Appropriate initial and boundary conditions reflect the fact there is no trace
of analyte in the column at t = 0 and then the injection occurs over a period 0
t t1, c(x, 0) = q(x, 0) = 0 , (A.6) uc D c x x=0+ = uc0 (H(t) H(t t1)) , (A.7)
26

---
**페이지 27**
---

where H is the Heaviside function. At the outlet we apply ci x x=L = 0 , (A.8)
where L is the length of the column (m). Appendix B. Derivation of variable
velocity model Appendix B.1. Dimensional model For a long thin column the flow
is primarily along the axis, such that u = v(r, x, t)ex, where ex is the unit
vector in the axial direction. Momentum conservation of the carrier fluid is
described by the Navier-Stokes equation, assuming radial symmetry and no
external forces applied, this reads µ 1 r r rv r + 2v x2 = p x , (B.1) where µ
is the dynamic viscosity of the carrier fluid (Pas) and p(x, t) is the pressure
inside the column (Pa). The boundary conditions of equation (B.1) account for
no-slip at the solid surface (r = R) and radial symmetry in the center of the
column (r = 0). Equation (B.1) is first order in p, yet it must satisfy two
pressure conditions (at the inlet p = p0, at the outlet p = pL). Here we choose
to apply the outlet condition, the inlet condition then determines the flow
rate. Hence, we apply v(R, x, t) = 0 , v r r=0 = 0 , p(L, t) = pL . (B.2) Mass
conservation for analyte follows Eq. (A.1) but the advection term is written as
(uc)x = (vc)x. Due to the vc term we cannot immediately carry out the averaging
process and must therefore first consider the reduced non-dimensional system. 27

---
**페이지 28**
---

Appendix B.2. Dimensionless model Following the non-dimensionalisation outlined
in 2.2 we have the system 1 ˆr ˆr ˆrˆv ˆr + ε2 2ˆv ˆx2 = ˆp ˆx , (B.3) Peε2 Daˆc
ˆt + (ˆvˆc) ˆx = ε2 ˆx D ˆc ˆx + 1 ˆr ˆr Dˆr ˆc ˆr , (B.4) where Pe = u0L/D0, Da
= L/(u0τ), ε = R/L and the pressure scale P = µu0L/R2. Since the diffusion
coefficient depends on pressure and temperature, we define D = D/D0 to
distinguish between the diffusion coefficient at the inlet pressure with initial
temperature conditions (D0), and the dimensionless variable function D(ˆx, ˆt).
Note that the dynamic viscosity of the fluid µ is taken as a constant regardless
of the composition of the mixture. This is based on the assumption that only
trace amounts of analyte are injected in the carrier fluid. The boundary
conditions of (B.3) are ˆv(R, x, t) = 0 , ˆv ˆr ˆr=0 = 0 , ˆp(ˆL, ˆt) = ˆpL .
(B.5) The initial and boundary conditions of equation (B.4) are ˆc(ˆx, 0) =
ˆq(ˆx, 0) = 0 , (B.6) ˆvˆc Pe1D ˆc ˆx ˆx=0+ = H(ˆt) H(ˆt ˆt1) , (B.7) ˆc ˆx
ˆx=ˆL = 0 , (B.8) D ˆc ˆr ˆr= ˆR = ε2ˆq ˆt , ˆc ˆr ˆr=0 = 0 . (B.9) Note that if
temperature is constant, D(0, ˆt) = 1. We consider now the asymptotic expansion
f = f (0) + ε2f (1) + O ε4 , where f = {ˆc, ˆq, ˆv, ˆp}. The leading order in ε2
of equation (B.3) reads 1 ˆr ˆr ˆrˆv(0) ˆr = ˆp(0) ˆx . (B.10) 28

---
**페이지 29**
---

Integrating equation (B.10) with boundary conditions in (B.5) gives ˆv(0) = ˆr2
ˆR2 4 ˆp(0) ˆx . (B.11) We define the dimensionless average velocity as ˆu = 2
ˆR2 Z ˆR 0 ˆvˆr dˆr , (B.12) which applied to (B.11) gives ˆu(0) = ˆR2 8 ˆp(0)
ˆx . (B.13) Taking the leading order in ε2 of equation (B.4) we get 1 ˆr ˆr
Dˆrˆc(0) ˆr = 0 . (B.14) Solving equation (B.14) subject to the boundary
condition on r = 0 in (B.9), we get ˆc(0) = ˆc(0)(ˆx, ˆt). The first order in
ε2, equation (B.4) reads Daˆc(0) ˆt + (ˆv(0)ˆc(0)) ˆx = Pe1 ˆx Dˆc(0) ˆx + Pe1
ˆr ˆr Dˆrˆc(1) ˆr . (B.15) Integrating equation (B.15) over the cross-section we
obtain Daˆc(0) ˆt + (ˆu(0)ˆc(0)) ˆx = Pe1 ˆx Dˆc(0) ˆx + 2Pe1D ˆR ˆc(1) ˆr ˆr=
ˆR . (B.16) The averaging affects the boundary condition at the inlet ˆu(0)ˆc(0)
Pe1Dˆc(0) ˆx ˆx=0+ = H(ˆt) H(ˆt ˆt1) . (B.17) Note that the derivative of ˆc(1)
at the boundary ˆr = ˆR in equation (B.16) must match with the sink term to
leading order. Consequently we write Dˆc(1) ˆr ˆr= ˆR = ˆq(0) ˆt . (B.18) 29

---
**페이지 30**
---

Note that in order to obtain the desired order of magnitude of ε2 in the sink
term, the radial length-scale must be defined as R = δqeL2/(c0D0τ). Now,
replacing (B.18) in (B.16) we get Daˆc(0) ˆt + (ˆu(0)ˆc(0)) ˆx = Pe1 ˆx Dˆc(0)
ˆx ˆq(0) ˆt , (B.19) where the length-scale has been defined as L =
τu0c0R/(2δqe). Once the length-scale is defined, the radial length-scale and the
pressure scale can be rewritten as R = u2 0R2c0τ/(4δqeD0) and P = 8µδqeD2 0/ (u2
0R3c0τ), and the parameter ε = u0R/(2D0). The pressure of the column to leading
order may be defined in terms of the concentration of the carrier fluid
ˆp(0)(ˆx, ˆt) = ˆp0ˆc(0) N , (B.20) where, anticipating the extension to
multiple contaminants, we denote the concentration of the carrier fluid as c(0)
N (ˆx, ˆt). As far as the sink term is concerned, as before we assume the
adsorption capacity is much greater than the concentration of analyte and write
ˆq(0) ˆt = ˆc(0) Kdˆq(0) , (B.21) where Kd = kdqe/(kac0). Appendix B.3. Reduced
Model When dealing with gases, the diffusion coefficient can reach values of the
order of magnitude of 105 m2/s [11]. Even higher values have been reported in
packed columns where dispersion acts to increase the value of D. In packed
columns the effect on the concentration has been demonstrated to be negligible
[29]. Consequently we assume that Pe1 1. Neglecting diffusion reduces the order
of equation (B.19). Then boundary condition at x = L doesnt hold, and only
(B.17), neglecting the Pe1 term, applies. To understand the flow of the carrier
fluid we write a reduced form (B.19) where Da Pe1 1 and there is no adsorption,
resulting in (ˆu(0)ˆc(0) N ) ˆx = 0 . (B.22) 30

---
**페이지 31**
---

Integrating, subject to ˆu(0) = ˆc(0) N = 1 determines (rather obviously)
ˆu(0)ˆc(0) N = 1 . (B.23) If we now replace ˆu(0) with equation (B.13), and ˆcN
with equation (B.20) we get ˆR2ˆp(0) 8ˆp0 ˆp(0) ˆx = 1 . (B.24) Integrating and
applying the boundary condition in (B.5) ˆp(ˆL, ˆt) = ˆpL we obtain ˆp(0)(ˆx) =
ˆpL s 1 + ˆp2 0 ˆp2 L 1 1 ˆx ˆL . (B.25) Note that this defines the pressure and
hence the velocity as time-independent functions, i.e. ˆp(0)(ˆx) and ˆu(0)(ˆx).
In order to provide a simpler definition of the dimensionless pressure, we
rescale pressure as p = p/p0. With the new scaling, equation (B.25) reads p(0) =
q 1 (1 p2 L) ˆx/ˆL , (B.26) and then ˆu(0) = Da1 1 p2 L / pLp(0) , (B.27) where
Da = 16µu0L/(R2pL) is the Darcy number. With the new scaling the pressure at the
outlet may be written in terms of the Darcy number 1/pL = (Da/2) 1 + 1 + 4Da2 .
(B.28) and then ˆu = 1/p. References [1] Boada E, Santos-Clotas E,
Cabrera-Codony A, Martın M, Baneras L, Gich F. The core microbiome is
responsible for volatile silicon and organic compounds degradation during anoxic
lab scale biotrickling filter performance. Science of the Total Environment.
2021;798:149162. 31

---
**페이지 32**
---

[2] Cabrera-Codony A, Ruiz B, Gil RR, Popartan LA, Santos-Clotas E, Martın M, et
al. From biocollagenic waste to efficient biogas purification: Applying circular
economy in the leather industry. Environmental Technology & Innovation.
2021;21:101229. [3] ˇSpanik I, Machyˇnakova A. Recent applications of gas
chromatography with high-resolution mass spectrometry. Journal of Separation
Science. 2018;41(1):163-79. [4] Zhou M, Yan F, Ma L, Jiang P, Wang Y, Chung SH.
Chemical speciation and soot measurements in laminar counterflow diffusion
flames of ethylene and ammonia mixtures. Fuel. 2022;308:122003. [5] Grob RL,
Barry EF. In Modern Practice of Gas Chromatography. Ch. 2 Theory of Gas
Chromatography. John Wiley & Sons, Ltd; 2004. [6] Aldaeus F, Thewalim Y, Colmsjo
A. Prediction of retention times of polycyclic aromatic hydrocarbons and
n-alkanes in temperature- programmed gas chromatography. Analytical and
bioanalytical chemistry. 2007;389:941-50. [7] Degerman M, Jakobsson N, Nilsson
B. Modeling and optimization of preparative reversed-phase liquid chromatography
for insulin purification. Journal of Chromatography A. 2007;1162(1):41-9. [8]
Kaczmarski K, Chutkowski M. Note of solving Equilibrium Dispersive model with
the Craig scheme for gradient chromatography case. Journal of Chromatography A.
2020;1629:461504. [9] Karolat B, Harynuk J. Prediction of gas chromatographic
retention time via an additive thermodynamic model. Journal of Chromatography A.
2010;1217(29):4862-7. [10] Dose EV. Simulation of gas chromatographic retention
and peak width using thermodynamic retention indexes. Analytical Chemistry.
1987;59(19):2414-9. [11] Cuevas AR, Brancher RD, Topin FR, Le Calve S, Graur I.
Numerical simulation of the sorption phenomena during the transport of VOCs
inside a capillary GC column. Chemical Engineering Science. 2021;234:116445.
[12] Guiochon G, Lin B. Modeling for preparative chromatography. 1st ed.
Academic Press; 2003. 32

---
**페이지 33**
---

[13] Lapidus L, Amundson NR. Mathematics of Adsorption in Beds. VI. The Effect
of Longitudinal Diffusion in Ion Exchange and Chromatographic Columns. J Phys
Chem. 1952;56:984-8. [14] Aris R. Diffusion and reaction in flow systems of
Turners structures. Chemical Engineering Science. 1959;10(1):80-7. [15] Golay M.
Gas Chromatographic Terms and Definitions. Nature. 1958;182:1146-7. [16] Kuˇcera
E. Contribution to the theory of chromatography: Linear non- equilibrium elution
chromatography. Journal of Chromatography A. 1965;19:237-48. [17] Grubner O,
Zikanova A, Ralek A. Statistical moments theory of gas-solid chromatography:
Diffusion controlled kinetics. Journal of Chromatography A. 1967;28:209-18. [18]
Kiyoshi Y, Terumichi N. Statistical moments in linear equilibrium
chromatography. Journal of Chromatography A. 1974;93(1):1-6. [19] Romdhane IH,
Danner RP. Polymer-solvent diffusion and equilibrium parameters by inverse
gas-liquid chromatography. AIChE Journal. 1993;39(4):625-35. [20] Lee WC, Huang
SH, Tsao GT. A unified approach for moments in chromatography. AIChE Journal.
1988;34(12):2083-7. [21] Madjar CV, Guiochon G. Experimental characterization of
elution profiles in gas chromatography using central statistical moments: Study
of the relationship between these moments and mass transfer kinetics in the
column. Journal of Chromatography A. 1977;142:61-86. [22] Pawlisch CA, Macris A,
Laurence RL. Solute diffusion in polymers. 1. The use of capillary column
inverse gas chromatography. Macromolecules. 1987;20(7):1564-78. [23] Jaulmes A,
Madjar CV, Ladurelli A, Guiochon G. Study of peak profiles in nonlinear gas
chromatography. 1. Derivation of a theoretical model. The Journal of Physical
Chemistry. 1984;88(22):5379-85. [24] Asnin L, Kaczmarski K, Guiochon G.
Empirical development of a binary adsorption isotherm based on the
single-component isotherms in the framework of a two-site model. Journal of
Chromatography A. 2007;1138(1):158-68. 33

---
**페이지 34**
---

[25] Ahmad AG, Okechi NF, Uche DU, Salaudeen AO. Numerical Simulation of
Nonlinear and Non-Isothermal Liquid Chromatography for Studying Thermal
Variations in Columns Packed with Core-Shell Particles. Journal of the Nigerian
Society of Physical Sciences. 2023:1350-0. [26] Perveen S, Khan A, Iqbal A,
Qamar S. Simulations of liquid chromatography using two-dimensional
non-equilibrium lumped kinetic model with Bi-Langmuir Isotherm. Chemical
Engineering Research and Design. 2022;181:14-26. [27] Myers TG, Font F. Mass
transfer from a fluid flowing through a porous media. International Journal of
Heat and Mass Transfer. 2020;163:120374. [28] Myers TG, Font F, Hennessy MG.
Mathematical modelling of carbon capture in a packed column by adsorption.
Applied Energy. 2020;278:115565. [29] Myers TG, Cabrera-Codony A, Valverde A. On
the development of a consistent mathematical model for adsorption in a packed
column (and why standard models fail). International Journal of Heat and Mass
Transfer. 2023;202:123660. [30] Patel H. Fixed-bed column adsorption study: a
comprehensive review. Applied Water Science. 2019;9(3):45. [31] Aguareles A,
Barrabes E, Myers TG, Valverde A. Mathematical analysis of a Sips-based model
for column adsorption. Physica D. 2023;448:133690. [32] Myers TG. Is it Time to
Move on from the Bohart-Adams Model for Column Adsorption? International
Communications in Heat and Mass Transfer. 2024;159:108062. [33] Myers TG,
Calvo-Schwarzwalder M, Font F, Valverde A. Modelling large mass removal in
adsorption columns; Submitted International Communications in Heat and Mass
Transfer, Feb. 2024. Preprint available at SSRN,
http://dx.doi.org/10.2139/ssrn.4792851. [34] Abramowitz M, Stegun IA. Handbook
of Mathematical Functions. Dover Publications; 9th Edition 1970. [35] Ferziger
JH, Kaper HG. Mathematical Theory of Transport Processes in Gases. North-Holland
Publishing Company; 1972. 34

---
**페이지 35**
---

[36] Nasreddine R, Person V, Serra CA, Le Calve S. Development of a novel
portable miniaturized GC for near real-time low level detection of BTEX. Sensors
and Actuators B: Chemical. 2016;224:159-69. [37] Lara-Lbeas I, Cuevas AR,
Andrikopoulou C, Person V, Baldas L, Colin S, et al. Sub-ppb level detection of
BTEX gaseous mixtures with a compact prototype GC equipped with a
preconcentration unit. Micromachines (Basel). 2019;10(3):187. 35