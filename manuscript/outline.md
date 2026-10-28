# Outline 

The one-body Green's function provides a common framework for electronic screening, quasiparticles, satellites, and charged excitations. GW and ADC are two successful but complementary ways of approximating it. GW is particularly effective for principal quasiparticle ionizations, while ADC provides a perturbative hierarchy and a useful spectral representation. However, neither approach guarantees reliable improvement when more correlation effects are included. The Perspective will examine these limitations, the connections to coupled-cluster theory, and possible directions beyond GW and low-order ADC.

**Central question:** Why do comparatively simple resummations sometimes outperform formally more complete approximations? A central possibility is that selecting and *consistently coupling* the relevant correlation contributions matters more than independently resumming increasingly large classes of diagrams. The article will distinguish three qualities that do not necessarily go together: perturbative correctness, formal consistency, and predictive accuracy.

## 1. Introduction — One Green's function, two communities

### 1.1. Why the one-body Green's function?

- Electronic structure problems in molecules and materials: ionization potentials, electron affinities, band structures, core ionizations, and photoelectron spectra.
- An added electron or hole interacts with the surrounding electrons. Its response reveals electronic correlation and may generate quasiparticles and satellites.
- The one-body Green's function describes both the main charged excitations and the redistribution of spectral weight among more complicated states.

### 1.2. Two traditions

- **Physics:** Hedin's equations, dynamical screening, GW, vertex corrections, parquet theory, and embedding.
- **Chemistry:** Electron propagators, ADC, intermediate-state representations, equation-of-motion coupled cluster (EOM-CC), and multireference methods.
- These communities study the same object but have developed different ways of organizing correlation effects.

### 1.3. Main questions

- Why are GW and ADC successful, and what distinguishes their resummations?
- How are they connected to coupled-cluster theory?
- What is the difference between Dyson and non-Dyson formulations?
- Why is improving systematically upon these approaches so difficult?
- Why can formally more complete resummations perform worse, and how should we judge the accuracy of a Green's function?

## 2. What does the one-body Green's function tell us?

### 2.1. Charged excitations and spectral information

- Introduce the Lehmann representation and the poles associated with the $N-1$ and $N+1$ electron systems.
- Explain ionization potentials, electron affinities, Dyson orbitals, and spectral weights.
- Distinguish principal quasiparticle peaks from inner-valence states, shake-up satellites, and states with strongly mixed configurations.
- Introduce the spectral function and its connection with photoelectron spectroscopy.

### 2.2. The Dyson equation and the self-energy

- Present the Dyson equation,

  $$G(\omega)=\left(G_0^{-1}(\omega)-\Sigma(\omega)\right)^{-1}.$$

- Interpret the self-energy as an effective, generally nonlocal and frequency-dependent interaction.
- Explain why the challenge is constructing a good self-energy, rather than solving the Dyson equation itself.

### 2.3. Dynamical screening

- An electron or hole polarizes its electronic environment, modifying the Coulomb interaction.
- Introduce the screened interaction $W$, static versus dynamical screening, and the role of collective electron-hole excitations.
- Connect screening to the success of GW and discuss missing exchange and self-screening effects.

### 2.4. Screening and embedding

- Explain environmental polarization around a molecule, defect, or interface.
- Relate effective screened interactions to embedding and downfolding methods, including constrained RPA and correlated-subspace approaches.
- Distinguish environmental screening from screening already treated explicitly within an active space, and explain the possibility of double counting.

### 2.5. Beyond excitation energies

- Briefly mention ground-state energies, density matrices, occupations, and spectral sum rules.
- Distinguish charged excitations in the one-body Green's function from neutral excitations obtained from two-body response functions and the Bethe–Salpeter equation.

## 3. Two complementary resummations: GW and ADC

### 3.1. GW from Hedin's equations

- Introduce Hedin's coupled equations for $G$, $\Sigma$, $W$, the polarizability, and the vertex.
- Explain the GW approximation, $\Sigma^{GW}=iGW$, and the resummation of ring diagrams through RPA screening.
- Discuss its physical transparency and practical success for principal quasiparticle energies.
- Briefly distinguish single-shot and self-consistent variants where relevant to the numerical results.

### 3.2. ADC: a perturbative hierarchy with an effective Hamiltonian

- Introduce ADC as a perturbative construction of the propagator or self-energy, reorganized into an intermediate-state matrix.
- Explain the ADC(2), ADC(3), and higher-order hierarchy, and how diagonalization resums selected contributions beyond the nominal order.
- Present $1h$, $2h1p$, $1p$, and $2p1h$ configurations and their relation to quasiparticles and satellites.
- Emphasize that perturbative order correctness does not guarantee good numerical convergence.

### 3.3. Dyson versus non-Dyson formulations

- **Dyson ADC:** The addition and removal sectors enter a common self-energy or effective supermatrix.
- **Non-Dyson ADC:** Electron attachment and ionization are constructed separately, with effective Hamiltonians and transition moments for each sector.

### 3.4. Resolvents and enlarged effective Hamiltonians

- Show how a frequency-dependent self-energy can be written as a resolvent of auxiliary configurations.
- Explain upfolding and downfolding, and why a suitable Hermitian spectral representation helps preserve real excitation energies and positive weights.
- Connect the effective-Hamiltonian description of ADC with analogous representations of GW.

### 3.5. Connections to coupled-cluster theory

- Compare ADC with IP/EA-EOM-CC: excitation spaces, ground-state correlation, matrix properties, and transition strengths.
- Explain the connection between RPA and ring-CCD through Riccati equations, while distinguishing it from an equivalence of complete GW and CC self-energies.
- Introduce particle-particle ladder/CCD connections and the importance of coupling different correlation channels within full CC approximations.
- Explain connection between ADC and UCC

## 4. Why are GW and ADC so difficult to improve?

### 4.1. GW is particularly successful for principal ionization potentials

- Explain the usefulness of GW when the main ionized state retains strong one-hole character and the dominant correlation effect is screening.
- Give representative principal-IP benchmark results for molecules and materials, specifying the GW variant and reference method.
- Stress that a small error in the main quasiparticle energy does not imply a good description of the full Green's function.

### 4.2. GW is less reliable for the rest of the spectrum

- **Inner-valence ionizations:** Strong mixing of $1h$ and $2h1p$ configurations and fragmentation of spectral weight.
- **Satellites:** Problems with both peak energies and intensities; missing scattering channels and vertex corrections.
- **Core ionizations:** Strong relaxation and shake-up effects. Distinguish main core-level binding energies, which can be accurate with suitable GW variants, from the much harder satellite spectra.
- **Electron affinities and strong correlation:** Assess their accuracy separately rather than assuming the success for principal IPs carries over.
- Include spectral comparisons, not only tables of the first IP.

### 4.3. The deceptive convergence of ADC

- Introduce the formal perturbative order $n$ and the difference between correctness through order $n$ and accuracy at the physical interaction strength.
- Use Arno's high-order calculations for Ne and Ar to show oscillatory convergence, including stronger oscillations in some augmented bases.
- Show difficult molecular examples and aggregate errors for ADC(2)–ADC(5), using the same benchmark set for comparison with EOM-CC methods.
- In Arno's 50-state benchmark, the mean absolute deviations for diagrammatic ADC(3), ADC(4), and ADC(5) are approximately **0.42, 0.40, and 0.27 eV**, compared with **0.17, 0.07, and 0.04 eV** for CCSD, CC3, and CCSDT, respectively.
- Explain why apparently accurate results at one order may reflect error cancellation rather than systematic convergence.


### 4.4. What does it mean to benchmark a Green's function?

- Most assessments focus on a few quasiparticle energies, but the Green's function also contains spectral weights, satellites, and ground-state information.
- Compare methods at increasingly demanding levels: principal IPs and EAs; inner-valence and satellite energies; spectroscopic strengths and spectral-weight redistribution; the complete spectral function and sum rules; and quantities derived from $G$, such as occupations and ground-state energies.
- Distinguish a correct pole position from an accurate residue or overall spectral shape. Assess Dyson and non-Dyson constructions by the Green's functions they reconstruct, not just by their eigenvalues.
- For comparisons of full spectra, use consistent conventions, energy references, broadening, and reference data; where possible, examine both energies and spectral weights.

### 4.5. Lessons from these limitations

- Accurate principal poles, correct perturbative orders, and physically acceptable spectral functions are different criteria.
- Increasing the order or adding selected diagrams can worsen results if important correlations remain unbalanced.
- A convincing benchmark should test the physical content of the propagator, not just the best-performing quasiparticle energy.


## 5. Beyond GW and ADC: vertex corrections, Faddeev, and parquet

### 5.1. Inner and outer vertex corrections

- Explain that vertex corrections modify both the polarizability (inner vertex) and the self-energy (outer vertex).
- Discuss the cancellation between these corrections and why treating only one can give worse quasiparticle energies.
- Use Arno's work on this balance as a concrete illustration.

### 5.2. Other higher-order self-energy approximations

- Briefly discuss SOSEX, $G3W2$, screened exchange, particle-particle $T$ matrices, electron-hole ladders, and selected positive-spectral-density constructions.
- Identify which physical contributions each method adds and which remain missing.
- Explain the tension between perturbative exactness, avoiding double counting, and preserving positive spectral weights.

**Suggested table:** Selected beyond-GW self-energy approximations, their correlation channels, second-order exactness, and spectral positivity (inspired by Table III of `3particleG.pdf`).

### 5.3. Combining scattering channels

- Introduce the direct electron-hole, transverse electron-hole, and particle-particle channels.
- Explain how GW and $T$-matrix approximations emphasize different resummations.
- Distinguish simply adding channel contributions from coupling them consistently, as in CC or parquet-inspired approaches.

### 5.4. Faddeev and parquet theory

- Introduce the parquet decomposition,

  $$F=\Lambda+\Phi^{eh}+\Phi^{\overline{eh}}+\Phi^{pp}.$$

- Explain how parquet organizes and couples different two-particle scattering channels.
- Describe the Faddeev construction as a route from a three-particle propagator to an effective self-energy Hamiltonian, connecting ADC, RPA-type resummations, and parquet.
- Summarize the simplifying assumptions used in `3particleG.pdf`, particularly separate-channel approximations and neglected overlaps.

### 5.5. Why more complete resummations can perform worse

- Use the Faddeev-ADC(3) numerical example from `3particleG.pdf`: a formally attractive resolvent can give poor ionization energies when backward amplitudes and triplet channels are treated without sufficient balance.
- Discuss the improvements obtained by limiting the triplet contributions or replacing separate ring/ladder amplitudes by CCD amplitudes.
- Make the point that **coupling correlation channels** can matter more for accuracy than merely resumming each channel to all orders.

### 5.6. Self-consistency, conservation laws, and spectral positivity

- Distinguish self-consistency from systematic improvement: iterating an approximation does not restore diagrams or channels that it omits.
- Explain that the Baym–Kadanoff notion of a conserving approximation concerns conservation laws, while causality and positive spectral weights impose different requirements.
- A positive spectral function does not guarantee conservation laws or accurate ionization energies; a conserving approximation need not yield quantitatively accurate spectra.
- Use GW and the Faddeev examples to illustrate why perturbative exactness, conservation laws, spectral positivity, and numerical accuracy must be assessed separately.

### 5.7. The remaining difficulties of parquet

- Multiple independent frequencies, large four-point objects, expensive self-consistency, and the difficulty of preserving causal spectra and consistent diagram counting.
- Static kernels simplify calculations but discard important dynamical information.
- Explain why parquet is conceptually attractive without yet providing a generally practical solution.

**Suggested table:** Formal properties of GW, vertex-corrected GW, Faddeev-ADC, CC-based variants, and parquet-inspired approximations: double counting, perturbative exactness, Hermitian/resolvent form, and coupled channels (inspired by Table VI of `3particleG.pdf`). Include conservation laws only where the property is established for the *specific implementation and self-consistency scheme*; otherwise leave the entry qualified rather than inferring it from the method's name.

## 6. Multireference Green's functions

### 6.1. Why an interacting reference is needed

- Single-reference descriptions can fail for bond breaking, near-degeneracies, open-shell systems, and strongly correlated ionic states.
- Separate static correlation in the reference from additional dynamical correlation and screening.

### 6.2. Multireference ADC

- Introduce Sokolov and co-workers' MR-ADC, based on multiconfigurational references and an effective Liouvillian.
- Explain its non-Dyson treatment of ionization and electron attachment.
- Discuss its strengths and challenges: active-space choices, density matrices, excitation spaces, perturbative convergence, and cost.

### 6.3. Multireference GW

- Present the recent multireference GW work of Wang, Fang, and Li.
- Explain how the interacting reference and screened interactions differ from conventional GW.
- Discuss active-space screening, double counting, the single-reference limit, and spectral positivity as points requiring careful assessment.

### 6.4. Open challenges

- Defining appropriate reference propagators and self-energies for interacting states.
- Combining active-space correlation with screening outside the active space without double counting.
- Developing causal, systematically improvable, and computationally feasible multireference propagators.

## 7. Conclusions and outlook

### 7.1. What future methods should achieve

- Balanced treatment of screening, exchange, and the main scattering channels.
- Reliable spectral properties, not just accurate principal ionization energies.
- Controlled perturbative or nonperturbative improvement and meaningful numerical benchmarks.
- Consistent treatment of static and dynamical correlation.
- Computationally practical representations and algorithms.

### 7.2. Promising connections

- ADC-like spectral representations for screened and vertex-corrected self-energies.
- Connections between coupled-cluster amplitudes and channel-coupled resummations.
- More tractable representations of multifrequency vertices and parquet equations.
- Multireference Hedin and effective-Liouvillian approaches.

### 7.3. Closing message

The one-body Green's function offers a unified description of electron addition and removal, but the quality of an approximation should not be judged from principal ionization potentials alone. The central challenge is to reproduce both the energies and the spectral weights of quasiparticles and satellites across different correlation regimes.

The contrast between GW, ADC, and Faddeev-type methods raises a broader question: **Why do comparatively simple resummations sometimes outperform formally more complete approximations?** ADC may be perturbatively correct yet converge irregularly; GW may predict a principal IP accurately while missing much of the spectrum; and a more elaborate, spectrally well-behaved Faddeev construction may give worse numerical results. These examples suggest that choosing the relevant correlation contributions and coupling them consistently can be more important than resumming each channel separately to higher order. Progress will require combining the physical understanding of screening and scattering with the spectral structure of ADC and the coupled treatment of correlation found in wavefunction methods.
