# References

Master bibliography for the paper. Each section's prose cites entries from this list using inline `[Author Year]` keys. The per-section `References` blocks in the existing drafts (`02_background.md`, `03_related_work.md`) are redundant convenience copies; this file is the source of truth and will replace them at the integration pass (task 10.12).

## Citation conventions

- **Inline keys** look like `[Author Year]` (no comma, no parentheses). For three or more authors, the key uses the first surname only (e.g., `[Ouyang 2022]` for the full InstructGPT author list).
- **Same-author, same-year** disambiguates with a lowercase letter suffix: `[p-e-w 2025]` for the main Heretic repo, `[p-e-w 2025b]` for the Arbitrary-Rank Ablation pull request.
- **Software, model cards, GitHub issues** use the maintainer handle or organization name as the author key: `[elder-plinius 2025]`, `[TrevorS 2026]`, `[Heretic Issue 265]`. This keeps community artifacts and academic papers in one alphabetical list.
- **Course-aligned mathematical references** (Mirsky, Hoffman-Wielandt, Jacot, Vershynin) are cited from §4 (Mathematical Framework) and §9 (Discussion); the canonical lecture-to-result mapping lives in `docs/findings/course-material-mapping.md`.

## Entries (alphabetical by citation key)

- **[Arditi 2024]** Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. *Refusal in Language Models is Mediated by a Single Direction.* arXiv:2406.11717, June 2024. <https://arxiv.org/abs/2406.11717>

- **[Bai 2022]** Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073, December 2022. <https://arxiv.org/abs/2212.08073>

- **[Christiano 2017]** Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D. *Deep Reinforcement Learning from Human Preferences.* arXiv:1706.03741, June 2017. <https://arxiv.org/abs/1706.03741>

- **[Cui 2024]** Cui, J., Chiang, W.-L., Stoica, I., Hsieh, C.-J. *OR-Bench: An Over-Refusal Benchmark for Large Language Models.* arXiv:2405.20947, May 2024; ICML 2025. <https://arxiv.org/abs/2405.20947>

- **[elder-plinius 2025]** elder-plinius. *OBLITERATUS: Gemma-4-E4B-it-OBLITERATED model card and toolkit.* HuggingFace + GitHub, 2025. <https://huggingface.co/OBLITERATUS/gemma-4-E4B-it-OBLITERATED> ; <https://github.com/elder-plinius/OBLITERATUS>

- **[Gemma Team 2025]** Gemma Team (Google DeepMind). *Gemma 3 Technical Report.* arXiv:2503.19786, March 2025. <https://arxiv.org/abs/2503.19786>

- **[grimjim 2025]** Lai, J. (grimjim). *Norm-Preserving Biprojected Abliteration.* HuggingFace blog, November 2025. <https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration>

- **[HauhauCS 2025]** HauhauCS. *Gemma-4-E4B-Uncensored-HauhauCS-Aggressive.* HuggingFace model card, 2025. <https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive>

- **[Heretic Issue 265]** *Error when loading gemma-4-31b-it model.* GitHub issue, `p-e-w/heretic#265`, 2025. <https://github.com/p-e-w/heretic/issues/265>

- **[Hoffman-Wielandt 1953]** Hoffman, A. J., Wielandt, H. W. *The variation of the spectrum of a normal matrix.* Duke Mathematical Journal, 20(1): 37–39, 1953. The original eigenvalue perturbation bound for normal matrices; §4 notes that the rectangular `o_proj` / `down_proj` weights fall outside its scope and that Mirsky's singular-value generalization is required instead.

- **[Jacot 2018]** Jacot, A., Gabriel, F., Hongler, C. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS 2018; arXiv:1806.07572. <https://arxiv.org/abs/1806.07572>. Source for the lazy-training / NTK linearization regime invoked in §4 (Lec 8–9 connection) and §9 (Discussion); cited as the *plausibility* basis for partial first-order behavioural response under a rank-1 weight perturbation, not as a quantitative predictor of the 40.5% partial-refusal finding.

- **[Mirsky 1960]** Mirsky, L. *Symmetric gauge functions and unitarily invariant norms.* Quarterly Journal of Mathematics, 11(1): 50–59, 1960. Source for the singular-value perturbation bound `Σᵢ (σᵢ(W+E) − σᵢ(W))² ≤ ‖E‖_F²` invoked throughout §4. The rank-1 specialization used in the paper — `‖E‖_F = ‖E‖_2 = |α|·‖dᵀW‖_2` for `E = α·d·(dᵀW)` with unit `d` — is an immediate consequence of `E` having a single non-zero singular value.

- **[Mlabonne 2024]** Labonne, M. *Uncensor any LLM with abliteration.* HuggingFace blog, 2024. <https://huggingface.co/blog/mlabonne/abliteration>

- **[Ouyang 2022]** Ouyang, L., Wu, J., Jiang, X., et al. *Training Language Models to Follow Instructions with Human Feedback (InstructGPT).* arXiv:2203.02155, March 2022. <https://arxiv.org/abs/2203.02155>

- **[Park 2023]** Park, K., Choe, Y. J., Veitch, V. *The Linear Representation Hypothesis and the Geometry of Large Language Models.* arXiv:2311.03658, November 2023; ICML 2024. <https://arxiv.org/abs/2311.03658>

- **[p-e-w 2025]** p-e-w. *Heretic: Fully automatic censorship removal for language models.* GitHub, 2025. <https://github.com/p-e-w/heretic>

- **[p-e-w 2025b]** p-e-w. *Arbitrary-Rank Ablation (ARA), Pull Request #211.* GitHub, `p-e-w/heretic`, 2025. <https://github.com/p-e-w/heretic/pull/211>

- **[Rafailov 2023]** Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290, May 2023. <https://arxiv.org/abs/2305.18290>

- **[Röttger 2024]** Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., Hovy, D. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* arXiv:2308.01263, August 2023; NAACL 2024. <https://arxiv.org/abs/2308.01263>

- **[TrevorS 2026]** TrevorS. *gemma-4-abliteration: Gemma 4 abliteration research — biprojection + EGA for E2B, E4B, 26B MoE, 31B.* GitHub, 2026. <https://github.com/TrevorS/gemma-4-abliteration>

- **[Vershynin 2018]** Vershynin, R. *High-Dimensional Probability: An Introduction with Applications in Data Science.* Cambridge University Press, 2018. Reference text for the Lec 5 / Lec 6 material on volume concentration of the n-ball / n-cube, near-orthogonality of random unit vectors in ℝᵈ, and the matrix-norm concentration inequalities that ground the §4 Mathematical Framework. Used in the §4 "near-orthogonality of learned conceptual directions" claim (Lec 5 connection) and in the §9 over-parametrization framing.

- **[Zou 2023]** Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405, October 2023. <https://arxiv.org/abs/2310.01405>

## Maintenance

When a new prose task (10.2, 10.5, 10.6, 10.7, 10.8, 10.10, 10.11) introduces a new citation, the writing subagent SHALL add the entry to this file as part of the same commit. The integration pass (task 10.12) confirms that every `[Author Year]` key in `paper/sections/*.md` resolves to an entry here and removes the per-section duplicate `References` blocks in `02_background.md` and `03_related_work.md`. The sources audit (task 10.15, `paper/sources.md`) is a different artifact: it maps every **numeric claim** to a source CSV / JSON path + commit hash; it does not duplicate this bibliography.
