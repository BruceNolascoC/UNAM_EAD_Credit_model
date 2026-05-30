# Introduction to EAD Modelling in Consumer Credit

## 1. Modelling objective

In consumer credit, **Exposure at Default** (**EAD**) is the expected outstanding exposure at the moment a borrower defaults. It is one of the three core components of expected credit loss:

$$
\text{Expected Loss} = PD \times LGD \times EAD
$$

where **PD** is probability of default and **LGD** is loss given default.

In the supplied dataset, the modelling target is:

$$
\texttt{Factor} = \frac{EI}{\texttt{STOTAL at } t_0}
$$

where `STOTAL` is the accounting balance at the observation date \(t_0\), and `EI` is the exposure at default or exposure at incumplimiento. Therefore, the model is not directly predicting raw EAD. It is predicting an **EAD factor**:

$$
\widehat{EI} = \texttt{STOTAL} \times \widehat{\texttt{Factor}}
$$

This is close to an **EAD factor** or **loan-equivalent factor** formulation. For revolving credit, regulatory and academic treatments often express EAD as current drawn exposure plus a fraction of the currently undrawn amount, usually called the **Credit Conversion Factor** (**CCF**). Basel describes committed-but-undrawn exposures as exposures converted through CCFs.[^1]

## 2. Why EAD is different from PD

EAD modelling should not be treated as a simple extension of default prediction. A **PD model** asks whether the borrower will default. An **EAD model**, conditional on default, asks how much exposure will exist by the time default occurs.

This distinction matters especially for **credit cards and revolving consumer credit**, where exposure can change between the observation date and the default date. A borrower may continue drawing funds before default, while the lender may freeze the line, reduce the limit, or stop authorizations. Qi’s OCC study of unsecured credit cards frames this as a borrower-lender “race to default,” where borrowers and lenders both react as default approaches.[^3]

For the uploaded variables, this means predictors such as `DiasMora`, `utilizacion`, `STOTAL`, and the moving averages should be interpreted through the mechanics of **drawdown capacity**, **credit-line access**, **arrears progression**, and **recent behavioural change**.

## 3. Expected behaviour of the dataset variables

| Variable family | Theoretical relevance for `Factor = EI / STOTAL` |
|---|---|
| `STOTAL` | Drives raw EAD mechanically, but its effect on the factor can be negative, flat, or nonlinear. |
| `utilizacion` | Key structural variable: it proxies remaining draw capacity. |
| `DiasMora` | Captures arrears stage; expected effect is nonlinear rather than purely monotonic. |
| `df_Life_M` | Captures account seasoning and behavioural maturity. |
| `prom_saldo_*` | Measures persistent balance level versus temporary spikes. |
| `prom_diasmora_*` | Measures chronic delinquency versus recent deterioration. |
| `prom_utilizacion_*` | Measures stable line usage versus acute drawdown. |
| `mar_emp` | Likely product/channel flag; may segment accounts with different exposure dynamics. |

## 4. Balance and utilization

`STOTAL` is the current accounting balance. For raw EAD, a larger current balance generally implies larger exposure at default. However, because the target is a **ratio over current balance**, the relationship with `Factor` is not mechanically positive.

For example:

- a borrower with low `STOTAL` but a large unused line may draw more before default, producing a high `Factor`;
- a borrower already near the credit limit may have a high raw EAD but a `Factor` close to 1;
- a borrower whose exposure amortizes, pays down, or is blocked may have a `Factor` below 1.

`utilizacion` is likely more structurally informative than `STOTAL` alone because it encodes remaining credit capacity:

$$
\texttt{utilizacion} =
\frac{\text{current balance}}{\text{authorized credit line}}
$$

For revolving credit, low utilization means the borrower still has room to increase exposure before default. High utilization means the exposure is already mostly drawn, so the marginal increase before default is constrained.

This is consistent with the EAD literature. Tong et al. note that CCF modelling is difficult partly because CCF distributions can be bounded, irregular, and concentrated near endpoints. Their results suggest that direct EAD models and segmented approaches can be useful alternatives to pure CCF modelling.[^4]

## 5. Days past due and delinquency averages

`DiasMora` measures days past due at \(t_0\). In a PD model, more days past due would usually imply higher default probability. In an EAD-factor model, the expected effect is more subtle.

Early arrears may indicate liquidity stress while the borrower still has access to the credit line. This can increase exposure because the borrower may continue drawing. But advanced arrears may trigger lender interventions: line blocking, limit reduction, account cancellation, or collection procedures. In that case, additional drawdown becomes less likely.

Therefore, `DiasMora` should probably be modelled with nonlinear effects: bins, splines, tree-based splits, or interactions with utilization. A plausible behavioural pattern is:

$$
\text{low arrears} \rightarrow \text{normal use}
$$

$$
\text{moderate arrears} \rightarrow \text{stress drawdown}
$$

$$
\text{severe arrears} \rightarrow \text{blocked or saturated exposure}
$$

The moving averages `prom_diasmora_6m` and `prom_diasmora_12m` add important context. A borrower with one recent arrears event is different from a borrower with chronic delinquency. Chronic delinquency may indicate persistent weakness, but it may also indicate that the account has already exhausted available credit or entered a controlled status.

## 6. Account age and seasoning

`df_Life_M` measures months since origination. Seasoning matters because consumer-credit behaviour is often unstable early in the account lifecycle. Newer accounts may have uncertain usage patterns, less observed payment history, and greater sensitivity to initial credit-line assignment. Older accounts usually have more stable behaviour and better-observed utilization cycles.

However, the sign of `df_Life_M` is not theoretically obvious. Older accounts may have higher limits and more established usage, but they may also be more behaviourally predictable. Younger accounts may have lower balances but more volatile drawdown patterns.

The 6- and 12-month moving averages of account age may be highly collinear with current age, so they are probably less economically meaningful than the balance, arrears, and utilization averages unless they encode observation gaps, panel irregularities, or differences in historical availability.

## 7. Moving averages as behavioural state variables

The moving-average variables are theoretically valuable because EAD is a **transition quantity**: it depends on how the borrower moves from the current state to default.

For example, `prom_saldo_6m` and `prom_saldo_12m` help distinguish:

- stable high-balance revolvers;
- recent sharp balance increases;
- accounts with declining balances;
- accounts with irregular usage.

Similarly, `prom_utilizacion_6m` and `prom_utilizacion_12m` distinguish persistent saturation from sudden liquidity stress. A borrower with current high utilization but historically low utilization may be entering financial distress. A borrower with consistently high utilization may simply be a stable revolver with limited additional draw capacity.

Recent academic work on credit-card EAD has used mixture and distributional models because EAD behaviour is heterogeneous across borrower states. Wattanawongwan et al. specifically model credit-card EAD using a mixture framework and discuss the distinction between accounts that hit their credit limit before default and those that do not.[^6]

## 8. Modelling implications

For this dataset, a reasonable modelling strategy should treat `Factor` as a continuous but potentially irregular target. It may be concentrated near 1, below 1 for amortizing or decreasing exposure, and above 1 where borrowers draw more before default.

Useful candidate approaches include:

1. **Linear or GLM benchmark**  
   Useful for interpretability, but likely insufficient unless nonlinear terms are added.

2. **Segmented models**  
   Especially by utilization bands, delinquency stage, or `mar_emp`.

3. **Tree-based models**  
   Gradient boosting or random forests can capture interactions between `DiasMora`, `utilizacion`, and moving averages.

4. **Distributional models**  
   Beta, gamma, zero/one-adjusted, Tobit, or mixture models may be appropriate depending on the empirical support of `Factor`.

5. **Direct EAD versus factor comparison**  
   Literature does not universally support one formulation. Gürtler et al. compare CCF, loan-equivalent factor, EAD factor, and direct EAD approaches, showing that modelling choice can materially affect expected loss accuracy.[^5]

## 9. Summary

The uploaded variables are well aligned with the main theoretical drivers of consumer-credit EAD: current exposure, credit-line usage, delinquency stage, account seasoning, and behavioural history.

The most important modelling point is that these variables should not be interpreted exactly as in a PD model. For EAD, the key question is not only whether the client is risky, but whether the client still has the capacity and opportunity to increase exposure before default.

In this context, `utilizacion`, `STOTAL`, `DiasMora`, and their behavioural averages should be treated as state variables describing the borrower’s position before default. Their expected effects are nonlinear, conditional, and strongly dependent on whether the exposure is already saturated, still drawable, or already under lender control.

---

## References

[^1]: Basel Committee on Banking Supervision. *CRE20 — Standardised approach: individual exposures.* Bank for International Settlements. https://www.bis.org/basel_framework/chapter/CRE/20.htm

[^2]: Basel Committee on Banking Supervision. *CRE32 — IRB approach: risk components.* Bank for International Settlements. https://www.bis.org/basel_framework/chapter/CRE/32.htm

[^3]: Qi, M. *Exposure at Default of Unsecured Credit Cards.* OCC Economics Working Paper 2009-2. Office of the Comptroller of the Currency. https://www.occ.gov/publications-and-resources/publications/economics/working-papers-archived/economic-working-paper-2009-2.html

[^4]: Tong, E. N. C., Mues, C., Brown, I., & Thomas, L. C. *Exposure at default models with and without the credit conversion factor.* European Journal of Operational Research, 2016. https://www.sciencedirect.com/science/article/pii/S0377221716001004

[^5]: Gürtler, M., Hibbeln, M., & Usselmann, P. *Exposure at Default Modeling — A Theoretical and Empirical Analysis.* Journal of Banking & Finance, 2018. https://www.sciencedirect.com/science/article/abs/pii/S0378426617300547

[^6]: Wattanawongwan, S., Mues, C., Thomas, L. C., & others. *A Mixture Model for Credit Card Exposure at Default using the GAMLSS Framework.* International Journal of Forecasting, 2023. https://eprints.soton.ac.uk/454482/1/EADtoIJF_Re_revision.pdf