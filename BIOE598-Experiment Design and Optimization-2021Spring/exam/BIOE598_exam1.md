### 1. Use the bootstrap to build a null distribution and calculate a p-value.
![bootstrap_1.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go66svrogcj20nv096dh6.jpg)
![bootstrap_2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go66tgyfdmj20o40fvtbz.jpg)
![bootstrap_3.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go66umu5qxj20jr0dogm6.jpg)
![bootstrap_p.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go771afhbhj20n606hjsd.jpg)
### 2. Use and interpret the results of a t-test.
![t-test.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go679rbie2j20lv0ckgng.jpg)
statistic: the value of the t-statistic.
conf.int: a confidence interval for the mean appropriate to the specified alternative hypothesis.
estimate: the estimated mean or difference in means depending on whether it was a one-sample test or a two-sample test.
### 3. Vocab: response, predictor, factor, intercept, coefficient, effect size, parameter, residual.
![lm_1.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6ok8t087j20il073wfa.jpg)
### 4. Use effect sizes to relate changes in factor levels to changes in the response.
### 5. Use linear models for hypothesis testing.
t test. 
Multivariate linear models consider how all factors affect the response. 
### 6. Explain the meaning of interactions.
Interactions are modeled as the product of variables.
![inter_1.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6q636g89j20p0076wf2.jpg)
Higher order interactions are possible but are rare.
### 7. Calculate the number of interactions in a model with n factors.
A model with n factors has 2^n possible terms; 2^n − n − 1 of these are interactions
### 8. Explain how transformations affect the relationship between factors and response.
**Log transformation**: the model changes from an additive model to a multiplicative one.
Models with transformed responses are more difficult to interpret. There is a tradeoff between prediction and interpretation.
### 9.  Transformations: mean centering, z-scoring, rescaling to compare binary and continuous factors.
**scaling**: $y=\beta_0 +\beta_1(kx)+\epsilon\rightarrow y=\beta_0+(k\beta_1)x+\epsilon$
**mean centering**: $\bar{x}=x-mean(x)$ when intercept is uninterpretable.
$\beta_1$ remains the increase in response given a unit increase in factor; $\beta_0$ is the predicted response for an factor is on average value.
**z-scoring**: $\hat{x}=\cfrac{x-mean(x)}{stdev(x)}$
$\beta_1$ is the change in response based on an increase of one standard deviation in factor.
Why rescale by two standard deviations? 
![two_std.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6r71u4m3j20n40arwgc.jpg)
When to use scaling?
- leave binary factors unscaled
- mean center and scale continuous factors by 1 stdev.
- if continuous variables only have a few discrete values, used coded factors
- if having both binary and continuous variables, center and scale continuous factors by 2 stdev. 
### 10. Apply and interpret the results of a Box-Cox analysis.
![box_cox.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6qkbuwztj20n2098q46.jpg)
Box-Cox suggests a common transformation. 
### 11. Vocab: run, experiment, experimental unit, replicate, duplicate, background variable, effect, experimental design, confounded factors, biased factors, bias error, random error.
![crd_def.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6rdo60swj20n909f76f.jpg)
![crd_def2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6rhtgqvfj20m1097dhz.jpg)
![crd_def3.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6rj2agsqj20mq09xq5e.jpg)
### 12. Explain the differences between continuous, ordinal, and nominal factors.
![var_type.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6rlv9o83j20n705xq49.jpg)
### 13. Apply one-hot encoding to nominal factors.
![one-hot.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6rmttxkkj20mf05jaau.jpg)
### 14. Explain why degeneracy arises in models with an intercept and multilevel factors.
redundant constraints. The design matrix with an intercept is not full rank, so there could be many coefficients resulting in the same predictions and residuals.
$\beta_0-\Delta,\beta_1+\Delta,\beta_2+\Delta,...$ 
### 15. Define and interpret contrasts.
Contrast is a linear combination of variables whose coefficients add up to 0, allowing comparison of different treatment with multiple conditions.
![contrast.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6s21z6g5j20ol0hzgou.jpg)
Adjust p-value threshold (Tukey's HSD method) when testing all contrasts. 
### 16. Determine if a contrast is estimable.
- its coefficients sum to zero
- a linear combination of the rows of the design matrix
### 17. Understand and apply blocking factors.
To account for differences between each block of runs when the runs cannot be performed under the same condition. 
The blocking factor is included as a main effect in the model and adjusts the intercept for each group of runs.
![block_lim.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6sf0sm1nj20n608staf.jpg)
### 18. Explain the advantages and disadvantages of factorial designs.
A factorial design studies multiple factors at discrete intervals. It includes runs with every combination of factors set at every level.
**Pro:** 
- find better optima
![ofat_vs_fd.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6spa1yi3j20lm0ccn03.jpg)
- more efficient
- make better estimates of effect sizes
![FD_effect_size.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6sxhv93uj20jq08lmy3.jpg)
FD are nested. 
A model is solvable if the design matrix is full rank but need extra rows to estimate the model's uncertainty. To estimate the error in FD, 
![est_error.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6t7jdhnfj20my08kabm.jpg)
**Cons:** 
- the number of runs is prohibitive
- rarely need to higher-order interactions 
### 19.  Calculate the number of runs for a factorial design.
![fd_runs.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go6ssvfb0ej20m508tq43.jpg)
A factorial design in n variables has 2^n runs, but 2^(n-1) replicates at each level. Adding another replicate: OFAT is nk; FD is about k. 
### 20. Find the degrees of freedom in a model.
DoF=N-k-1, N is the number of observations, k is the number of variables
### 21. Explain and interpret half-normal plots.
No DoF to estimate all interactions. No DoF to estimate confidence intervals.
Instead, all factor levels are coded to units -1 and +1, thus the effect sizes are directly comparable. Assume **practical significance** of an effect is proportional to its magnitude.
![half_norm.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go72uqsodgj20mv0ee779.jpg)
![half_norm_step.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go72yspwmvj20mv07r75v.jpg)
### 22. Vocab: effect sparsity principle, hierarchical ordering principle, heredity principle.
**Effect sparsity principle**: only a small proportion of the factors in an experiment will have significant effects.
**Hierarchical ordering principle:** 
- lower order effects are more likely to be important than higher order effects. 
- effects of the same order are equally likely to be important. 
**heredity effect:** a model that includes an interaction should also include the corresponding main effect.
### 23. Vocab: practical and statistical significance.
**practical significance**: the magnitude of the effect size
**statistical significance**: assume effect sizes normally distributed with mean zero. The z-score of the effect sizes can be compared with a standard normal to find a p-value. It refers to whether the effect size matters or is equal to 0. 
### 24. Fractional Factorial Designs
- Use generators to derive the defining relation.
confounding: factors vary together that cannot estimate effects separately. $D=ABC;\beta_{D|ABC}=\beta_D+\beta_{ABC}\approx\beta_D$.
![ff_comf.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go7182jm7hj20nw0e075n.jpg)
Generator: $XX=I$ and $IX=X$
![define_relation.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go71aec1zhj20ng08o755.jpg)
- Use the defining relation to compute confounding structure.
![conf_struc.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go71bort73j20fh0dqmxv.jpg)
- Compute and interpret the resolution, aberration, and clarity of a design.
**Resolution:** difference in the level of confounding. The length of shortest word in the defining relation.
It measures the degree of the confounding. The resolution R design has no i-level interaction aliased with effects lower than R-i.
![reso.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go71qxjmwoj20lp07twfj.jpg)
A design with resolution R contains a full factorial design for any subset of k=R-1 factors. After the factorial experiment, drop to k factors and could re-analyze the data for all the interactions.
**Aberration:** the multiplicity of the worst confounding. The number of words with length equal to the resolution.
![abber.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go71vz4cl1j20jv02xgly.jpg)
Favor design with lower aberration, fewer main effects confounded with low-order interactions.
**Clarity:** # of confounded main effect or two-way interactions
**Clear effects:** the main effect or two-way interaction effect is clear if it is only confounded with higher order terms.
- Use foldover and mirror image designs to clear confounded factors.
![foldover.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go7346s8w9j20ph0h5acu.jpg)
![mirror_image.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go735niupkj20mt07mwfq.jpg) 
Estimate main effects clear of any two-way interactions.
### 25. Plackett-Burman Designs
- Construct PB designs for a set number of factors.
Allow run sizes in multiples fo 4 regardless of the number of factors. Have no generators of defining relation. 
![pb_design.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go73aw5xwwj20ms0bxmyg.jpg)
- Vocab: complex aliasing, hidden projection property.
**complex aliasing**: factors of PB design is partially correlated.
**hidden projection property**: the complex aliasing of PB designs allow us to fit models with main and TWI terms provided the number of terms is small.
- Explain how to fit a PB design with a linear model.
Effects are estimated for all columns with factors and interactions with complex aliasing.
- Interpret the results of subset selection.
![subset_select.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go7429qsxsj20n80f5jvh.jpg)
Be mindful of heredity effect: a model that includes an interaction should also include the corresponding main effects. 
![FD_sum.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go74bg5a4uj20l7070jsm.jpg)
### 26. Vocab: mixed-level factorial designs and Orthogonal Arrays.
Fractional factorial design and PB design use two level factors. 
Mixed level factorial design are factorial design with multi-level factors.
Orthogonal array design (OA): hand-crafted for mixtures of 2- and 3-level factors. 
Factors with > 2 levels require OA designs.
OA is similar to PB, with resolution III, no defining relation, complex aliasing, hidden project. Models with few parameters could be fit directly to the data.
### 27. Interpret the 95% CI for effects in a model.
$s.e.=\sigma/\sqrt{n}$.
95% CI for a parameter is 1.96 standard error: 
$95\%\ CI\ of\ \beta=[\beta-1.96s.e., \beta+1.96s.e.]$
A parameter estimate is significant if and only if 95% C.I. excludes zero. 
### 28. Perform power analysis (standard normal and t-test) on model coefficients.
Assume the standard deviation ($\sigma$) will not change in subsequent experiment.
The parameter estimate $\beta$ will change when new samples are added, since it's the estimate of true parameter values. 
Be more conservative in the estimate of n, adding another **0.84s.e.** to the bound ensure the 95% CI for $\beta$ excludes zero for **80% of the new estimates of $\beta$**. 
$\beta-(1.96s.e.+0.84s.e.)>0$
![power_ana.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go74zpx4s6j20n508x0u1.jpg)
### 29. Explain the limitations of power analysis.
High chance the estimate is not significant.
Given enough runs, any effect size, no matter how small, will become statistically significant. But statistical significance doesn't imply practical significance. Focus on effect size, not the p-value.
### 30. ANOVA
- Explain the decomposition of the sum of squares for a model.
$SS_{total} = SS_{explained}+SS_{residual}$
$SS_{total}=\sum_i(y_i-mean(y))^2$
$SS_{residual}=\sum_i(y_i-predicted(y_i))^2$

- Compute SStotal, SSexplained, SSresidual, and the degrees of freedom for each.
For SS_total, DoF = (# data points)-1, 1 means the mean value.
For SS_explained, DoF=# parameters
For SS_residual, DoF= (# data points) - (# parameters) -1 
- Compute the F statistic for an entire model and an individual factor.
F-statistic is the ratio between the explained variance and the residual variance after adjusting for the DoF.
For entire model, $F=\cfrac{SS_{explained}/DoF(SS_{explained})}{SS_{residual}/DoF(SS_{residual})}$
F-statistic follows F-distribution. Use F distribution convert the F-statistic into p-value.
For single factor, apply ANOVA (analysis of variance).
![anova_1.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go75i627cqj20l6094q41.jpg)
![anova_2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1go75jnbkwij20mm04y3zi.jpg)
