# Modeling Dynamics in Time

For introduction to some of the mathamatical formulation, please check the following:

1. [Intro to FitRec Mathamatical Formulation](https://github.com/KevinBian107/RSDB/blob/main/math/Intro%20to%20FitRec%20Mathamatical%20Formulation.pdf)
2. [Intro to Sequential Modeling](https://github.com/KevinBian107/RSDB/blob/main/math/Intro%20to%20Sequential%20Modeling.pdf)


There are usually two approaches to tackling dynamics in time: one focuses on using temporal features (timestamps), and the other focuses on looking at the order of things (sequence). We will introduce two common approaches (one from each family) in this section.

## Baseline Latent Factor (BLF)

For our baseline evaluation, we use a plain latent factor model that only models the interaction between user and item through rating interactions (\( \gamma_u \) and \( \gamma_i \)) and bias terms (user bias \( \beta_u \), item bias \( \beta_i \), and global bias \( \beta_g \)). We also added regularization on each of the terms mentioned above, which can frame the whole optimization using MSE as follows:

\[
\arg \min_{\beta, \gamma} \sum_{u,i} \left( \beta_g + \beta_u + \beta_i + \gamma_u \cdot \gamma_i - R_{u,i} \right)^2 + \lambda \left[ \sum_u \beta_u^2 + \sum_i \beta_i^2 + \sum_i \left\| \gamma_i \right\|_2^2 + \sum_u \left\| \gamma_u \right\|_2^2 \right]
\]

## Factorized Personalized Markov Chain Variants (FMPC-V)

Factorized Personalized Markov Chains (FPMC) extends from the Markov Chain model using the first-order Markovian property. It includes both user-item interactions and item-item sequential transitions and represents them as a factorized latent factor model. The core idea of FPMC is to do two things:

1. Item should match user preferences.
2. The next item should be consistent with the previous item.

To achieve this, we use the scoring function \( f(i \mid u, j) \) that represents the probability of seeing item \( i \) given the joint probability of item \( j \) and the user \( u \). We can use tensor decomposition to reduce such a probabilistic formulation into a latent factor formulation:

\[
f(i \mid u, j) = \underbrace{\gamma_{ui} \cdot \gamma_{iu}}_{f(i \mid u)} + \underbrace{\gamma_{ij} \cdot \gamma_{ji}}_{f(i \mid j)} + \underbrace{\gamma_{uj} \cdot \gamma_{ju}}_{f(u, j)}
\]

Here:
- \( f(i \mid u) \): Captures the compatibility between the user \( u \) and the next item \( i \) (user-item interaction, personalized).
- \( f(i \mid j) \): Captures the sequential relationship between the previous item \( j \) and the next item \( i \) (Markov chain).
- \( f(u, j) \): Captures the relationship between the user \( u \) and the previous item \( j \). This term is usually neglected because a user's compatibility with a previously rated item is trivial.

Our variants expand upon this system and incorporate feature components into the model. The features take many formats, but for demonstration purposes, we separate them into categorical and numerical features, which are embedded into the model as follows:

\[
\begin{aligned}
f(i \mid u, j, \mathbf{F}) = &\ 
\underbrace{\gamma_{ui} \cdot \gamma_{iu}}_{\text{user/next item compatibility}} + 
\underbrace{\gamma_{ij} \cdot \gamma_{ji}}_{\text{next/previous item compatibility}} + \underbrace{\beta_u + \beta_i}_{\text{user and next-item biases}} \\ + 
&\underbrace{\mathbf{w}^\top \mathbf{F}_{\text{cat}}}_{\text{categorical embeddings}} + \underbrace{\mathbf{v}^\top \mathbf{F}_{\text{num}}}_{\text{numerical embeddings}} + 
\underbrace{b_g}_{\text{global bias}}
\end{aligned}
\]

Traditionally, FPMC is optimized with Bayesian Personalized Ranking. However, since we are making a non-binary categorical prediction (rating), we use plain MSE for optimization, framed as follows:

\[
\begin{aligned}
\arg \min_{\gamma, \beta, \mathbf{w}, \mathbf{v}} \sum_{u,i,j} 
\Bigg(\gamma_{ui} \cdot \gamma_{iu} + \gamma_{ij} \cdot \gamma_{ji} + 
\beta_u + \beta_i + \mathbf{w}^\top \mathbf{F} + 
\mathbf{v}^\top \mathbf{F} + 
\beta_g - R_{u,i}
\Bigg)^2 \\ + 
\lambda \Bigg( \sum_u \beta_u^2 + \sum_i \beta_i^2 + \sum_u \|\gamma_u\|_2^2 + \sum_i \|\gamma_i\|_2^2 + \|\mathbf{w}\|_2^2 + \|\mathbf{v}\|_2^2 \Bigg)
\end{aligned}
\]

The features are directly embedded using embedding methods with neural networks.

## Temporal Dynamic Latent Factor Model With Neural Correlative Variants (TDLF-V)

The second approach in this system extends from the model that won the Netflix Prize. This model incorporates temporal features into a latent factor framework, specifically adding bias terms that are time-dependent (\( \beta_i(t) \)) and assuming that the user latent factors (\( \gamma_{u,k}(t) \)) change over time. The bias term includes both binning and periodic changes:

\[
\beta_i(t) = \beta_i + \beta_{i,\text{bin}}(t) + \beta_{i,\text{period}}(t)
\]

Additionally, we use a neural correlative approach, passing latent factors through a neural network. The features (\( \mathbf{w}^\top \mathbf{F} \)) are also incorporated into the neural network as follows:

\[
f(\gamma_u, \gamma_i) = \text{NN}([\gamma_u, \gamma_i])
\]

The prediction model is formulated as:

\[
\begin{aligned}
\hat{r}_{u,i,t, \mathbf{F}} = &\ 
\underbrace{\mu}_{\text{Global bias}} + 
\underbrace{\beta_i}_{\text{Static item bias}} + 
\underbrace{\beta_i(t)}_{\text{Dynamic item bias}} + \underbrace{\beta_u}_{\text{Static user bias}}
+ \underbrace{f(\gamma_{u,k}(t), \gamma_{i,k})}_{\text{Interaction score}} + 
\underbrace{\mathbf{w}^\top \mathbf{F}_{\text{item}}}_{\text{Item-specific feature effect}}
\end{aligned}
\]

The optimization process is framed as follows with regularization on each component:

\[
\begin{aligned}
\arg \min_{\alpha, \beta, \gamma, \mathbf{W}} \sum_{u,i} 
\Bigg(&
\mu + \beta_i + \beta_i(t) + \beta_u + f(\gamma_{u,k}(t), \gamma_{i,k}) + \mathbf{w}^\top \mathbf{F}_{\text{item}} - R_{u,i}
\Bigg)^2 \\
&+ 
\lambda \Bigg( \sum_u \beta_u^2 + \sum_i \beta_i^2 + \sum_u \|\gamma_u\|_2^2 + \sum_i \|\gamma_i\|_2^2 + \sum \|\mathbf{w}\|_2^2 \Bigg)
\end{aligned}
\]
