---
layout: post
title: What is going on with the T in low-rank matrix completion?
date: 2024-09-12
description: The tangent space type object in matrix completion appears across the literature. This brief exposition will hopefully explain it!
tags: matrix-completion, riemannian-geometry
categories: posts
related_posts: true
---
If you've done any reading across matrix completion literature, you've undoubtedly run into this object that mathematicians in the field like to denote as $$T$$. I'll briefly set up the problem first, introducing some standard notations before diving into what's hopefully an interesting exposition. If you're familiar with matrix completion, you can skip through the first section. Throughout this, I'll be working with square matrices, although these ideas generalize to non-square matrices quite readily.

***Intuitive introduction to matrix completion***

Imagine that you want to access an arbitrary matrix $$X\in\mathbb{R}^{n\times n}$$, but only have access to some of the entries in said matrix. This problem is quite ill-posed in general, but with the edifying assumption that $$X$$ is low-rank, you *might* have a chance of accomplishing this task. By low-rank, I mean that if $$\textrm{rank}(X)=r$$, $$r\ll n$$, far less than the ambient dimension of the matrix one wishes to recover. Intuitively, this is possible because of the sheer amount of repeated information in a high-dimensional low-rank matrix. Once you've accessed a comparably small fraction of the overall entries, you've probably seen enough information to recover said matrix. What constitutes a reasonably sized fraction and what low-rank matrices are recoverable is a different question, and not the focus of this blog post. Under an elementwise Bernoulli sampling model, just take $$p\geq C\frac{\log(n)}{n}$$ and you're likely to have enough for some matrix-specific constant $$C$$. How this $$C$$ scales is very interesting as well, but again, not the focus of this.

Let $$\Omega\subset \{1,...,n\}\times\{1,...,n\}$$ denote the subset of revealed entries in your ground truth matrix, and let $$\mathcal{R}_\Omega :\mathbb{R}^{n\times n} \to \mathbb{R}^{\vert\Omega\vert}$$ be the map that extracts entries of a matrix: that is,

$$\mathcal{R}_\Omega(X) = (X_{ij})_{(i,j)\in\Omega}$$

(Note: If you're coming to this blogpost from a paper I've written on low-rank completion for EDG, this $$\mathcal{R}_\Omega$$ is not the same! This one is following notation from <a href="https://arxiv.org/abs/0805.4471"> [1]</a>).

A first pass objective function for recovering a ground-truth matrix $X$ from your observed information might be something along the lines of minimizing the following:

$$\textrm{min}_{Y\in\mathbb{R}^{n\times n}} ~ \textrm{rank}(Y) ~\textrm{ subject to  } \mathcal{R}_\Omega(Y) = \mathcal{R}_\Omega(X)$$

This problem is NP-hard, however, and therefore not computationally feasible. Seemingly inspired from compressive sensing literature , <a href="https://arxiv.org/abs/0805.4471"> [1]</a> proposed the following objective to recover a low-rank matrix instead:

$$
\textrm{min}_{Y\in\mathbb{R}^{n\times n}} ~ \Vert Y\Vert_\star ~\textrm{ subject to } \mathcal{R}_\Omega(Y) = \mathcal{R}_\Omega(X)
$$

where $$\Vert Y \Vert_\star = \sum_{i=1}^r \sigma_i $$ is the sum of the singular values of a matrix. This is now a convex optimization procedure, and we can use the tools of convex analysis to characterize solutions to the above optimization routine.

***Duality***

In <a href="https://arxiv.org/abs/0805.4471"> [1]</a>, the authors note the following in Section 3 of their text. From above, we have a characterization of our constraint set, and we can state the following result from convex optimization theory: $Y$ is a solution to the minimization routine above if there exists a Lagrange multiplier $$\lambda\in\mathbb{R}^{\vert\Omega\vert}$$ such that

$$
\mathcal{R}^\star_\Omega \lambda \in \partial\Vert X\Vert_\star
$$

where $$\partial\Vert X\Vert_\star$$ is the subdifferential of $$\Vert X\Vert_\star$$ and $$\mathcal{R}^\star_\Omega$$ is the adjoint to $$\mathcal{R}_\Omega$$. Elements of the subdifferential, which is a set, are called subgradients, so the above equation says that $$\mathcal{R}^\star_\Omega \lambda$$ is a subgradient at $$X$$. Recall that for a function $$f:\mathbb{R}^{n\times n}\to \mathbb{R}$$, a subgradient $$Y$$ at a point $$X_0$$ is characterized by the following inequality, which must hold for all $$X$$:

$$
f(X)\geq f(X_0) + \langle Y,X-X_0\rangle
$$

where $$\langle A,B\rangle = \textrm{Tr}(A^T B) = \sum_{(i,j)\in\Omega}A_{ij}B_{ij}$$ is the standard trace inner product. Thus, the subdifferential is just the collection of objects that satisfies the above inequality. Note that if the function $$f$$ is smooth, the subdifferential would have one element in it, which would be the Euclidean gradient!

The subdifferential of the nuclear norm is as follows:

$$\partial \Vert A\Vert_\star = \left\{ B\in\mathbb{R}^{n\times n} \quad \big\vert \quad \Vert C\Vert_\star \geq  \Vert A\Vert_\star + \langle B,C-A\rangle ~\forall C\in \mathbb{R}^{n\times n} \right\}$$

***Watson's Subdifferential Characterization***

To characterize the subdifferential above, we look to <a href="https://www.sciencedirect.com/science/article/pii/0024379592904072">[2]</a>. This paper characterizes the subdifferential of orthogonally invariant norms and operator norms, with some specific results derived additionally. For simplification, <a href="https://www.sciencedirect.com/science/article/pii/0024379592904072">[2]</a> looks at unitarily invariant norms to start. Unitarily invariant norms need to satisfy a gauge function relationship, i.e. we have that $$\Vert A\Vert = \phi(\mathbf{\sigma})$$, where $$\mathbf{\sigma} = [\sigma_1,...,\sigma_n]^T$$ is a vector of singular values of $$A$$, for a positive definite gauge function $$\phi$$. Schatten p-norms are an example, as matrix norms that can be characterized by $$\Vert A\Vert = \Vert \mathbf{\sigma}\Vert_p$$, where , and $$\Vert\cdot\Vert_p$$ is an $$l_p$$ norm on $$\mathbb{R}^n$$. The nuclear norm is just the Schatten-1 norm, the spectral norm is the Schatten-$$\infty$$ norm, and the Frobenius norm is the Schatten-2 norm. As such, this is a pretty versatile class to investigate!

Before diving into the meat of <a href="https://www.sciencedirect.com/science/article/pii/0024379592904072">[2]</a>, we'll provide a couple of preliminary properties that will be used in the proof of the subdifferential. First, note a couple of properties of the subdifferential of matrix norms: $$G\in\partial \Vert A\Vert$$ is equivalent to

$$1. \Vert A\Vert = \textrm{Tr}(G^TA) \qquad
 2.\Vert G\Vert^\star \leq 1
 $$

  where $$\Vert\cdot\Vert^\star$$ is the dual norm to $$\Vert\cdot\Vert$$.

  Next, we define the polar $$\phi^\star$$ of the gauge function $$\phi$$ to be $$\phi^\star(x) = \max_{\phi(y)=1} x^T y$$. We have now that $$z\in\partial\phi(x)$$ is equivalent to

  $$1. \phi(x)=x^Tz \qquad 2. \phi^\star(z)\leq 1$$

by analogy to the matrix subdifferential property. The connection between the gauge function and the matrix norm is leveraged quite a lot, and leads to some strong results.

We can now begin the characterization of the subdifferential of an unitarily invariant norm.

**Lemma:** For $$A,R\in\mathbb{R}^{n\times n}$$, and $$A = U\Sigma V^T$$, we have that

$$\lim_{\gamma\to 0^+} \displaystyle\frac{\Vert A+\gamma R\Vert- \Vert A\Vert}{\gamma} = \max_{d\in\partial\phi(\sigma)}\sum_{i=1}^n
 d_i \mathbf{u}_i^TR \mathbf{v}_i$$


**Proof:** I'll provide a brief sketch of the proof below. Let $$\sigma_i$$ be a distinct singular value of $$A$$, or $$Av_i = \sigma_i u_i$$. If we further assume that $$A$$ smoothly depends on some parameter $$\gamma$$, then we can see then that

 $$\frac{\partial \sigma_i}{\partial \gamma} = u_i^T\frac{\partial A}{\partial \gamma} v_i$$

As such, we can use a Taylor expansion to see that, if our perturbation is of the form $$A+\gamma R$$ for an arbitrary matrix $$R$$, that $$\sigma_i(\gamma) = \sigma_i + \gamma u_i^T R v_i + o(\gamma)$$. Now, as $$\Vert A\Vert = \phi(\sigma)$$ for some gauge function $$\phi$$, we can see that for any $$d(\gamma)\in\partial \phi(\sigma(\gamma))$$ that

 $$\Vert A\Vert \geq \sigma^Td(\gamma) = \sigma(\gamma)^Td(\gamma) - \gamma \sum_{i=1}^n d_i(\gamma)u_i^TRv_i + o(\gamma) = \Vert A+\gamma R\Vert - \gamma \sum_{i=1}^n d_i(\gamma)u_i^TRv_i + o(\gamma)$$

Additionally, $$\Vert A+\gamma R\Vert = \phi(\sigma(\gamma))\geq \sigma(\gamma)^Td$$ for $$d\in\partial\phi(\sigma)$$. Rearranging these bounds and taking the limit gets us the result that we want. There's some trickiness with repeated singular values sometimes, but there are ways around this.

Now that the directional derivative has been characterized, we can use this to characterize the subdifferential as a convex hull of things in the subdifferential of the $l_p$ norm generating the Schatten-$p$ norm.

**Theorem:** For diagonal $$\textrm{diag}(\mathbf{d})=D\in\mathbb{R}^{n\times n}$$, we have that

$$\partial \Vert A\vert = \textrm{conv}\left\{ UDV^T,~A=U\Sigma V^T,~\mathbf{d}\in\partial\phi(\mathbf{\sigma}) \right\}$$

**Proof:** First denoting $$S =\left\{ UDV^T,~A=U\Sigma V^T,~\mathbf{d}\in\partial\phi(\mathbf{\sigma}) \right\}$$, we want to take an element $$G\in\textrm{conv}(S(A))$$. We have that, for $$\sum_i \lambda_i = 1, \lambda_i\geq0$$,

  $$\textrm{Tr}(G^T A) = \textrm{Tr}\left( A^T \sum_i \lambda_i U_i D_i V_i^T\right) = \sum_i \lambda_iV_i\Sigma U_i^TU_iD_i V_i^T = \sum_i \lambda_i d_i^T \sigma = \Vert A\Vert$$

  for $$d_i \in\partial\phi(\sigma)$$. This just follows from the properties that were outlined at the beginning of this section. Furthermore, we want that $$\Vert G\Vert^\star\leq 1$$. To see that this holds, we can take

  $$\Vert G\Vert^\star =\max_{\Vert R\Vert\leq 1} \textrm{Tr}(G^T R)=\max_{\Vert R\Vert\leq 1} \textrm{Tr}\left(R^T\sum_i U_i D_i V_i^T\right)$$

Additionally as

$$\Vert U_i D_i V_i^T\Vert^\star = \Vert D_i\Vert^\star = \phi^\star(d_i)=1$$

then we can see that as $$\Vert A\Vert^\star = \phi^\star(\sigma)$$, we can combine these results to see that $$\textrm{Tr}(R^T U_iD_iV_i^T)\leq \Vert R\Vert$$, giving us that $$\Vert G\Vert^\star\leq 1$$ so $$G\in\partial\Vert A\Vert$$.

To see the other way, that is $$G\in\partial \Vert A\Vert$$ but $$G\not\in\textrm{conv}(S(A))$$, we can use a standard convex separation result indicating that

$$\textrm{Tr}(R^T H)< \textrm{Tr}(R^T G)$$

 for any $$H\in S(A)$$. Now, we see that

 $$\max_{H\in S(A)} \textrm{Tr}(R^T H)< \max _{G\in\partial \Vert A\Vert}\textrm{Tr}(R^T G)$$

 which holds for any SVD

 $$\max_{d\in\partial\phi(\sigma)} \sum_i^nd_iu_i^TRV_i< \max _{G\in\partial \Vert A\Vert}\textrm{Tr}(R^T G)$$


but as the RHS is a directional derivative of a convex function $$\Vert A\Vert$$ in the direction $$R$$, the directional derivative theorem is violated, giving us the contradiction we desired.



Now we can characterize that this for the nuclear norm! This is then something we can characterize as, for a rank $$k$$ object with a thin SVD of $$A=U\Sigma V^T$$, $$U,V\in\mathbb{R}^{n\times k}$$,

$$\partial \Vert A\Vert_\star = \left\{UV^{T} + U_\perp TV_\perp^{T} \big\vert \sigma_1(T)\leq 1\right\}$$


where $$U_\perp$$ and $$V_\perp$$ consist of orthonormal vectors perpendicular to those in $$U$$ and $$V$$, respectively. We could build out a full SVD of $$A$$ as $$A=U'\Sigma'V'^T$$  

$$U' = \left[\underbrace{\mathbf{u_1}...\mathbf{u}_k}_{U} \right\vert\left.\underbrace{\mathbf{u_{k+1}}...\mathbf{u}_n}_{U_\perp} \right]$$


and the same with $$V$$, where now $$U'$$ and $$V'$$ are proper orthogonal/unitary matrices. This fully characterizes the subdifferential of nuclear norm.

To make things easier, we create the orthogonal decomposition of $$\mathbb{R}^{n\times n} = T\oplus T^\perp$$, where $$T$$ the set spanned by $$u_k x + yv_k^T$$. This has a nice projection formula too, and allows us to characterize when we're in the subdifferential of the ground truth as a projection onto $$T$$. This is given by

$$P_T X = P_U X+ X P_V - P_U XP_V$$

where $$P_U = UU^{T}$$, and similarly for $$P_V$$. As such, we have $$Y\in\partial \Vert X\Vert$$ if $$\mathcal{P}_T Y = \sum_{k=1}^r u_k v_k^T$$ and $$\Vert P_{T^\perp} Y \Vert_2\leq 1$$. Proving existence of unique minimizers is basically verifying this property.

***Characterization of the tangent space of the rank-r manifold***

At the end of <a href="https://arxiv.org/abs/0805.4471"> [1]</a>, they note that $$\textrm{dim}(T) = r(2n-r)$$, which is the same as the dimension of the set of rank-$$r$$ matrices. Well, this set happens to be a smooth manifold! Let's go through a proof to see if we can dig up a connection between the subdifferential of the nuclear norm and this manifold.

First, let's prove that the set of rank-$r$ matrices is actually a smooth manifold. This section is adapted from the wonderful text <a href="https://www.nicolasboumal.net/book/">[3]</a>.

Let

$$\mathbb{R}^{n\times n}_{r} = \left\{X\in\mathbb{R}^{n\times n} ~\vert ~\textrm{rank}(X)=r\right\}$$

 We can prove that this is a smooth manifold as follows. We can assume that

 $$X = \begin{bmatrix} X_{11} & X_{12} \\ X_{21} & X_{22}\end{bmatrix}$$


where $$X_{11}$$ is invertible, $$X_{12} \in\mathbb{R}^{r\times(n-r)}$$,$$X_{21} \in\mathbb{R}^{(n-r)\times r}$$, $$X_{22} = \mathbb{R}^{(n-r)\times (n-r)}$$. Under the assumptions, as the last $$n-r$$ columns must be linear combinations of the first $$r$$ (as its rank $$r$$), there should exist a $$W\in\mathbb{R}^{r\times (n-r)}$$ such that

$$ \begin{bmatrix}X_{12}\\ X_{22}\end{bmatrix} = \begin{bmatrix}X_{11}\\ X_{21}  \end{bmatrix}W$$


This means that we can solve this by $$W = X_{11}^{-1}X_{12}$$, indicating that $$X_{22} = X_{21}W = X_{21}X_{11}^{-1}X_{12}$$. If we let $$\mathcal{U}$$ be the open subset of $$\mathbb{R}^{n\times n}$$ consisting of invertible size $$r\times r$$ upper left submatrices, this suggests the local defining function $$h:\mathcal{U}\to\mathbb{R}^{(n-r)\times (n-r)}$$ as

$$h(X) = X_{22} - X_{21}X_{11}^{-1}X_{12}$$

As this is smooth, this is a candidate for the local defining function for a smooth manifold. Following this, we can see that the inverse image $$h^{-1}(0) = \mathbb{R}^{n\times n}_{r} \cap \mathcal{U}$$. Omitting gory details, we can see that

$$ \textrm{D}h(Y)[V] = V_{22} - V_{21}Y_{11}^{-1}Y_{12} + Y_{21}Y_{11}^{-1}V_{11}Y_{11}^{-1}Y_{12} - Y_{21}Y_{11}^{-1}V_{12}$$

 and as the image of $$\textrm{D}h(Y)$$ is $$\mathbb{R}^{(n-r)\times(n-r)}$$, we can see that the differential is surjective (set $$V_{11},~V_{21},~V_{12}$$ to zero so that the map is just $$V_{22}$$). As this is a surjective map, we have a local defining function! If the upper left submatrix isn't an $$r\times r$$ invertible matrix, we can change the local defining function at different parts of $$\mathbb{R}^{n\times n}_{r}$$ for each choice of submatrix and viola! We've constructed an embedded submanifold of $$\mathbb{R}^{n\times n}$$. To see the claim by <a href="https://arxiv.org/abs/0805.4471"> [1]</a> on the dimension, notice that

 $$\textrm{dim}(\mathbb{R}^{n\times n}_{r}) = \textrm{dim}(\mathbb{R}^{n\times n}) - \textrm{dim}(\mathbb{R}^{(n-r)\times (n-r)}) = n^2-(n-r)^2 = r(2n-r)$$


Now, the goal of this was to construct the tangent space of $$\mathbb{R}^{n\times n}_{r}$$, which we will do as follows. One could compute the tangent space by studying the kernel of $$\textrm{D}h$$ for the various $$h$$ functions that define the manifold, but this is difficult to work with for a number of reasons. As we like thin SVDs of matrices for storage, amongst other things, we will use this as a way to characterize our tangent space. Since we know the dimension of the tangent space from above, we just need to find a linear subspace of the proper dimension contained in $$T_X\mathbb{R}^{n\times n}_{r}$$ for any point $$X\in\mathbb{R}^{n\times n}_{r}$$, as we will have then found $$\mathbb{R}^{n\times n}_{r}$$. We can do this by constructing smooth curves, which should hopefully be more illuminating!

For a thin SVD of a point $$X = U\Sigma V^T$$, where $$U,V\in\textrm{St}(n,r)$$, we can try to construct curves on $$\mathbb{R}^{n\times n}_{r}$$. Let's let $U(t)$ be a smooth curve on $$\textrm{St}(n,r)$$ with $$U(0)=U$$, and similarly for $$V$$, and let $$\Sigma(t)$$ be a smooth curve on invertible $$r\times r$$ matrices (an open submanifold of $$\mathbb{R}^{r\times r})$$, with $$\Sigma(0)=\Sigma$$. As such, the curve $$c(t) = U(t)\Sigma(t)V(t)^T$$ is a smooth curve on $$\mathbb{R}^{n\times n}_{r}$$ with $$c(0)=X$$. We can compute the velocity vector at $$t=0$$ as follows:

$$ c'(0) = U'(0)\Sigma V^T + U\Sigma'(0)V^T + U\Sigma V'(0)^T\in T_X\mathbb{R}^{n\times n}_{r}$$


As $$U,V$$ are smooth curves in $$\textrm{St}(n,r)$$, we can see that the velocity vectors $$U'(0),V'(0)$$ are in the tangent space to $$\textrm{St}(n,r)$$ at $$U$$ and $$V$$ respectively. As the tangent space

$$T_X\textrm{St}(n,r) = \left\{X\Omega+X_\perp B\right\vert \left. \Omega\in\textrm{Skew}(r),~B\in\mathbb{R}^{(n-r)\times r}\right\}$$


is well-established, we can find a curve such that for any $$r\times r$$ skew-symmetric $$\Omega$$ and $$B\in\mathbb{R}^{(n-r)\times r}$$ we can take $$U'(0) = U\Omega +U_\perp B$$, where $$U_\perp$$ is such that $$[U~ U_\perp]$$ is an orthogonal matrix. We can do the same for $$V'(0) = V\Omega' + V_\perp C$$. For $$\Sigma'(0)$$, this can be any matrix $$A\in\mathbb{R}^{r\times r}$$. Now, we can combine all of this to see that

$$c'(0) = (U\Omega + U_\perp B)\Sigma V^T + UAV^T _ U\Sigma(V\Omega'+V_\perp C)^T$$

 $$ = U\underbrace{(\Omega\Sigma + A-\Sigma \Omega')}_MV^T + \underbrace{U_\perp B\Sigma}_{U_p} V^T + U\underbrace{(V_\perp C \Sigma)^T}_{V_p}$$

Since $$\Sigma$$ is invertible, $$M$$ is arbitrary, and the conditions on $$U_p$$ and $$V_p$$ amount to $$2r^2$$ constraints. As such, our total dimension amounts to $$r^2 + 2nr - 2r^2 = r(2n-r)$$. This means that we have found the whole tangent space using this curve.

Now, how can we project onto this space? It's not hard to see that the normal space, given $$X=U\Sigma V^T$$, can be computed by

$$ N_X \mathbb{R}^{n\times n}_r = \left\{U_\perp W V_\perp^T \left\vert  W\in\mathbb{R}^{(n-r)\times (n-r)}\right.\right\}$$


so an orthogonal projection onto $T_X$ should satisfy


$$Z-\textrm{Proj}_X (Z) = U_\perp W V_\perp^T$$ and $$ \textrm{Proj}_X (Z) = UMV^T + U_p V^T + UV_p^T$$


as long as $$U^TU_p = 0,~V^TV_p = 0$$. Together,

$$Z = UMV^T + U_pV^T + UV_p^T + U_\perp W V_\perp^T$$


If we set $$\mathcal{P}_U = UU^T$$, and similarly for $$V$$, we end up seeing that

$$\textrm{Proj}_X(Z) = \mathcal{P}_U Z \mathcal{P}_V + \mathcal{P}_U^\perp Z \mathcal{P}_V + \mathcal{P}_U Z \mathcal{P}_V^\perp$$


$$= \mathcal{P}_U Z + Z\mathcal{P}_V - \mathcal{P}_U Z \mathcal{P}_V$$


the same exact formula as the projection onto the space spanned by $$u_k x + yv_k^T$$, seen in the characterization of the subgradient of the nuclear norm. This indicates that these spaces are the same, which is what we expected at the beginning.


***Speculative link between these ideas***

So, what's the connection between these ideas? The nuclear norm/trace heuristic idea has been unreasonably successful for the idea of low-rank matrix completion, and further geometric ideas have been pretty effective as well. Why do these ideas seem so intertwined? I'd first speculate that the selection of $$T$$ in <a href="https://arxiv.org/abs/0805.4471">[1]</a> was in a large part a deliberate choice to attempt to match the structure with an existing object. There was a convenience to choosing an orthogonal decomposition of a space into something well-characterized, (i.e. the tangent space to $$\mathbb{R}^{n\times n}_r$$), but it also aesthetically matches neatly with the idea behind matrix completion. The notion of rank plays so strongly into this, as well as notions of incoherence (properties of the row/column spaces), that it made sense to choose this object as a reference point.

Now, the fact that the subdifferential of the nuclear norm has *most* of its components along the $UV^T$ direction hints at some of the utility of choosing the nuclear norm as a good surrogate for rank. Although there are other unitarily invariant norms that leverage a symmetric gauge function to build out a matrix from singular values, this particular one possesses such a neat decomposition into $$UV^T + U_\perp B V_\perp^T$$ for small $$B$$ that it captures much of the underlying information of the ground truth.

I think there's a lot I still don't understand about this relationship, since it feels a little deeper than a choice made from mere convenience, but I'll leave this as is for now.

***Bibliography***

<a href="https://arxiv.org/abs/0805.4471"> [1]: "Exact Matrix Completion via Convex Optimization" by Emmanuel Candès and Benjamin Recht</a>

<a href="https://www.sciencedirect.com/science/article/pii/0024379592904072">[2]: "Characterization of the subdifferential of some matrix norms" by G.A. Watson</a>

<a href="https://www.nicolasboumal.net/book/">[3]: "An Introduction to Optimization on Smooth Manifolds" by Nicolas Boumal</a>
