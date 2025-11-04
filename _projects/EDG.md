---
layout: page
title: Euclidean Distance Geometry
description: Point cloud recovery from partial pairwise distances
img: assets/img/cow.jpg
importance: 1
category: work
related_publications: Smith2023,Smith2024
---

In the applied sciences, a common measurement to make is the distance between objects, such as the distance between sensors in a network or the distance between atoms in a protein. A collection of distances contains a lot of powerful information; namely, if in a point cloud of $$n$$ objects in $$d$$ dimensions I know every pairwise distance between points, I can reconstruct the underlying object (up to translation/rotation.) The problem becomes considerably more interesting with access to only a few distances.

More mathematically, consider a set of vectors $$\{\mathbf{p}_k\}_{k=1}^n \subset \mathbb{R}^d$$. From these vectors we can construct a matrix $$\mathbf{P} = [\mathbf{p}_1 ... \mathbf{p}_n]^T\in\mathbb{R}^{n\times d}$$. We seek to recover this matrix from partial access to entries of the squared distance matrix $$\mathbf{D} = [d_{ij}^2]= \Vert \mathbf{p}_i - \mathbf{p}_j\Vert_2^2$$.

This problem has a rich, well-studied history that I won't cover in this blurb (see Euclidean Distance Geometry by Liberti and Lavor for an excellent introduction). I'll provide a brief explanation as to my approach with Dr. Abiy Tasissa to study this problem.

The first idea that one familiar with the compressive sensing/matrix completion literature of the 2010's would be to try and apply low-rank matrix completion directly on the squared distance matrix $$\mathbf{D}$$; after all, if one has many more points than dimensions (i.e. $$n \gg k$$) then $$\mathbf{D}$$ would be low-rank as $$\textrm{rank}(\mathbf{D})\leq k+2$$. This idea makes sense, but the only real issue is that distance matrices are notoriously finicky. Not only must the diagonals be zero, but more importantly the entries must satisfy a triangle inequality, which is a difficult condition to enforce.

This leads one to consider other ways to complete the point cloud from the partial distances. In (Tasissa, Lai 2018), the authors consider translating the problem into a non-orthogonal completion over the Gram matrix $$\mathbf{X} = \mathbf{PP}^T$$, still just relying on entries of $$\mathbf{D}$$, and provided nice guarantees for recovery when minimizing the nuclear norm of $$\mathbf{X}$$. This convex problem is slow, however, and non-convex algorithms provide better scalability.

One prior that can be leveraged in ths problem is the fact that the target dimension $$k$$ is often known in practice: either 2 or 3. The set of matrices of a fixed rank forms a manifold, so one can consider minimizing a distance functional on that manifold to solve the problem. This approach is outlined in our paper titled "Riemannian Optimization for Non-convex Euclidean Distance Geometry with Global Recovery Guarantees", found on arXiv or on this website. In order to solve this problem, one needs some statistical properties of an EDG-specific sampling operator to hold: much of the technical work in this paper is dedicated to showing that certain operators satisfy what is known as the *Restricted Isometry Property*.

Further extensions of this work can be done, as current state-of-the-art only assumes uniform at random sampling. We're in the drafting stages of a paper that describes how certain non-uniform sampling patterns are optimal for ill-conditioned data, allowing superior numerical performance over uniform sampling.

``
