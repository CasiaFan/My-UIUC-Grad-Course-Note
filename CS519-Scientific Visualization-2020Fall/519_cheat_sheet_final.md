### 1. Jia’s edge filtering algorithm
- **Vertex betweenness centrality (know what the symbols in the formula mean)**
![betweeness_centrality.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm5x7kd1dj208l03ewen.jpg)
$\sigma_{st}(v)$ is the number of those paths pass through node v; $\sigma_{st}$ is total number of shortest paths from s to t.  
- **Edge betweeness centrality (know what the symbols in the formula mean).**
![betweeness_cent_edge.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm677k1zvj207102z3ym.jpg)
- **For which type of graphs will this work well** 
scale-free network: fraction of nodes with degree k follows power law $k^{-\alpha}$. Few hubs with many incident edges.
Not feasible for large networks, relying on computing all-pairs shortest paths. Approximate by random sampling. Work not well for non-power law graph. 
- **Know the steps of the algorithm** 
remove low BC edges, remain backbone. 
![edge_fil_workflow.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm6fqwzhqj20h30ajwgt.jpg)
Graph feature detection: give user ability to change the metric to preserve certain features. 
Post-process: restore discarded edges if removing it would cause disconnectivity. 
![recover_conn.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm6qnloj8j204u055jrj.jpg)
Label vertices by component. If removed edge belongs to 2 components, restore it and unify their labels. 
Edge filtering: sort BC, and remove those below the threshold.
 
### 2. Jia’s edge bundling algorithm
- **Know the algorithm**
Based on clustering labels.
![graph_aggregation.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm6xnrsp5j209r0a3mzo.jpg)
Edge bundles are cluster of similar edges.
- **How is the vertex hierarchy constructed and used**
Remove high-BC edges to discover clusters
Vertices are places radially aroung circle. Root nodes of clusters in interior, not actually exist. Leaves on the perimeter. 
Edges are B-spline curves and control points to control hierarchies. 
**Balances Hierarchy construction:** Filter edges from highest BC edges first: min(deg(a),deg(b))>1, BC(a,b)>1.
![hierarchy_construnction.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm7g1blrvj20hm042wgk.jpg)
Scan removed edges in increasing order and merge subtrees connected by those edges. If Ta and Tb have same height, merge them under same new parent tree node. If Ta is taller than Tb, Sa is the unique subtree of Ta having a and have same height as Tb. Then merge them and Ta as parent of Tb. 
![merge_commu.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm7k5ftz4j20kf05utan.jpg)
**Bundle edges:**
![bundle_edges.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm7teqn6rj208607wn0a.jpg)
Gray lines shows hierarchies; blue lines shows actual edges. 
Edge bundling to draw them as curves. Edges between 2 communities would be drawn with simiar curves.
![enron_scandal.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm83d85xxj20i50b0n32.jpg)
This method is intended for graphs with relatively few vertices that have many neighbors.

- **How is it used for community discovery** 
Compute edge betweeness centrality.
Low BC connects nodes within a community; High BC edges connect communities. Remove high BC edges to discover communities. 

### 3. Mesh Simplification
**Vertex Clustering**
- Cluster generation
  uniform 3D grid, map vertices to cluster cells.
  ![uniform_grid.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm8cy5t3ej205f05aaam.jpg)
  hierarchical approach: bottom-up or top-down.
  ![cluster_hierarchy.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm8e3a4cmj207204baap.jpg)
- compute a representative
  average/median vertex position
  median works better: a vertex of **original mesh chosest to the average position** of the vertices in the cluster. Treated as sub-sampling. 
  error quadrics: optimal. squared distance to plane (implicit form of a plane: $ax+by+cz+d=0$): 
  ![error_quadric.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm8ld9yx3j20ee08l760.jpg) (should be $c^2$ instead of $b^2$)
  Compute vertice q to multiple planes.
  ![error_quadric2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm8o1x0xbj20pf09hjvo.jpg)
  $Q_p$ is sum of all 4x4 matrixes. The storage doesn't change. 
  ![comparison.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm8v83aryj20iy0a2tde.jpg)
- mesh generation
  regenerate the mesh: connect cluster q, p if there was an edge $(p_i,q_j)$
- Topology change
  If different sheets pass through one cell. Not manifold. 
  To make manifold, use edge collapes
  ![edge_collapse.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm9150f4tj20oq06640u.jpg)
  Choose edge to remove, merge vertices into singe vertex.
  Place new vertex use error quadrics: 
  ![error_metrics.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm93mxhzhj20g804rjt2.jpg)
  ![clustering_vs_error_metrics.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glm99ul7yyj20h609a77e.jpg)


### 4. Tree Layout (know the basic algorithms)
![trees.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmbz621qkj20g208p78f.jpg)
- **Rooted:** x axis is siblings; y axis is levels. Limited scalability: unbalanced aspect ratios can occur. Information is packed in the bottom line of the triangle. Directed acyclic graph could be represented in rooted form, but may have too much crossings and bad aspect ratios for large graph. 
- **Radial**: use space better. Layout in concentric circles in particular level. scalability: 1-10k nodes. Nodes close to root get less space. 
- **Bubble**: subtree gets own circle. Variable edge lengths. Hard to distinguish node depth. Better spread nodes in large tree. 
- **Cone**: subtree gets full cones. 3D show depths. Combine bubble tree with rooted tree. 3D is tricky, occlusion. 

### 5. Force-directed layout
- **Know the FR algorithm**
Repulse force (from all vertices) and attractive force (from vertices sharing edges).
![FR_layout.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmhc8mzphj20p205etbz.jpg). k is place where fr and fa are balanced. 
![FR_detail.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmhhji6zaj20g4096aey.jpg). Apply scale factor to the amount of movement.

- **Know the running time and possible optimizations**
Compuation time: slow $O(n^3)$
  Use spatial partitioning and compute repulsion for near-by nodes.
  Use multi-level computation: use average node to represent a group of vertices.
Visual limit for large amounts of vertices: hairball problem 
Layout can be trapped in locally optimal. 

### 6. Tractography
- **Know the algorithm**
Fluid tend to diffuse along cells rather than across cells. Diffusion property can be represented by a symmetric second order tensor. 
![diffusion_matrix.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmhw1e7v2j20fn07ataf.jpg)
Trace streamline: choose seed with high **anisotropy** and stop when anisotropy is too low. 
Measure of anisotropy: 
![anisotropy.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmi04z1q4j20hz04dabd.jpg) where $\lambda$ is the eigenvalues of diffusion matrix. 

First, use Moving least squares (MLS) to filter noisy data. Gaussian would blur directional information. MLS find **low-degree polynomial** to fit data in a **small** region, then replace data value at the point value of the polynomial (like what linear regression does).
![fiber_tracking_filter.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmija323uj20850783zv.jpg). Yellow is the filtering region, small ellipsoid are interpolated tensor and the green one is the filter tensor constructed by MLS. 
- **Understand the meaning of the tensor eigenvectors/values**
3-D local axis direction of neuron fibers will be the **dominant** eigenvector. 
- **How are tensors interpolated?**
Reconstruct continuous tensor field using linear interpolation. Value of a tensor inside a voxel is a linear combination of 8 corner values. Interpolation is **tri-linear and component-wise**.
![tensor_interp.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmi3wt858j20f004rac6.jpg)
$x=x-x_{min}/x_{max}-x_{min}$
Cannot use component-wise interpolation directly on eigenvectors: a linear interpolation between 2 unit vectors is not a unit vector anymore. 

### 7. Tensors
- **What is a tensor?**
A machine takes in some vectors and spits out other vectors in linear fashion. Describe the mapping between objects, including scalars, vectors, tensors. 
![tensor.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmj1381mwj20cz061gmc.jpg)
**NOT** all matrices are tensors. Tensors have certain properties. 
![tensor_exp2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmj5b9j4qj20he09idi3.jpg)
- **Be able to construct and evaluate a Hessian matrix**
![curvature.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmjdqa3ksj20kc08t0wa.jpg)
![hessian2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmjipuzyvj20i407vq52.jpg)
f: mapping $R^3$ to $R$. 
- **Understand the meaning of the anisotropy measures**
Diffusion tensor: $D(x,s)=\partial^2f(x)/\partial s^2$, where s is the diffusion direction s. 
![curvature_pca.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmjvz1b0cj20fv0ban0g.jpg). To achieve extreme, $\partial C/\partial \alpha=0$.
![find_alpha.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmk0kgug1j20hw073juz.jpg). The 2x2 tensor gives us the principle directions in which tensor has minimal and maximal values. 

For 3x3 tensor, 3 eigenvalues and 3 eigenvectors. 
![diffusion_33.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmk4259vtj20cr023aao.jpg)
![eigenvalues_effect.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmk5mrys7j20ic06jgoi.jpg)
Mean diffusivity: $\mu=(\lambda_1+\lambda_2+\lambda_3)/3$

Look for strong difference in the eigenvalue magnitudes:
![anisotropy_form.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmk8u6m4ej20h803b0td.jpg). Large part means fiber. 
- **Color coding using eigenvalue/vectors**
Measure of alignment. 
![color_coding.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmkcqfqmmj204p02rdfy.jpg), only magnitude matters. 
- **How are tensor glyphs associated with eigenvalues/vectors**
Use hedgehog or glyphs to indicate major eigenvector. Only use glyphs where anisotropy is large enough. 
![ellipsoid_glyph.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmkkxfm9yj20hv09941l.jpg)
Superquadrics: use most often.

### 8. Sampling (know the algorithms)
- **Jittered**
![jitter.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmkt74qytj207806ndfy.jpg)
create nxn cells and randomly generate one in each cell
x-y projections could be still poorly distributed. 
**Samples: nxn**
- **Multi-jittered**
![multi-jittered.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmkyqhse8j209k09iab3.jpg)
2 grids. Coarse grid keeps jittered condition. Find grid keeps rook condition. 
**Sample: n**
- **N-Rooks**
![n-rook.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmkv095ehj208f08bmxz.jpg)
Use nxn grid and one sample exactly in each row and column
2D distribution is worse than jittered. 
**Samples: n**
- **Hammersley**
Use quasi-random sequnene. 
![hammersley_seq.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glml6ooe23j20el0b778u.jpg)
Set n samples, $p_i=(x_i,y_i)=(i/n, \Phi_2(i))$

But 1-D projection are regular; for given n, only one sequence; need to know ahead of time.
- **Halton sequence**
no need to known the number of samples.
Take radical inverse based of some prime of i along each of the dimensions: $p_i=(\Phi_2(i), \Phi_3(i), \Phi_5(i), ...)$
- **Poisson disk sampling**
Not actually a low-discrepancy sequence. Assure **min distance** between points. Pick cell size $r/\sqrt{n}$ so each grid cell only has one sample at most.
![poisson_disk.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmltwnnkqj20pi0ag7ch.jpg)
Then select the initial sample and insert 0 into active list. 
1. while active list is not empty, choose a random index i.
2. generate up to k points chosen uniformly from the region between r and 2r
3. For each point, check if it is within distance r of existing samples
4. If far enough, emit it as next sample and add to active list.
5. If no such a point after k attempts, remove from active list. 
- **What does it mean to be well-distributed?**
not structured, have some randomness.
uniform distribution, avoud gaps.
projection into 1D are aslo uniform.
non-trivial minimum distances between sample points.
- **What does low discrepancy mean?** 
![low_discrepancy.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmlkcq4juj20ob08j43j.jpg)

### 9. Numerical Methods
- **Euler’s Method**
  - Memorize the formula and know the algorithm
    A vector field: a field of velocities. 
    Solve ODEs to get solutions: $x(t)=x_0+\int v(x(u))du$. 
    No analytical solution, use Euler's method to get numerical solution. 
    ![euler_method.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmoresqb5j20hi069jtz.jpg)
  - What is the error?
    Roudning error: finite precision of floating point arithmetic
    Truncations error: approximate an infinite process with a finite number of steps. 
    They are not independent and truncation error usually dominates. 
    ![truncation_error.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmozwppi9j20o107wtc5.jpg)
    Global error: $O(h)$; Local error: $O(h^2)$. Use higher-order method. 
- **RK-4**
  - Understand what the symbols mean…
    Higher accuracy. Euler's method is the first order RK method.
    ![second_rk.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmpi10qfhj20li0arjw2.jpg), $h_n$ is step size; could vary $h_n$ by step. 
    ![rk4.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmpnz85xnj20n909u0wg.jpg)
  - Ba able to apply the algorithm if given the formula
  - What is the error?
    ![rk_accuracy.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmpqf9s6sj20k00au79q.jpg)

- **Central difference formula for a numerical derivative**
  derivative: the rate of change. 
  ![centered_difference_form.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmp9saitij20gj05hmz7.jpg)
  Appropriate when function is known. Not appropriate for sampled or noisy data. 
  Better approach for sampled data: fit an approximation function first, then take derivative of the function. 

### 10. Vector Field Visualization
- **Glyphs**
![glyph.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmrflukesj20f10bejy1.jpg)
    - more sample: more potential clutter
    - fewer sample: higher clarity
    - more line scaling: easy to see high speed region, more clutter;
    - less line scaling: less clutter, hard to perceive direction. 
![glyph_sampling2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmrlpkvbtj20g40987b6.jpg)
glyph with cones or arrows: show orientation better but take more space. Use shading (**not on lines**) to separate overlaping glyphs. 
**3D glyph**: more data, occlusion, viewpoint selection.
1. Use alpha blending to reduce occlusion. Low-speed zones: highly transparent; high-speed zones: opaque and coherent. 
2. Use Glyph on surfaces. Select certain isosurface with particular flow velocity. 

**Problems:** 
1. no interpolation in glyph space. 
   ![glyph_interp.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmryhr3xoj20cp04hmyt.jpg)
2. a glyph needs more space
3. human aren't good at visully interpolating arrows. 
4. glyph plots are sparse, while scalar plots are dense. 
- **Color coding**
  
- **Vorticity**
  ![vorticity.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glms95j8zyj20hl08p41b.jpg). rot v is othogonal to rotation plane. Magnitude tells how much spin. 
  2D vorticity: use $v(x,y)=<v_x,v_y,0>=<0,0,\partial v_y/\partial v_x-\partial v_x/\partial v_y>$
- **Divergence**
  ![divergence.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glms0t4fouj20j8057taq.jpg)
  ![divergence_mean.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glms6mbemlj20f706wq5c.jpg)
  Give impression where the flow enters and exits.

- **Stream-based visualization**
  ![stream_object.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmr66m24aj20m0095n1o.jpg)
  - Timelines
  - Streaklines
    For unsteady flow
  - **Streamlines**
    For steady flow
    ![streamlines.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmqc13vgaj20is06xmz1.jpg)
    Stream object show trajectory for longer time interval; vector glyphs is over a short time. 
    2D starts from few positions. 3D has too many streamlines and scene is cluttered. Use samplings to keep appropriate positions to start. 
    ![streamline_tracing.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmqrozfqdj20nj09ewjc.jpg)
    Use euler's method to compute location. 

    **Stream tubes:** hyperstreamlines to represent extra data with tube thickness. 
    Adjust by varying **opacity (occlusion), seeding density (coverage), integration time (continuity)**.
    ![streamtube.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmqy4qeycj20io09wn3y.jpg)
    3D stream tube: stream tubes traced from inlet to outlet, must reduce number of seeds to reduce occlusion. 

    **Stream ribbons:** how vector field twists. Choose **pairs** of close seeds, trace streamlines and construct strip by connecting closest points. 
  - **LIC (Line Integral Convolution) principle** 
    highly coherent along streamlines; highly contrasting across the streamlines. 
    ![LIC.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glmq1olsn4j20kl08pgpd.jpg)
    Take each pixel of the image, trace a streamline from upstream and downstream, blend all streamlines, pixel-wise


### 11. Classification and Segmentation
![classification_rules.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnfk7g30nj20lj08yn29.jpg)
  ![neighbour8.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnfj1mjnij204o049jrn.jpg) 8-connectivity.
- **Graph cuts algorithm**
  Similarity matrix. 
  Idea: break graph into segments by deleting links. Break links that have low cost (low similarity). Similar pixels should be in the same segment; dissimilar pixels should be in different segment. 
  Not use all edges (not feasible). Use **neighboring pixels** (4 or 8) with direct links.
  ![pixel_affinity.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glney522pqj20mu06ddk4.jpg)
  f could be location distance, color, intensity, texture.
  ![cut_graph.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnf8vyqfrj20o204476n.jpg) The sum of edge weights. Find minimum cut. 

  ![source_sink_w.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnhh6fpbrj20nf0d3dnh.jpg)
  ![edge_w.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnhpi9nq2j20ov0c043o.jpg)
  ![edge_w2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnhv51921j20nl0a0jx3.jpg)
- **Thresholding**
  Intensity of a pixel: $i(x)=(r+g+b)/3$. Good to segment single object. 
  Problem: same object has different intensity at different locations in an image. 
  Solution: local thresholding. 
  ![local_thresholding.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnftb9zw3j20j2084n0z.jpg)
  Computing threshold:
  1. expect contrast: $T=i_{avg}+\epsilon$. Use histogram analysis. 
  ![histogram_analysis.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glnfy36fwxj20nq09uq7p.jpg). Filter outliers in pre-processing.

  Thresholding + connected component analysis for multiple object segmentation. 
  ![component_connection.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glngjmwehzj207b04egmx.jpg)
  Recursive region growing to do connected component analysis. 
  ![recursive_region.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glngrk4k2oj20ea078ady.jpg)

- **Otsu’s method**
  ![otsu_method.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glng0grtgej20gz06076z.jpg)
  $Var(X)=\sum p_i (x_i-\mu)^2$

  It's equivalent to maximizing the gap:
  ![max_gap.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glng5cfjdkj20nf0axgpy.jpg)

  ![otsu_algorithm.png](http://ww1.sinaimg.cn/large/8f5d6442ly1glng9udig7j20ol0b2n32.jpg)

  For multipke object segementing, use Expectation Maximization (EM) to fit GMM.

- **Understand what a transfer function is and its purpose**
Transfer function: map the information from every point along the ray to a color and opacity. A TF defines (1) which parts of the data are essential to depict and (2) how to depict these, often small, portions of the volumetric data.
6 categories: 1D data-based, gradient 2D, curvature-based, size-based, texture-based, and distance-based.
Traditional attributes of volum data: scalar value (density), gradient magnitude, and object/label ID (in the case of segmented data).
Different segmented objects can have different TFs, different rendering modes (such as DVR or MIP), and different compositing modes. The latter capability enables two-level volume rendering, which comprises one local compositing mode per object, and a second global compositing level that combines the contributions of different objects.
## Formulas
Memorize the following
- Euler’s method
- Halton sequence
- Hammersley sequence
- Central difference formula

Given the following formulas, understand what the symbols mean:
- Betweeness centrality
- Quadric Error metric
    - Memorize how to compute optimal point
    - Memorize how to add quadrics together
- Tensor anisotropy measures
- RK-4
- LIC
- Vorticity
- Divergence 
- Gradient