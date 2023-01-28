# CS519 Cheat Sheet
### 1) CIE XYZ and xyY color spaces
**CIE XYZ color space (Week 1: Perceptual Color Spaces - 15:30)**
- no negative values
- separate luminance from chromaticity

RGB to XYZ:
![rgb_to_xyz.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfyvcp7k7j20bu03agmh.jpg)

$Y$ corresonds to **brightness**. 

**CIE xyY color space (Week 1: Perceptual Color Spaces-18:10)**
- normalized chromaticity values in [0,1]
- no change luminance
Convertion: 
![XYZ_to_xyY.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfz01rvepj20hm02l0tb.jpg)

CIE xy chromaticity diagram:
![cie_xy_chromaticity.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfz1o5j37j209209sn06.jpg)

### 2) Gamma correction  (know that gamma means raising a value to some exponent, and correction is raising a value to the reciprocal)
**Week 1-Gamma Correction**
In CRT disaplays, $V_{display}=V_{signal}^\gamma$, where $\gamma$ varied by display, but **2.2** was a typical value ==> displayed colors were darker than input color. 

So **Gamma Correction** is: $V_{signal}^{1/\gamma}$

- LCD-LEDs don't use gamma. 
- sRGB standard uses gamma.

### 3) Phong and Blinn-Phong reflection models 
**Phong reflection model: (Week 2-Shading-8:20)**

![phong_reflection_model.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfyfjiqvij20bz078jt4.jpg)
![phong_model_formula.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfyh4u054j20js08ujvl.jpg)
ka is the reflectance of material; i is the intensity;

Specular reflection:
![specular_reflect.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjhqvr9ukxj20go08lq6e.jpg)

Diffuse reflection: 
![diffuse_reflect.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjhqy39a4zj20ju05l77c.jpg)

**Blinn-Phong Refrelection model: (Week 2-Shading-20:30)**
Replace the $(V.R)^a$ term with $(N.H)^b$ term where H is the halfway vector.  
- More efficient. 
- Use **higher** $b>a$ make output similar to Phong with a

$H=\cfrac{L+V}{\|L+V\|}$
![half_way_vector.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfysarec9j20aw07hdgw.jpg)

### 4) Linear and Bilinear interpolation (memorize how to do it)
**Linear Interpolation (Week 3-Linear interpolation-10:00)**
2 points: $p_0$ and $p_1$ where $f(p_0)=v_0$ and $f(p_1)=v_1$, then $f(t)=(1-t)v_0+tv_1$ and the t is: $t=\cfrac{dist(p_i,p_0)}{dist(p_1, p_0)}$

**Bilinear Interpolation (Week 3-Linear interpolation-13:00)**
![bilinear_interp.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfzfjyhrgj208x089gml.jpg)
![bilinear_interp_formula.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjfzic1ebej20cb06sq46.jpg)
where $Q_{11}=(x_1,y_1), Q_{12}=(x_1,y_2), Q_{21}=(x_2,y_1), Q_{22}=(x_2,y_2)$

### 5) Barycentric coordinates 
**Week 3-Barycentric Coordinates and interpolation-4:00**
describe location of a point $p$ in relation to the vertices of a given triangle. 
$p=(\lambda_1, \lambda_2, \lambda_3)$ where
$p=\lambda_1a+\lambda_2b+\lambda_3c$ and 
$\lambda_1+\lambda_2+\lambda_3=1$

**Interpolation:**
$f(p)=\lambda_1f(a)+\lambda_2f(b)+\lambda_3f(c)$
![barycentric_coord.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg359sy6oj20p20bggqx.jpg)
where $A$ means area. 

### 6) Structured grid representations of domains
**Week 4-Meshes and Elements-10:00**
Structured grid:
- uniform grid: need $m$ integers for \#vertices along each of the $m$ dimensions and 2 corrner points.
![uniform_grid.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg3n2k5l1j20ab030glu.jpg)
- rectilinear grids: all cells have same type but can have different sizes sharing along axes; need $\sum_{i=1}^{m}d_i$ floats (#vertices along each axis) and 1 corrner points
![rectilinear_grid.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg3sboz2oj20e2056jty.jpg)
- Curvilinear grids: all cells have same shape and cell vertex coordinates are **freely** specified; need $\prod_{i=1}^{m}d_i$ floats (coordinates of all vertices) and 1 for each axis (#vertices along each axis)
![curvilinear_grid.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg5bd4tqsj20er04w76t.jpg)

### 7) Half-edge data structure
**Week 4-Data structures for polygonal meshes-13:00**
Data structure to save ploygon mesh coordinates: simple and efficient traversal of vertex **neighbourhoods**.
![halfedge.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg7e96ieoj20ma07i776.jpg)
prev and opposite data could not be stored since they could be inferred. 
![halfedge_2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg7f49p22j20oo0afwk2.jpg)

Data storage: $vertex+face+halfedge$
- vertex: x,y,z, halfhedge ref ==> 4 \* 4 bytes/vertex
- face: 4 bytes/face ==> 2\*4 bytes/vertex
- halfedge: 3\*4 bytes/halfedge ==> since $E\approx 3V$, $HalfE\approx 6V$, then 6\*3\*4 bytes/vertex
Euler Characteristic
![euler_charact.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjhr5un97uj20g006z403.jpg)

Traverse the mesh in **counterclock** around the face using halfedge **next** reference and **vertex** reference for each halfedge.

Traverse the vertices on the face by one-ring traversal:
![halfedge_2.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg7f49p22j20oo0afwk2.jpg)

### 8) Colormap construction 
**Week 2-Colormap**
Color table: pre-compute colors and store colors 
color mapping function: N colors and index them into [0, N-1]
$i=min(\lfloor \cfrac{x-x_{min}}{\cfrac{x_{max}-x_{min}}{N}}\rfloor, N-1)$, get the floor value.

**Transfer functions:** define colors at certain scalar values (knots), then use interpolation to define colors between knots.  
![transfer_function_cm.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgtfj1ceej20nh05kaca.jpg)

rainbow color map is not linear. 

diverging colormap: have breakpoint in the middle.

Colormap design advice: 
![cm_desing.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgtjlgok5j20lr0aa42u.jpg)
### 9) Munzner's data taxonomy
**Week 3-A Data Taxonomy**
Data types: structure or mathematical interpretation of data
- item: individual entity.
- attribute: property of item
- links: relationship between items
- positions: spatial data / pixel data
- grids: sampling strategy for continuous data

![data_taxonomy.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgu3d5t93j20og0o3jxr.jpg)

### 10) DEM to TIN conversion process
**Week 4-Terrain Generation for Geographic information systems-**
DEM: digital elevation model
TIN: triangular irregular network
LIDAR points --> triangulating --> TIN --> interpolation --> raster DEM.
- compute TIN in places where all points have already arrived
- raster, output & deallocate
- keep parts that miss neighbourhoods
  
Delaunay Triangulation
![delaunary_triangl.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg8jw9nmqj20mp07m0wb.jpg)
Incremental point insertion in delaunay triangulation
![incremental_pt_insert.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg8l0f8rrj20ki0ah43e.jpg)
Spatial finalization of points
![finalization.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg8nf8awgj20j00db7bp.jpg)
### 11) Compositing with the over operator
**Week 6-Compositing-5:00**
![over_operator.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgu6esjhfj20i10ahdia.jpg)
Order matters.
### 12) Volume rendering using ray-casting
**Week 6-Ray casting-2:00**
![volumn_render_ray_cast.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgudtkikvj20hu0ahgo7.jpg)
![discrete_approximation.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgugow9hcj20fh08bn0c.jpg)
I is total illumination, q is the active emission of light at a point. TF is the transfer function, A is the opacity value generated by TF. 

### 13) Marching squares and cubes
**Marching squares: (Week 3-Marching Squares-9:00)**
For creating contour line vertices (x,y)
- assume the underlying, continuous functions is **linear** on the grid edge
- linear interpolation
- v is the isovalue
![marching_square.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg3dhb92kj20he069myg.jpg)
Encode the state of cell vertices in a **4-bit** id.
![marching_square_vertices_encode.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjg3fn60cgj20ks0mggsy.jpg)

---

**Marching cubes: (Week 5-marching cubes: algorithm)**
bipolar edges: edge with two differently classified endpoints

place polygon vertices on the edges, $w=(1-t)p+tq$, where $t=\cfrac{f_0-s_p}{s_q-s_p}$
![marching_cube.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgvrlwo4uj208q07b3yv.jpg)

8 bits to represent cube configuration:
if using **complementary** (swap positive and negative) and **rotational symmetry**, there are 15 cases.
![marching_cube_cases.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgvut5oedj20fr070n1d.jpg)

Steps:
- classify vertices of a cube and generate bitcode
- read isosurface lookup table using bit code
- retrieve triangles
- compute vertex coordinates using linear interpolation
- store triangles

To keep consistency (face ambiguity) and correctness, use rotational symmetry only (23 cases). 
![marching_cube_no_comp.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgw0qw96cj20dr09zdjl.jpg)

face ambiguity: 
![face_ambiguity.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjhyrqk2k7j20gf06ddi2.jpg)
align edges on the surface between cubes. 

internal ambiguity: 
![internal_amb.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgw2bljktj20nn0780uv.jpg)

To determine correctness, use trilinear interpolation.
![trilinear_interp.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgw5t47isj20ot09v7ak.jpg)
### 14) Dual marching squares
**Week 5-Dual marching squares**
dual contouring places isosurface vertices inside mesh elements. Isosurfaces vertices in **adjecent elements** are with edges. 
![dual_mc.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgwdjnl25j20io05kjsj.jpg)
dual marching square cell configurations
![dual_mc_cases.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgwf4xoubj20jb0a3tch.jpg)
No ambiguity

### 15) Scattered data interpolation using RBFs 
**Week 3-Scattered Data Interpolation-9:00**
Radial Basis Function: function dependent on distance from a center is radial. 
radial function: $\phi(x,p)=\phi(\|x-p\|)$
intepolation function: $f(x)\approx \sum_{i=1}{N}w_i\phi(x,pi)$
![rbf_kernel.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgzaad6pcj20e604g3zd.jpg)
compute weights: 
![rbf_weights.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjgzc9wmtij20ab0ehwhm.jpg)

$w=A^{-1}p$