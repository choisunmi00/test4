---
layout: single
title: LPF(Supplenetary Data)(1)
categories: LPF
tags: paper
---

# 논문 리뷰: LPF - Supplementary Data

[LPF: a framework for exploring the wing color pattern formation of ladybird beetles in Python](https://doi.org/10.1093/bioinformatics/btad430)  
[Supplementary Data](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/39/7/10.1093_bioinformatics_btad430/2/btad430_supplementary_data.pdf?Expires=1736057160&Signature=ynQSiSKNANknwTVvSYi5cTxcCp9gHsKwCX1ply7Ys0QzNZ0TQilTTt0xPhjEJXNwXfgbw3M-Rk7jPuxqt9JicwdLO-RS1wej0LarlSNwLrT3~I8iV8b6YrDm-UhelBilpi5P4lgoq-s9mn8M-89FfFf0LIn7ZiE6cPwYsdqRvdMP3CQNw0wocNRet~9qnUMM25qWD7wD68GOKpbxIEBE9eOiGfDq33m6tzWM3fUMTuMXqL8YAm53YsL54Dw8ZPb0T7UZ62vRyCSCmn-vGKRjrcD65lM-C1fJ1krF9hUAAFRdFHgmHiDG~YRJjy9Jn-AaroBOVcLBPGZDkBGki3nocg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

---
## 1. Introduction
- Main features of LPF
1. Reaction-diffusion PDE models and numerical methods.
2. GPU acceleration of the PDE solvers for a batch of parameter sets.
3. Visualization of the wing color patterns of various morphs.
4. Evolutionary search for discovering the parameter sets of a PDE model.
5. Diploid models for analyzing population evolution in crossing experiments.
             
## 2. Numerical sumulation
### 2.1 Basic workflow
1. 모델 파일들로부터 Conditions, Initial states, Parameters로 Mini batch 구성
2. Initializer 생성 -> Initial states
3. Parameter sets 생성 -> Parameters
4. PDE model 생성
5. numerical simulation 수행: Model / PDE solver / CPU, GPU 
6. 결과 시각화  

- [tutorial01_solve_single_parameset](https://github.com/cxinsys/lpf/blob/main/tutorials/tutorial01_solve_single_paramset.ipynb)  

**2.1.1**  Configuring a simulation experiment  
- ``dt``: the time setp size
- ``n_iters``: the number of iterations in the numerical solution
- 2D 공간에서의 RD 시스템
- ``width``, ``height``: detirmined the size of 2D space

<img src="https://github.com/user-attachments/assets/47faa67a-4249-485a-a40e-19c3f86a420a" width="40%" height="40%"/>   

- ``dx``: determines both sapce step sizes of the SD cartesian space (∆x = ∆y)   
 
<img src="https://github.com/user-attachments/assets/ddd9ae3b-c612-443a-b163-93067ec55bd7" width="60%" height="60%"/> 

- os.makedirs: output directory 경로 정의. timestamp 기반 생성

- ``u0``, ``v0``: initial values
- ``Du``, ``Dv``: diffusion parameters   
- ``ru``, ``rv``, ``su``, ``sv``, ``k``, ``mu``: kinetic parameters   
- ``param_dict``: single parameter set
- ``model_dict``: determines the batch size
- 모델 파일(JSON 형식)을 불러와 사전 정의된 모델 설정 가능

**2.1.2**  Creating an initializer   
- 초기화 프로그램 생성   
```python
from lpf.initializers import LiawInitializer

initializer = LiawInitializer()
initializer.update(model_dicts)
params = LiawModel.parse_params(model_dicts)
```
- 초기화 클래스 제공   
- ``LiawInitializer``: 2D공간에서 사용자 정의 위치에 대해서만 $u$를 $u_0$으로 초기화, 모든 $v$를 $v_0$으로 초기화   
- ``TwoComponentConstantInitializer``: 2D 공간에서 $u$와 $v$의 모든 점에 $u_0$, $v_0$을 할당   

**2.1.3**  Creating an array of parameters sets   
- 매개변수 집합의 배열 생성     
```python
from lpf.models import LiawModel

params = LiawModel.parse_params(model_dicts)
```
- parse_params 정의   
```python
import numpy as np
from lpf.models import TwoComponentModel

class LiawModel(TwoComponentModel):

	@staticmethod
	def parse_params(model_dicts):
		"""Parse the parameters from the model dictionaries.
			A model knows how to parse its parameters.
		"""
		batch_size = len(model_dicts)
		params = np.zeros((batch_size, 8), dtype=np.float64)

		for index, n2v in enumerate(model_dicts):
			params[index, 0] = n2v["Du"]
			params[index, 1] = n2v["Dv"]
			params[index, 2] = n2v["ru"]
			params[index, 3] = n2v["rv"]
			params[index, 4] = n2v["k"]
			params[index, 5] = n2v["su"]
			params[index, 6] = n2v["sv"]
			params[index, 7] = n2v["mu"]

		return params
```   
- ```LiawModel```: 변수 값들을 분석, 넘파이 배열 생성.   
- ```parmas```: ```LiawModel```에서 반환된 값   
- ```parse_params```: ```LiawModel```에서 static method로, 분석하지 않고 배열 생성 가능   

**2.1.4**  Creating a PDE model       
- LPF에 정의된 PDE model   

|        Model        |        Class        |     Reactions     |     Parameters     |
|:-------------------:|:-------------------:|:-----------------:|:------------------:|
| Gierer-Meinhardt    | ```GiererMeinhardtModel```|    $f(u, v) = \rho_u \frac{u^2}{v} - \mu u$ <br> $g(u, v) = \rho_v u^2 - \nu v$   |       $ρ_u$: ``ru``, $ρ_v$: ``rv``,  $µ$: ``mu``, $ν$: ``nu``     |
| Gray-Scott          | ```GrayScottModel```      |     $f(u, v) = -u^2 v + F(1 - u)$ <br> $g(u, v) = u^2 v - (F + k)v$     |   $F$: ``F``, $k$: ``k``      |
| Gierer-Meinhardt    | ```LiawModel```           |    $f(u, v) = \rho_u \frac{u^2 v}{1 + \kappa u^2} + \sigma_u - \mu u$ <br> $g(u, v) = -\rho_v \frac{u^2 v}{1 + \kappa u^2} + \sigma_v$   |   $ρ_u$: ``ru``, $ρ_v$: ``rv``, $κ$: ``k``, $σ_u$: ``su``, $σ_v$: ``sv``, $µ$: ``mu``     |
| Gray-Scott          | ```SchnakenbergModel```   |     $f(u, v) = \sigma_u - \mu u + \rho_u^2 v$ <br> $g(u, v) = \sigma_v - \rho_u^2 v$     |   $σ_v$: ``sv``, $σ_u$: ``su``, $ρ$: ``rho``, $µ$: ``mu``      |  

- custom PDE model class  
```python
from lpf.models import TwoComponentModel

class MyModel(TwoComponentModel):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._name = "MyModel"

	def reactions(self, t, u_c, v_c):
		"""Define the reactions of a two-component system.
		"""
		return f, g

	def to_dict(self, *args, **kwargs):
		"""Create a dict to store parameter values.
		"""
		return n2v

	@staticmethod
	def parse_params(model_dicts):
		"""Parse the parameter sets from model dictionaries.
		"""
		return params

	def get_param_bounds(self):
		"""The bounds of the decision vector in EvoSearch.
		"""
		return bounds_min, bounds_max

	def len_decision_vector(self):
		"""The length of the decision vector in EvoSearch.
		"""
		return 0
```   

- ```__init__```, ```reactions```, ```to_dict```, ```parse_params```: ```TwoComponentModel``` model class에서 custom하기 위해 정의할 methods. ```reactions```에서 방정식이 정해진다.   
- ```get_param_bounds```, ```len_decision_vector```: 원하는 진화 탐색 수행을 위해 정의할 methods   
- Liaw model 생성   
```python
from lpf.models import LiawModel

# Create the Liaw model.
model = LiawModel(
	initializer=initializer,
	params=params,
	dx=dx,
	width=width,
	height=height,
	device=device
)
```   

**2.1.5**  Performing a numerical simulation   
- LPF에 구현된 Numerical methods   

| **Method**     | **Class** | **Definition**  |
|-----------------|--------------------|--------------------|
| Euler          | `EulerSolver`      | $y_{n+1} = y_n + h \cdot f(t, y_n)$ |
| Heun           | `HeunSolver`       | $k_1 = h \cdot f(t, y_n)$ <br> $k_2 = h \cdot f(t + h, y_n + k_1)$ <br> $y_{n+1} = y_n + \frac{k_1 + k_2}{2}$                                                                                                                                                                                       
| Runge-Kutta    | `RungeKuttaSolver` | $k_1 = h \cdot f(t, y_n)$ <br> $k_2 = h \cdot f(t + \frac{h}{2}, y_n + \frac{k_1}{2})$ <br> $k_3 = h \cdot f(t + \frac{h}{2}, y_n + \frac{k_2}{2})$ <br> $k_4 = h \cdot f(t + h, y_n + k_3)$ <br> $y_{n+1} = y_n + \frac{k_1 + 2k_2 + 2k_3 + k_4}{6}$  

- Neumann boundary conditions of two-component model   
$\text{boundary} \left( \frac{\partial u}{\partial t} \right) = 0 \quad \rightarrow \quad u'(0:h-1,\, 0:w-1) = 0$ <br> $\text{boundary} \left( \frac{\partial v}{\partial t} \right) = 0 \quad \rightarrow\quad v'(0:h-1, \, 0:w-1) = 0$


- Performing a numerical simulation   
```python
from lpf.solvers import EulerSolver

# Create a solver and perform a numerical simulation.
solver = EulerSolver()

solver.solve(
	model=model,
	dt=dt,
	n_iters=n_iters,
	period_output=1000,
	dpath_model=dpath_output,
	dpath_ladybird=dpath_output,
	dpath_pattern=dpath_output,
	verbose=1
)
```  
- numerical solvers: 시간 매개변수를 이용해 모델에 대한 numerical simulation 수행. 이미지 및 파일들이 포함된 디렉토리를 경로에 출력.  
- `dpath_pattern`: 2D 패턴 이미지  
- `dpath_ladybird`: 무당벌레 형태 이미지  
- `dpath_model`: 모델 정보 파일  

**2.1.6**  Visualizing the results   
- 무당벌레 형태, 패턴의 진화 시각화  
```python
from os.path import join as pjoin
from lpf.visualization import merge_single_timeseries

dpath_images = pjoin(dpath_output, "model_1")

img_patterns = merge_single_timeseries(
    dpath_input=dpath_images,
    n_cols=10,
    infile_header="pattern",
    ratio_resize=0.5,
    text_format="n = ",
    font_size=10,
    text_margin_ratio=0.1
)

img_patterns.save(pjoin(dpath_output, "output_pattern.png"))

img_ladybirds = merge_single_timeseries(
    dpath_input=dpath_images,
    n_cols=10,
    infile_header="ladybird",
    ratio_resize=0.5,
    text_format="n = ",
    font_size=10,
    text_margin_ratio=0.1
)

img_ladybirds.save(pjoin(dpath_output, "output_ladybird.png"))
```  
- `merge_single_timeseries`: `infile_header`로 정의된 문자열로 시작하는 이미지 파일을 찾아 단일 이미지로 병합  
- `save`: 이미지 파일로 저장하는 method  

- 출력 디렉토리 구조     
<OUTPUT_DIR>     
├── model_1    
│   ├── ladybird_000001.png     
│   ├── pattern_000001.png   
│   ├── ladybird_000002.png   
│   ├── pattern_000002.png   
│   ├── ladybird_000003.png   
│   ├── pattern_000003.png    
│   └── ...      
└── models    
    └── model_1.json    
