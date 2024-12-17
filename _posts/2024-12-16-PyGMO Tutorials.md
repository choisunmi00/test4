---
layout: single
title: PyGMO Tutorials
categories: Network_science
tags: PG
---

# class
## `problem`

**0. class `problem`**
- 최적화 문제의 틀(blueprint)을 제공 
- 구성 요소  
  1__목표 함수 (Objective Function): 최적화에서 최소화/최대화할 값을 정의. 다중 목적 문제의 경우 여러 목표 함수도 지원    
  2__설계 변수 (Decision Variables): 최적화에서 조작할 변수들. 각 변수는 상한과 하한으로 제약 조건을 가짐  
  3__제약 조건 (Constraints): 설계 변수나 결과 값이 만족해야 할 부등식 또는 등식 조건   
  4__문제 차원 (Dimensionality): 설계 변수와 목표 함수의 개수를 정의
  
**1. User Defined Problems(UDPs) 정의**  
- `pygmo.problem`을 상속

```python3
import pygmo as pg

# 사용자 정의 문제 클래스
class MyProblem:
    def __init__(self):
        self.dimension = 2  # 설계 변수의 수
        self.bounds = ([0, 0], [1, 1])  # 변수의 하한과 상한
    
    def fitness(self, x):
        # 목적 함수 정의 (예: 단일 목적)
        return [x[0]**2 + x[1]**2]  # 최소화 문제
    
    def get_bounds(self):
        return self.bounds

# PyGMO Problem 객체로 변환
problem = pg.problem(MyProblem())

```
**2. Pygmo 내장 UDPs**

```python3
import pygmo as pg

prob = pg.problem(pg.rosenbrock(dim=10))
print(prob)
```
```
Problem name: Multidimensional Rosenbrock Function  
        C++ class name: struct pagmo::rosenbrock   
        
        Global dimension:                       10  
        Integer dimension:                      0  
        Fitness dimension:                      1  
        Number of objectives:                   1  
        Equality constraints dimension:         0  
        Inequality constraints dimension:       0  
        Lower bounds: [-5, -5, -5, -5, -5, ... ]  
        Upper bounds: [10, 10, 10, 10, 10, ... ]  
        Has batch fitness evaluation: false

        Has gradient: true  
        User implemented gradient sparsity: false  
        Expected gradients: 10    
        Has hessians: false    
        User implemented hessians sparsity: false

        Fitness evaluations: 0  
        Gradient evaluations: 0

        Thread safety: constant  
```
**3. 클래스 `problem` 주요 메서드**  
1. `fitness(x)`
- 입력 변수 x에 대해 목적 함수 값을 계산  
- 반환값은 리스트 형태로 여러 목표 값을 가질 수 있다
2. `get_bounds()`  
- 설계 변수의 상한과 하한을 정의  
- 반환값: 튜플 `(lower_bounds, upper_bounds)`  
3. `get_name()`  
- 문제의 이름을 반환  
4. `get_nobj()`  
- 목적 함수의 개수를 반환  
5. `get_nec()`  
- 등식 제약 조건의 수를 반환  
6. `get_nic()`  
- 부등식 제약 조건의 수를 반환  
7. `gradient(x)`
- 입력 변수 `x`에서의 목적 함수의 그래디언트를 반환  


```python3
import pygmo as pg

class ConstrainedProblem:
    def __init__(self):
        self.dimension = 2  # 변수 개수
        self.bounds = ([0, 0], [10, 10])  # 변수 제한

    def fitness(self, x):
        # 목적 함수
        f = [x[0]**2 + x[1]**2]
        # 부등식 제약: g(x) >= 0
        g = [x[0] + x[1] - 5]
        return f + g  # 목적 함수 + 제약 조건 반환
    
    def get_bounds(self):
        return self.bounds
    
    def get_nic(self):
        return 1  # 부등식 제약 조건 수

# PyGMO Problem 생성
problem = pg.problem(ConstrainedProblem())

# 알고리즘 및 최적화
algo = pg.algorithm(pg.sade(gen=100))
pop = pg.population(problem, size=20)
pop = algo.evolve(pop)

print("최적의 값:", pop.champion_f)
print("최적의 변수:", pop.champion_x)

```

## `algorithm`

**0. class `algorithm`**  
- 최적화 문제를 푸는 전략을 정의  
  
**1. User Defined Algorithms(UDAs) 정의**   
-  `pg.algorithm`을 통해 생성  

```python3
# Differential Evolution 알고리즘 생성
algo = pg.algorithm(pg.de(gen=100))  # 100 세대 실행

# Self-Adaptive Differential Evolution 알고리즘
algo = pg.algorithm(pg.sade(gen=200, variant=2, ftol=1e-5))

# Particle Swarm Optimization 알고리즘
algo = pg.algorithm(pg.pso(gen=100, omega=0.5, eta1=2.0, eta2=2.0))
```
```pytnon3
import pygmo as pg

# 문제 정의
problem = pg.problem(pg.sphere(dim=10))  # 10차원 Sphere 문제

# 알고리즘 생성
algo = pg.algorithm(pg.de(gen=100))

# 문제와 알고리즘 결합
island = pg.island(algo=algo, prob=problem, size=20)

# 최적화 실행
island.evolve()
island.wait_check()

# 결과 출력
print("최적의 목적 함수 값:", island.get_population().champion_f)
print("최적의 설계 변수 값:", island.get_population().champion_x)
```
**2. Pygmo 내장 UDAs**  

```python3
import pygmo as pg

algo = pg.algorithm(pg.cmaes(gen=100, sigma0=0.3))
print(algo)
```
```
Algorithm name: CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy [stochastic]
        C++ class name: class pagmo::cmaes

        Thread safety: basic

Extra info:
        Generations: 100
        cc: auto
        cs: auto
        c1: auto
        cmu: auto
        sigma0: 0.3
        Stopping xtol: 1e-06
        Stopping ftol: 1e-06
        Memory: false
        Verbosity: 0
        Force bounds: false
        Seed: 1467419779
```
1. 진화 알고리즘 (Evolutionary Algorithms)
- Differential Evolution (DE): 차분 진화를 사용하여 최적화를 수행
- Self-Adaptive Differential Evolution (sade): DE의 파라미터를 자동 조정
- Genetic Algorithm (GA): 유전자 알고리즘을 사용하여 최적화를 수행
2. 다목적 최적화 알고리즘
- NSGA-II: 빠르고 비지배적인 다목적 최적화 알고리즘
- MOEA/D: 분해 기반 다목적 최적화 알고리즘
3. 전역 최적화 알고리즘
- Simulated Annealing (SA): 모의 담금질 알고리즘
- Particle Swarm Optimization (PSO): 입자 군집 최적화 알고리즘
- CMA-ES: 공분산 행렬 진화 전략
4. 지역 탐색 알고리즘
- Nelder-Mead: 단순체 방법을 사용한 지역 탐색
- Compass Search: 탐색 기반 지역 최적화

**3. 클래스 `algorithm` 주요 메서드**  
1. `evolve(population)`
- 알고리즘이 입력된 인구(population)를 사용하여 최적화를 수행
- 입력: `population` 객체
- 반환: 진화된 `population` 객체
2. `set_verbosity(level)`
- 알고리즘의 동작 중 출력되는 로그의 상세 수준을 설정
- `level`이 높을수록 자세한 정보를 출력
```
algo.set_verbosity(2)
```
3. `extract()`
- 알고리즘 객체의 내부 정보를 추출
- 주로 알고리즘의 동작 세부 사항을 확인할 때 사용
```python
sade_algo = pg.algorithm(pg.sade(gen=100))
sade_details = sade_algo.extract(pg.sade)
print(sade_details.get_variant())
```
4. `get_name()`
- 알고리즘의 이름을 반환합니다.
```python
print(algo.get_name())
```

## `population`  

**0. class `population`**  
- 개체군(population)은 최적화 문제(problem)의 후보 해(solution)의 저장소: 결정 벡터 및 적합도 벡터 포함.
- 구성 요소
  1__개체(individual): 개체군의 각 요소는 하나의 후보 해. 설계 변수와 이에 따른 목적 함수 값을 포함
  2__크기(size): 개체군 내 개체의 수
  3__문제(problem): 개체군이 해결하려는 최적화 문제

**1. `population` 객체 생성**   

```python3
import pygmo as pg

prob = pg.problem(pg.rosenbrock(dim=4))
pop1 = pg.population(prob)
pop2 = pg.population(prob, size=5, seed=723782378)

print(len(pop1))
print(pop1.problem.get_fevals())
print(len(pop2))
print(pop2.problem.get_fevals())

print(pop2)
```
```
0
0
5
5
Problem name: Multidimensional Rosenbrock Function
        C++ class name: struct pagmo::rosenbrock

        Global dimension:                       4
        Integer dimension:                      0
        Fitness dimension:                      1
        Number of objectives:                   1
        Equality constraints dimension:         0
        Inequality constraints dimension:       0
        Lower bounds: [-5, -5, -5, -5]
        Upper bounds: [10, 10, 10, 10]
        Has batch fitness evaluation: false

        Has gradient: true
        User implemented gradient sparsity: false
        Expected gradients: 4
        Has hessians: false
        User implemented hessians sparsity: false

        Fitness evaluations: 5
        Gradient evaluations: 0

        Thread safety: constant

Population size: 5

List of individuals:
#0:
        ID:                     15730941710914891558
        Decision vector:        [-0.777137, 7.91467, -4.31933, 5.92765]
        Fitness vector:         [470010]
#1:
        ID:                     4004245315934230679
        Decision vector:        [3.38547, 8.94985, 0.924838, 4.39905]
        Fitness vector:         [628823]
#2:
        ID:                     12072501637330415325
        Decision vector:        [-1.17683, 1.16786, -0.291054, 4.99031]
        Fitness vector:         [2691.53]
#3:
        ID:                     15298104717675893584
        Decision vector:        [1.34008, -0.00609471, -2.80972, 2.18419]
        Fitness vector:         [4390.61]
#4:
        ID:                     4553447107323210017
        Decision vector:        [-1.04727, 6.35101, 6.39632, 5.80792]
        Fitness vector:         [241244]

Champion decision vector: [-1.17683, 1.16786, -0.291054, 4.99031]
Champion fitness: [2691.53]
```
**2. 클래스 `population` 주요 메서드**  
1. 생성자
- ``__init__(self, prob, size=1)``:
  - `prob`: 최적화 문제 객체 (`pg.problem`)
  - `size`: 개체군의 초기 크기 (기본값은 1)

```python3
pop = pg.population(prob=problem, size=10)  # 문제와 크기 설정
```
2. 개체군 크기 관련 메서드
- `size`: 현재 개체군의 크기를 반환

```python3
print(pop.size)  # 출력: 개체 수
```
- `push_back(x)`: 개체군에 새로운 개체를 추가
  - `x`: 설계 변수 값 리스트

```python3
pop.push_back([0.5, 0.5, 0.5, 0.5, 0.5])  # 새로운 개체 추가
```
3. 최적 해 관련 메서드
- `champion_x`: 개체군에서 가장 좋은 해의 설계 변수 값을 반환
- `champion_f`: 개체군에서 가장 좋은 해의 목적 함수 값을 반환

```python3
print(pop.champion_x)  # 최적의 설계 변수 값
print(pop.champion_f)  # 최적의 목적 함수 값
```
4. 개체 정보 조회
- `get_x(i)`: 인덱스 `i`에 해당하는 개체의 설계 변수 값을 반환
- `get_f(i)`: 인덱스 `i`에 해당하는 개체의 목적 함수 값을 반환

```python3
print(pop.get_x(0))  # 첫 번째 개체의 설계 변수 값
print(pop.get_f(0))  # 첫 번째 개체의 목적 함수 값
```
5. 개체군 초기화
- `problem`: 개체군이 해결하고 있는 문제를 반환


```python3
print(pop.problem.get_name())  # 문제 이름 출력
```
6. 병합 및 진화
- `crossover(p1, p2)`: 두 개체 간 교차 연산을 수행합니다.
- `mutate(i)`: 특정 개체의 돌연변이를 수행합니다.

```python3
import pygmo as pg

# 문제 정의
problem = pg.problem(pg.sphere(dim=5))  # 5차원 Sphere 문제

# 개체군 생성
pop = pg.population(prob=problem, size=20)

# 알고리즘 정의
algo = pg.algorithm(pg.de(gen=100))

# 개체군 진화
pop = algo.evolve(pop)

# 최적화 결과 출력
print("최적의 목적 함수 값:", pop.champion_f)
print("최적의 설계 변수 값:", pop.champion_x)
```

## `island`  

**0. class `island`**   
- 단위 병렬화 블록. 최적화 문제와 알고리즘을 통합하여 진화를 수행하는 단위
- 하나의 문제(problem)과 하나의 알고리즘(algorithm)을 연결하여 개체군(population)을 진화시키는 환경 제공
- 진화의 결과는 최적의 해로 수렴  
- 구성 요소  
  1__문제 (problem): 섬에서 풀고자 하는 최적화 문제.  
  2__알고리즘 (algorithm): 문제를 최적화하기 위해 사용하는 알고리즘.  
  3__개체군 (population): 섬에서 진화하는 후보 해들의 집합.  

**1. `island` 객체 생성**    

```python3
import pygmo as pg

# 문제 정의
problem = pg.problem(pg.sphere(dim=5))  # 5차원 Sphere 문제

# 알고리즘 정의
algo = pg.algorithm(pg.de(gen=100))  # Differential Evolution, 100 세대

# 섬 생성
island = pg.island(algo=algo, prob=problem, size=20)  # 20개의 개체

# 섬에서 진화 실행
island.evolve()

# 섬의 진화가 완료되기를 기다림
island.wait_check()

# 최적화 결과 확인
print("최적의 목적 함수 값:", island.get_population().champion_f)
print("최적의 설계 변수 값:", island.get_population().champion_x)
```

**2. 클래스 `island` 주요 메서드**     
1. 생성자
- `__init__(self, algo, prob, size)`
  - `algo`: 알고리즘 객체 (`pg.algorithm`).
  - `prob`: 문제 객체 (`pg.problem`).
  - `size`: 개체군 크기.

```python3
island = pg.island(algo=pg.algorithm(pg.de(gen=100)), prob=problem, size=30)
```
2. 진화
- `evolve()`: 섬 내의 개체군을 알고리즘을 사용해 진화
- `wait_check()`: 진화가 끝났는지 확인

```python3
island.evolve()
island.wait_check()
```
3. 개체군 정보
- `get_population()`: 섬 내의 개체군 객체를 반환
- `set_population(population)`: 특정 개체군으로 섬의 개체군을 설정

```python3
population = island.get_population()
print(population.champion_f)  # 최적의 목적 함수 값
```
4. 이주
- `migrate(other_island)`: 현재 섬에서 다른 섬으로 개체를 이주
- `status()`: 섬의 상태를 확인

```python3
island1.migrate(island2)
print(island1.status())
```



```python3
import pygmo as pg

# 문제 정의
problem = pg.problem(pg.sphere(dim=10))

# 알고리즘 정의
algo1 = pg.algorithm(pg.de(gen=50))
algo2 = pg.algorithm(pg.pso(gen=50))

# 두 섬 생성
island1 = pg.island(algo=algo1, prob=problem, size=20)
island2 = pg.island(algo=algo2, prob=problem, size=20)

# 각 섬에서 진화 실행
island1.evolve()
island2.evolve()

# 섬 간 이주 수행
island1.migrate(island2)

# 결과 확인
print("섬 1 최적의 목적 함수 값:", island1.get_population().champion_f)
print("섬 2 최적의 목적 함수 값:", island2.get_population().champion_f)
```

## `archipelago`

**0. class `archipelago`**  
- 다중 섬 모델(multi-island model)을 관리.
- 구성 요소
  1__여러 섬 (`islands`): 최적화 문제와 알고리즘을 결합한 `island` 객체들의 집합.
  2__이주 정책 (`migration policy`): 섬 간 개체 교환 방식을 정의.
  3__병렬 실행: 여러 섬에서 최적화를 병렬로 실행 가능.

**1. `archipelago` 객체 생성**    


```python3
archi = pg.archipelago(
    n=10,                # 섬의 개수
    algo=pg.algorithm(pg.de(gen=100)),  # 알고리즘
    prob=pg.problem(pg.sphere(dim=5)), # 문제
    pop_size=20          # 각 섬의 개체군 크기
)
```


```python3
import pygmo as pg

# 문제 정의
problem = pg.problem(pg.sphere(dim=5))  # 5차원 Sphere 문제

# 알고리즘 정의
algorithm = pg.algorithm(pg.de(gen=100))  # Differential Evolution

# 아키펠라고 생성 (10개의 섬, 각 섬은 20개의 개체로 구성)
archi = pg.archipelago(n=10, algo=algorithm, prob=problem, pop_size=20)

# 병렬로 모든 섬에서 진화 실행
archi.evolve()

# 모든 섬의 진화가 완료될 때까지 대기
archi.wait_check()

# 결과 확인
for idx, isl in enumerate(archi):
    print(f"섬 {idx+1}의 최적 해: {isl.get_population().champion_f}")
```


```python3
import pygmo as pg

class toy_problem:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        return [sum(x), 1 - sum(x*x), - sum(x)]

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x) # numerical gradient

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self):
        return "A toy problem"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)
    
a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=1000))
p_toy = pg.problem(toy_problem(50))
p_toy.c_tol = [1e-4, 1e-4]
archi = pg.archipelago(n=21,algo=a_cstrs_sa, prob=p_toy, pop_size=70)
print(archi) 
```
```
Number of islands: 21
Topology: Unconnected
Migration type: point-to-point
Migrant handling policy: preserve
Status: idle

Islands summaries:

        #   Type                    Algo                                          Prob           Size  Status  
        -------------------------------------------------------------------------------------------------------
        0   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        1   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        2   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        3   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        4   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        5   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        6   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        7   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        8   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        9   Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        10  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        11  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        12  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        13  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        14  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        15  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        16  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        17  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        18  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        19  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
        20  Multiprocessing island  sa-CNSTR: Self-adaptive constraints handling  A toy problem  70    idle
```
**2. 클래스 `archipelago` 주요 메서드**     
1. 생성자 및 초기화
- __init__(n, algo, prob, pop_size):
  - `n`: 섬의 개수.
  - `algo`: 섬에서 사용할 알고리즘 객체
  - `prob`: 섬에서 풀고자 하는 최적화 문제
  - `pop_size`: 각 섬의 개체군 크기
2. 진화 및 이주
- `evolve()`: 각 섬에서 병렬로 진화를 수행
- `wait_check()`: 진화가 완료되었는지 확인
- `migrate()`: 섬들 간의 개체를 교환
3. 섬 관리
- `get_islands()`: 모든 섬 객체를 반환
- `push_back(island)`: 새로운 섬을 추가

```python3
archi.push_back(pg.island(algo=pg.algorithm(pg.pso(gen=100)), prob=problem, size=20))
```
4. 결과 확인
- 각 섬의 개체군(population)에 접근하여 최적 해를 확인

```python3
for isl in archi:
    print(isl.get_population().champion_f)  # 최적의 목적 함수 값
```



```python3
import pygmo as pg

# 문제 정의
problem = pg.problem(pg.rosenbrock(dim=5))  # 5차원 Rosenbrock 문제

# 다양한 알고리즘 정의
algo1 = pg.algorithm(pg.de(gen=100))  # Differential Evolution
algo2 = pg.algorithm(pg.pso(gen=100))  # Particle Swarm Optimization
algo3 = pg.algorithm(pg.sade(gen=100))  # Self-Adaptive DE

# 아키펠라고 생성
archi = pg.archipelago()

# 섬 추가
archi.push_back(pg.island(algo=algo1, prob=problem, size=20))
archi.push_back(pg.island(algo=algo2, prob=problem, size=20))
archi.push_back(pg.island(algo=algo3, prob=problem, size=20))

# 병렬로 진화 실행
archi.evolve()
archi.wait_check()

# 결과 출력
for idx, isl in enumerate(archi):
    print(f"섬 {idx+1}의 최적 해: {isl.get_population().champion_f}")

```











