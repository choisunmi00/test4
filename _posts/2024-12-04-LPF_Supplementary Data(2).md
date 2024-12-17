---
layout: single
title: LPF_Supplementary Data(2)
categories: LPF
tags: paper
---

# 논문 리뷰: LPF - Supplementary Data

[LPF: a framework for exploring the wing color pattern formation of ladybird beetles in Python](https://doi.org/10.1093/bioinformatics/btad430)  
[Supplementary Data](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/39/7/10.1093_bioinformatics_btad430/2/btad430_supplementary_data.pdf?Expires=1736057160&Signature=ynQSiSKNANknwTVvSYi5cTxcCp9gHsKwCX1ply7Ys0QzNZ0TQilTTt0xPhjEJXNwXfgbw3M-Rk7jPuxqt9JicwdLO-RS1wej0LarlSNwLrT3~I8iV8b6YrDm-UhelBilpi5P4lgoq-s9mn8M-89FfFf0LIn7ZiE6cPwYsdqRvdMP3CQNw0wocNRet~9qnUMM25qWD7wD68GOKpbxIEBE9eOiGfDq33m6tzWM3fUMTuMXqL8YAm53YsL54Dw8ZPb0T7UZ62vRyCSCmn-vGKRjrcD65lM-C1fJ1krF9hUAAFRdFHgmHiDG~YRJjy9Jn-AaroBOVcLBPGZDkBGki3nocg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

---

## 3. Pattern visualization  
- 무당벌레의 색상 패턴 다형성이 두드러지는 딱지날개(앞날개)에 초점  
- Gautier et al. 표현형 참고, 단순화된 이미지 고안  

### 3.1 Visualizing a single morph  
- 무당벌레의 날개 색상 패턴은 대칭적이라 가정(생물학적 시스템의 대칭성과 모듈성)   

<img src="https://github.com/user-attachments/assets/dfea6281-2407-4f47-b4a4-e0c9c7725d49" alt="1" width="40%" height="40%"/>    

1. PDE 모델을 풀고, 주요 상태에 따라 2D 공간 채색: Liaw 모델에서 멜라닌 합성을 결정하는 주요 상태를 상태 u라고 가정한다면, 임계값보다 큰 상태의 u가 있는 위치에 멜라닌을 나타내는 검은색을 할당한다.  
2. 패턴 이미지에서 미리 정의된 영역을 잘라내어 날개 색상 패턴 추출  
3. 잘라낸 날개 패턴을 무당벌레 날개 이미지의 절반을 생성하는 템플릿 이미지와 합성  
4. 무당벌레 날개 패턴의 왼쪽과 오른쪽 절반을 병합  

### 3.2 Visualizing multiple morphs  

```python
# A model knows how to colorize its states.
arr_color = model.colorize(thr_color=0.5)

imgs = []
for i in range(arr_ladybird.shape[0]):
img_ladybird, img_pattern = model.create_image(i, arr_ladybird)
imgs.append(img_ladybird)

img_output = merge_multiple(imgs=imgs,
n_cols=4,
ratio_resize=2.0,
text_format="morph = ",
font_size=32)
# Save the image file.
img_output.save("morphs.png", dpi=(600, 600))
```  

### 3.3 Creating a video for temporal evolution  

- ```models``` directory: contains model JSON files that include metadata, parameter, values, initializing positions
- ```model_*``` directory: contains the ladybird images of each model generated during the numerical simulation

<img src="https://github.com/user-attachments/assets/a6104125-9ea4-4ed4-b8cf-0020cbb48524" alt="2" width="60%" height="60%"/>   

- ```merge multiple timeseries```: concatenates multiple morph images into a frame and performs the concatenation over time. ```model_*``` directory에 있는 동일한 파일 이름을 가진 여러 이미지 파일을 단일 이미지로 병합.
- ```create video```: 프레임 병합, MP4 비디오 생성.

## 4. Evolutionary search

- PyGMO에 기반한 진화 탐색 알고리즘 제공  
- A workflow of searching mathematical models for *H. axyridis*  

<img src="https://github.com/user-attachments/assets/aaf1c378-8744-4a63-9c07-5f9b85454adb" alt="3" width="60%" height="60%"/>   

### 4.1 Parameter optimization  

- 진화 탐색 알고리즘을 사용해 모델의 매개변수 집합 최적화    
- [tutorial05_evosearch](https://github.com/cxinsys/lpf/blob/main/tutorials/tutorial05_evosearch.ipynb)  
1. PyGMO에서 알고리즘을 선택하고 population, algorithm, island와 같은 객체를 생성한다.  
2. 미리 정의된 모델을 사용하여 population을 초기화  
3. converter 객체 생성: 검색 알고리즘의 decision vector를 수학적 모델의 매개 변수에 매핑  
4. 검색 알고리즘에 의해 생성된 매개변수 집합에 대해 PDE 모델을 푼다.  
5. 검색 알고리즘은 합성 형태 이미지 생성, 합성 이미지와 대상 이미지의 유사도를 측정  
6. 검색 알고리즘은 유사도를 fitness score로 해석  
7. 검색 알고리즘은 population 업데이트  
8. 진화 과정 계속할지 여부 결정  
- ```pygmo.sade```:  self-adaptive differential evolution  
- ```pygmo.pso```: particle swarm optimization  
- ```pygmo.problem```: ```fitness``` 함수 정의하는 모든 개체 포함  
- ```fitness```: decision vector의 objective score혹은 fitness score 평가  
- ```Converter```: floating-point 숫자 배열인 decision vector를 모델의 매개변수에 매핑, 반대의 경우도 마찬가지(예를 들어 ```LiawConverter.to params()```는 decision vector의 첫 번째 및 두 번째 값을 각각 ```LiawModel```의 확산 변수 $D_u$, $D_v$에 매핑 )
- ```LiawConverter.to initializer()```: decision vector로부터 ```LiawInitializer```를 인스턴스화

- Fitness function EvoSearch class in LPF.  

```python
class EvoSearch:

    def fitness(self, x):
        digest = get_hash_digest(x)

        if digest in self.cache:
            arr_color = self.cache[digest]
        else:
            x = x[None, :]
            initializer = self.converter.to_initializer(x)
            params = self.converter.to_params(x)

            self.model.initializer = initializer
            self.model.params = params

            try:
                self.solver.solve(self.model)
            except (ValueError, FloatingPointError) as err:
                print("[ERROR IN FITNESS EVALUATION]", err)
                return [np.inf]

            # Colorize the ladybird model.
            arr_color = self.model.colorize()

            # Store the colorized object in the cache.
            self.cache[digest] = arr_color
        # end of if-else

        # Evaluate objectives.
        ladybird, pattern = self.model.create_image(0, arr_color)
        sum_obj = 0
        for obj in self.objectives:
            val = obj.compute(ladybird.convert("RGB"), self.targets)
            sum_obj += val
            return [sum_obj]
```   

### 4.2 Fitness score  

- ```EvoSearch```: uses single or multiple objectives to evaluate the fitness score of a decision vector  

$$
F = c_1 \cdot S_{\text{MSE}} + c_2 \cdot S_{\text{CP}} + c_3 \cdot S_{\text{VGG16}} + c_4 \cdot S_{\text{LPIPS:VGG16}} + c_5 \cdot S_{\text{LPIPS:ALEX}}
$$  

- $S$: 유사도 점수, $c$: 계수  
- $S_{\text{MSE}}$: $MSE$에 기반한 유사도 점수. 여러 대상 이미지에 대한 $MSE$의 평균  

$$
S_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} \text{MSE}(X, T(i))
$$  

- $m$: 타겟의 수  
- 형태, 타겟 이미지 사이의 $MSE$  

$$
\text{MSE}(X, T) = \frac{1}{n} \sum_{j=1}^{n} (X_j - T_j)^2
$$

- $X$: 형태 이미지, $T$: 타겟 이미지, $n$: 이미지의 픽셀 수  

- $S_{\text{CP}}$: 색상 비율에 기반한 유사도 점수. OpenCV의 ```inRange```를 사용해 비율 계산.  

$$
pdf(x, \mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2}
$$  

$$
S_{\text{CP}} = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{pdf(\text{CP}(X), \text{CP}(T(i)), 0.1)}
$$  

- *pdf*: probability density function of normal distribution, $CP$: color proportion  

- $S_{\text{VGG16}}$: VGG16 perceptual loss에 기반한 유사도 점수. $VGG16$ perceptual loss를 형태, 타겟 이미지 feature maps 사이의 $LAE$ 혹은 $L1$ loss로 정의  

$$
S_{\text{VGG16}} = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{4} \text{L1-loss} \left( \phi_j(X) - \phi_j(T(i)) \right)
$$  

- $S_{\text{LPIPS}}$: learned perceptual image patch similarity(LPIPS)을 기반으로 하여 $S_{\text{VGG16}}$와 유사

- 4개의 붉은 반점이 있는 spectabilis(반점 사이의 간격이 다소 좁다)와 가장 유사한 morph-3의 경우, 3개의 합성 신경망을 갖춘 $LPIPS$ 지표가 유사성을 정확히 설명  

### 4.3 Results of a case study: reproducing schematic images  
- *H. axyridis*의 succinea subtype 조사   
- Fitness score를 결정하는 방정식에서 계수를 설정하여 지표(유사도 점수에 대한)의 크기와 중요도 조정.   
- schematic images 제작  

<img src="https://github.com/user-attachments/assets/d63d46c1-6184-400f-90c2-b5416ff9779a" alt="4" width="40%" height="40%"/>   

- axyridis, succinea 형에 대한 진화 탐색: 개체군 최대 크기인 16에 도달했을 때 세대를 거치며 전체가 검은색인 형태는 사라져감. axyridis의 200번째 세대와 succinea의 500번째 세대에서 타겟 이미지와 유사한 형태가 나왔으나 패턴이 완벽하게 일치하지 않음.  
- conspicua, spectabilis 형에 대한 진화 탐색: 초기 개체군의 매개변수 집합을 이전에 평가된 것으로 사용. 비교적 더 단순한 패턴이므로 800번째 세대에서 타겟과 유사한 형태로 채워짐. 이는 LPF에서 딥러닝 모델이 평가한 fitness score를 기반으로 진화 탐색을 수행했을 때, 타겟 이미지와 유사한 무당벌레의 날개 색상 패턴을 생성할 수 있는 수학적 모델의 매개변수 집합을 찾을 수 있음을 시사함.  

### 4.4 Results of a case study: reproducting real images  
- *H. axyridis*의 spectabilis subtype 진화 탐색 수행. 타겟은 연구실에서 촬영한 비대칭 패턴의 노이즈가 있는 spectabilis 사진.  

<img src="https://github.com/user-attachments/assets/425c7381-9a2b-426f-85ce-e92d53ce334d" alt="5" width="40%" height="40%"/>   

- 300번째 세대에서 유사한 형태 관찰. 이는 LPF를 통해 실제 이미지를 재현할 수 있는 수학적 모델을 발견할 수 있음을 시사. 다만 노이즈가 많고 비대칭 패턴은 수치적 오류를 일으킬 수 있음.

## 5. Diploid model 

- 유전자 교배 실험은 *H. axyridis*의 날개 색상 패턴이 단일 상염색체에서 분리되는 다양한 대립유전자에서 유래함을 입증  
- ‘mosaic dominance’ phenomena: 이형접합체의 색상 패턴은 두 대립유전자의 색상 패턴의 조합   
- *H. axyridis*의 색상 패턴 다형성의 근원은 repeated inversion within a *pannier* intron.   
- Diploid models of *H. axyridis* in LPF   

<img src="https://github.com/user-attachments/assets/541a998f-4b89-44eb-b6fb-e725c52fed78" alt="6" width="40%" height="40%"/>       

- 유전적 특징을 반영하기 위해 diploid models 개발. *pannier*의 발현이 부계와 모계 대립유전자에서 발현되는 가상의 형태발생인자에 의해 조절된다 가정. 또한 $u$와 $v$ 사이의 crosstalks 고려.  
- two-component system에서 PDE 모델 정의  

$$
\frac{\partial u_p}{\partial t} = D_{u_p} \nabla^2 u_p + f_p(u_p, v_p),
$$  

$$
\frac{\partial v_p}{\partial t} = D_{v_p} \nabla^2 v_p + g_p(u_p, v_p),
$$  

$$
\frac{\partial u_m}{\partial t} = D_{u_m} \nabla^2 u_m + f_m(u_m, v_m),
$$  

$$
\frac{\partial v_m}{\partial t} = D_{v_m} \nabla^2 v_m + g_m(u_m, v_m),
$$  

- $m$, $p$는 부계와 모계의 origins.  
- 자손 모델의 total $u$, $v$는 부계와 모계 상태의 선형 조합으로 정의.  

$$
u = \alpha u_p + \beta u_m,
$$  

$$
v = \alpha v_p + \beta v_m,
$$  

- crosstalk model of a two-component system  

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u + \alpha f_p(u, v) + \beta f_m(u, v),
$$  

$$
\frac{\partial v}{\partial t} = D_v \nabla^2 v + \alpha g_p(u, v) + \beta g_m(u, v),
$$  

- 반응은 부계 및 모계 반응의 선형 조합으로 모델링.  
- 식들은 LPF의 ```TwoComponentDiploidModel```과 ```TwoComponentCrosstalkDiploidModel```으로 구현  

### 5.1 Numerical simulation  

- *H. axyridis*의 succinea와 conspicua 모델을 교차시킨 시뮬레이션 실험 수행. crosstalks가 없는 diploid model은 mosaic dominance phenomena가 나타남. 반면 crosstalks가 있는 모델의 색상 패턴은 그렇지 않음. *pannier*를 더 잘 이해하기 위해 LPF의 diploid model은 무당벌레 다형성의 기본 메커니즘에 대한 다양한 가설을 테스트하는데 유용할 것으로 기대.  

<img src="https://github.com/user-attachments/assets/5b73fcb1-2904-460a-8391-0ab4a71c12b6" alt="7" width="40%" height="40%"/>     

### 5.2 Population evolution

- 단순화를 위해 단일 매개변수, 초기 상태 및 diploid model의 반수체 모델 간의 초기화 위치에서 무작위로 발생할 수 있는 *in silico* crossover 구현.  

<img src="https://github.com/user-attachments/assets/973eb106-98e3-4ae1-9ce6-1ef1976a3cc2" alt="8" width="60%" height="60%"/>     

- An example of evolving a diploid population by crossing.  
[evopop_liawmodel](https://github.com/cxinsys/lpf/blob/main/search/evopop_liawmodel.py)   

<img src="https://github.com/user-attachments/assets/9913f2b4-55fd-4808-9b8e-ee19fb3b9db6" alt="9" width="50%" height="50%"/>   

<img src="https://github.com/user-attachments/assets/eeba9d2f-75a6-45cd-8832-357e2f11ecd8" alt="10" width="50%" height="50%"/>   

- 초기 세대에서는 모두 mosaic dominance를 볼 수 있다. 두 개의 같은 실험에서 나온 개체군 진화의 결과라 하더라도 최종 개체군은 무작위 선택과 교차로 인해 완전히 달라진다. 또 몇 가지 형태가 세대를 거쳐 개체군을 지배하기도 하는데 이는  일종의 genetic drift를 시사한다.   

