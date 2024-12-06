---
layout: single
title: Network science
categories: Network_science
tags: [study, graph theory, GNN]
---

# Network science  
- 복잡한 시스템을 분석하는 데 network를 수학적으로 기술하여 이용하는 학문. 구조와 동작을 연구한다.   
- 소셜, 생물학적, 통신, 경제 및 금융, 교통, 생태 네트워크 등에서 응용.

- network model   
  - Erdős–Rényi random graph model  
  - Configuration model  
  - Watts–Strogatz small world model   
  - Barabási–Albert (BA) preferential attachment model   
  - Non-linear preferential attachment
  - Mediation-driven attachment (MDA) model
  - Fitness model
  - Exponential random graph models

# Graph theory    
- 오일러가 Königsberg Bridge Problem에 대한 풀이를 보이며 처음 제시했다.   
  강의 섬들을 vertex로, 다리를 edge로 표현하여 문제를 그래프 형태로 변환했다.   
- Euler's Theorem   
  - Eulerian Path: 그래프의 모든 간선을 한 번씩만 지나가는 경로   
    조건: 모든 vertex의 차수가 짝수, 연결 그래프(vertex 모두 연결)   
  - Eulerian Curcuit: 그래프의 모든 간선을 한 번씩만 지나가고, 시작점과 끝점이 같은 경로   
    조건: 차수가 홀수인 vertex 2개, 연결 그래프   
- 1936년 Dénes Kőnig가 "Theory of Finite and Infinite Graphs"를 출판했다.   
  vertex와 edge를 정의하고 그래프의 구조적 속성, 유한 그래프와 무한 그래프, Bipartite Graph, Matching Theory, Graph Coloring Problem, Flow Network에 대해 다루었다.   
- 그래프 알고리즘의 응용: Edsger Dijkstra의 최단 경로 알고리즘, Kruskal과 Prim의 Minimum Spanning Tree, 네트워크 유량 문제를 해결하는 Ford-Fulkerson Algorithm 등.   
- NP-hard 문제와의 연관성: Graph Coloring Problem, Hamiltonian Circuit Problem 등.
  
## 그래프 이론과 현대 과학  
- 빅데이터와 네트워크 과학  
  - Social Network Analysis  
    - 친구 추천 알고리즘: 사용자를 vertex로, 친구 관계를 edge로 하여 공통 친구 수는 네트워크 내 연결성을 분석하여 추천한다.   
    - 인플루언서 탐지: 그래프의 중심성(Centrality) 개념을 사용해 네트워크에서 중요한 역할을 하는 사용자를 찾는다. (ex. 구글의 PageRank 알고리즘)  
  - 인터넷 네트워크 최적화  
    - 최단 경로 알고리즘(Dijkstra Algorithm): 서버와 라우터를 node로, 네트워크 연결을 link로 하여 인터넷을 거대한 그래프로 보고 데이터 전송 경로를 최적화한다.  
  - Recommendation System  
    - 플랫폼에서 그래프를 활용해 사용자와 콘텐츠 간 관계를 모델링.  
    - 양방향 그래프: 사용자와 콘텐츠를 vertex, 사용자의 특정 콘텐츠 이용 기록을 edge로 하여 유사성을 찾거나 콘텐츠 간 연결성을 분석해 추천한다.
- 시스템 생물학
  - 생물학적 네트워크의 유형
    - Protein-Protein Interaction, PPI: 단백질 간의 물리적/기능적 상호작용을 그래프로 모델링  
    - Gene Regulatory Network: 특정 유전자가 다른 유전자의 발현을 조절하는 관계를 나타냄. 질병의 메커니즘 연구 등.  
    - Metabolic Network: 대사 반응 경로를 그래프로 모델링. 약물이 특정 경로에 미치는 영향을 분석하는 데 활용.  
  - 그래프의 생물학적 활용
    - Disease Module: 그래프 클러스터링을 통해 질병 관련 유전자를 그룹화.
    - Drug Repurposing: 그래프를 이용해 기존 약물의 새로운 용도를 발견.
- 양자 컴퓨팅과 그래프
  - 최소 에너지 경로 탐색: 분자 상호작용 그래프에서 최소 에너지를 찾는 문제 해결.
  - Quantaum walk: 고전적 그래프 탐색 알고리즘(DFS, BFS 등)을 양자 컴퓨터에서 확장한 개념.
- AI와 머신러닝
  - GNN(Graph Neural Network): 머신러닝에서 그래프 데이터를 분석하고 예측하는 데 사용.  

# GNN
- 그래프 구조 학습
  - 비정형 데이터로 vertex와 edge의 관계가 표현되는 것이 특징.
  - Local Features: 각 vertex는 인접한 vertex(Neighborhood)과 edge 정보를 통해 의미 있는 특징을 가진다.
- 표현 학습 (Representing Learning)
  - Vertex Embedding: vertex를 고차원 벡터로 변환하여 의미 표현.
  - Graph-Level Embedding: 그래프 전체를 하나의 벡터로 표현하여 전체 구조 분석.
  - 인접 vertex 정보 통합: 해당 vertex뿐 아니라 인접 vertex와 edge의 정보를 종합하여 결정. (ex. 그래프 합성곱 연산으로 인접 vertex의 정보를 반영) 
- GNN 주요 알고리즘
  - GCN (Graph Convolutional Networks): 그래프 데이터를 합성곱 연산으로 처리하여 vertex의 지역적 특징 학습, 인접 vertex에서 정보를 집계(Aggregation)하여 각 vertex의 특징 업데이트.
  - GAT (Graph Attention Networks): GCN 확장 형태. Attetion Mezhanism을 도입해 인접 vertex의 중요도 학습, 중요한 vertex에 더 큰 가중치 부여.
  - GraphSAGE (Graph Sample and Aggregate): GCN과 달리 그래프를 한 번에 학습하지 않고, vertex 주변의 sampling된 서브 그래프를 학습. 대규모 그래프에 효율적.
  - DiffPool (Differentiable Pooling): 그래프의 계층적 구조를 학습하여 더 큰 구조적 관계를 파악하는 데 사용, 작은 그래프들을 병합하여 상위 레벨에서 특징을 학습.
  

