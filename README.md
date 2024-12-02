# 📽️ Movie Recommendation

이 프로젝트의 목표는 **사용자의 영화 시청(평가) 이력 데이터를 기반으로, 사용자가 시청할 영화와 선호하는 영화를 예측하는 것**입니다. 

자세한 프로젝트 내용은 [Wrap-up Report](https://github.com/boostcampaitech7/level2-recsys-movierecommendation-recsys-04-lv3/blob/main/Wrap_Up_Report_Movie_Recommendation.pdf) 를 통해 확인해주세요.

## ⭐ 프로젝트 설명

1. **데이터**
    - **explicit feedback**인 Movie Lens 데이터를  **implicit feedback 데이터**로 변환
    - user-item interaction data와 item 정보에 대한 데이터 존재
2. **문제 정의**
    - time-ordered sequence에서 **일부 item이 누락된 상황**을 가정하여 **일반적인 sequential recommendation보다 복잡한 환경**
    ![사진 출처: AI Stages](image-1.png)
    
3. **side-information 활용**
    - item과 관련된 다양한 부가 정보를 함께 사용하는 방향으로 설계 가능

4. **예측 목표**
    
    Training Data에 존재하는 전체 유저에 대해서 각각 10개의 아이템 추천
    
    - **특정 시점 이후**의 사용자 행동 예측(Sequential)
    - **특정 시점 이전** 데이터로부터의 사용자 선호 파악(Static)
5. **평가 지표**
    - **normalized recall@10**

$$
\text { Recall@K }=\frac{1}{|U|} \sum_{u \in U} \frac{\left|\left\{i \in I_u \mid r a n k_u(i) \leq K\right\}\right|}{\min \left(K,\left|I_u\right|\right)}
$$

## 💡Team

| 강현구 | 서동준 | 이도걸 | 이수미 | 최윤혜 | 양시영 |
| --- | --- | --- | --- | --- | --- |
| <img src="" width="150" height="150"/> | <img src="" width="150" height="150"/> | <img src="" width="150" height="150"/> | <img src="" width="150" height="150"/> | <img src="" width="150" height="150"/> | <img src="" width="150" height="150"/> |
| Sequential-based models, Ensemble | EDA, LRML, K-fold Ensemble | ADMMSLIM, Parameter Tuning, General model | EDA, VAE기반 모델, MultiVAE 구현, Hard Voting | EDA, context-aware model, EASE/Multi-EASE 구현 | EDA, MLflow, CDAE, Soft Voting |

## 📑 구현된 모델 목록

- CDAE
- EASE
- LRML
- MultiVAE
- ADMMSLIM

## 📂 Architecture
```
.
├── EDA
│   ├── movie_title_similarity.ipynb
│   ├── user_genre_preference.ipynb
│   ├── usertimediff_genrecorr.ipynb
│   └── viewing_frequency_analysis_popular_vs_exploration.ipynb
├── Ensemble
│   ├── config.yaml
│   └── hard_voting.py
├── Experiments
│   ├── DeepFM_with_MLflow.ipynb
│   └── make_filtering_df.py
├── Models
│   ├── CDAE
│   │   ├── CDAE.py
│   │   ├── CDAE.yaml
│   │   ├── CDAE_dataset.py
│   │   ├── CDAE_train.py
│   │   ├── mlflow_setup.py
│   │   ├── random_seed.py
│   │   ├── run_CDAE.py
│   │   └── utils.py
│   ├── EASE
│   │   ├── EASE.py
│   │   ├── dataset.py
│   │   ├── run_train.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── LRML
│   │   ├── config.yaml
│   │   ├── dataset.py
│   │   ├── inference.py
│   │   ├── lrml.py
│   │   ├── main.py
│   │   ├── train.py
│   │   └── utils.py
│   └── MultiVAE
│       ├── config.py
│       ├── config.yaml
│       ├── dataset.py
│       ├── early_stopping.py
│       ├── main.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── README.md
└── requirements.txt
```

## ⚒️ Development Environment

- 서버 스펙 : AI Stage GPU (Tesla V100)
- 협업 툴 : Github / Zoom / Slack / Notion / Google Drive / MLflow
- 기술 스택 : Python / Scikit-Learn / Torch / Recbole