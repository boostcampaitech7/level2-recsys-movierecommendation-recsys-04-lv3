import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiVAE(nn.Module):
    """
    MultiVAE 모델을 구현한 클래스입니다.
    이 모델은 다중 아이템 추천 시스템에서 사용될 수 있는 VAE 기반의 모델입니다.

    이 모델은 입력 데이터를 잠재 공간으로 매핑하고, 이를 통해 아이템에 대한 예측을 수행합니다. 
    VAE는 데이터의 확률적 특성을 고려하여 더 일반화된 학습을 가능하게 합니다.

    주요 구성 요소:
        - Encoder: 입력 데이터를 잠재 공간으로 인코딩
        - Decoder: 잠재 공간에서 데이터를 복원
        - reparameterize: 잠재 변수 샘플링을 위한 재파라미터화 기법

    메서드:
        - `__init__`: 모델 초기화
        - `reparameterize`: 잠재 공간에서 샘플을 샘플링하는 함수
        - `forward`: 입력 데이터를 인코딩하고 디코딩하여 복원된 값을 반환하는 함수
    """
    
    def __init__(self, num_items: int, hidden_dim: int = 600, latent_dim: int = 200, dropout_prob: float = 0.5) -> None:
        """
        MultiVAE 모델을 초기화하는 함수입니다.

        Args:
            num_items (int): 아이템 수, 입력 크기입니다.
            hidden_dim (int): 은닉층의 차원 크기입니다. 기본값은 600입니다.
            latent_dim (int): 잠재 공간의 차원 크기입니다. 기본값은 200입니다.
            dropout_prob (float): 드롭아웃 확률입니다. 기본값은 0.5입니다.
        """
        super(MultiVAE, self).__init__()

        # 입력 데이터의 아이템 수, 은닉 차원, 잠재 차원, 드롭아웃 확률을 설정
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob

        # Encoder(인코더) 네트워크 구성
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),  # 입력 차원 -> 은닉 차원
            nn.Tanh(),                        # 활성화 함수: Tanh
            nn.Dropout(dropout_prob),          # 드롭아웃 적용
            nn.Linear(hidden_dim, latent_dim * 2)  # 은닉 차원 -> 잠재 차원의 두 배 크기(평균과 로그 분산)
        )

        # Decoder(디코더) 네트워크 구성
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # 잠재 차원 -> 은닉 차원
            nn.Tanh(),                         # 활성화 함수: Tanh
            nn.Dropout(dropout_prob),           # 드롭아웃 적용
            nn.Linear(hidden_dim, num_items)    # 은닉 차원 -> 아이템 수 (출력 크기)
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        재파라미터화 기법을 사용하여 잠재 공간에서 샘플을 생성하는 함수입니다.
        
        Args:
            mu (Tensor): 평균 벡터, 크기는 (배치 크기, 잠재 차원)입니다.
            logvar (Tensor): 로그 분산 벡터, 크기는 (배치 크기, 잠재 차원)입니다.

        Returns:
            Tensor: 샘플링된 잠재 벡터입니다.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)  # 표준편차 계산
            epsilon = torch.randn_like(std)  # 표준정규분포에서 샘플링
            return mu + epsilon * std  # 재파라미터화된 벡터 반환
        else:
            return mu  # 학습 모드가 아닐 경우, 평균을 반환

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        모델의 순전파 함수입니다. 입력을 인코딩하고 디코딩하여 복원된 값을 반환합니다.

        Args:
            x (Tensor): 입력 데이터, 크기는 (배치 크기, 아이템 수)입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]: 복원된 입력 데이터(아이템 수 크기), 잠재 공간의 평균 벡터(mu), 
                                           잠재 공간의 로그 분산 벡터(logvar)
        """
        # 입력 데이터 정규화 (아이템 간의 차이를 최소화하기 위해)
        x = F.normalize(x, dim=1)

        # Encoder(인코더) 네트워크를 통해 잠재 공간의 평균(mu)와 로그 분산(logvar) 계산
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]  # 평균 벡터
        logvar = h[:, self.latent_dim:]  # 로그 분산 벡터

        # Reparameterize(재파라미터화)하여 잠재 변수 z 샘플링
        z = self.reparameterize(mu, logvar)

        # Decoder(디코더) 네트워크를 통해 잠재 변수 z로부터 입력 데이터를 복원
        x_recon = self.decoder(z)

        # 복원된 입력, 평균, 로그 분산 반환
        return x_recon, mu, logvar
