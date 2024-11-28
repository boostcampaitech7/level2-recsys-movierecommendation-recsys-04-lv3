class EarlyStopping:
    """얼리 스토핑 클래스"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max') -> None:
        """
        얼리 스토핑을 위한 초기화 함수입니다.
        
        Args:
            patience (int): 개선되지 않을 때 허용하는 에폭 수. 기본값은 5입니다.
            min_delta (float): 개선으로 간주되는 최소 변화량. 기본값은 0.001입니다.
            mode (str): 'max'는 recall과 같이 높을수록 좋은 메트릭, 'min'은 loss와 같이 낮을수록 좋은 메트릭. 기본값은 'max'입니다.
        """
        self.patience: int = patience  # 개선되지 않을 때 허용하는 에폭 수
        self.min_delta: float = min_delta  # 개선으로 간주되는 최소 변화량
        self.mode: str = mode  # 'max' 또는 'min' 모드
        self.counter: int = 0  # 개선되지 않은 에폭 수
        self.best_score: float | None = None  # 최고 점수 (최초엔 None)
        self.early_stop: bool = False  # 얼리 스토핑 여부 (초기값은 False)

    def __call__(self, score: float) -> bool:
        """
        현재 스코어가 이전 최고 스코어보다 개선되었는지 확인하고, 개선되지 않으면 얼리 스토핑 여부를 결정합니다.
        
        Args:
            score (float): 현재 스코어 값입니다.
        
        Returns:
            bool: 얼리 스토핑을 적용해야 하는지 여부. 개선되지 않으면 True, 그렇지 않으면 False.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False
