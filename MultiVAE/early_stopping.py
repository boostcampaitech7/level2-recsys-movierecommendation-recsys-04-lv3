# early_stopping.py
class EarlyStopping:
    """얼리 스토핑 클래스"""
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        """
        patience: 개선되지 않을 때 허용하는 에폭 수
        min_delta: 개선으로 간주되는 최소 변화량
        mode: 'max'는 recall과 같이 높을수록 좋은 메트릭, 'min'은 loss와 같이 낮을수록 좋은 메트릭
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        현재 스코어가 이전 최고 스코어보다 개선되었는지 확인
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