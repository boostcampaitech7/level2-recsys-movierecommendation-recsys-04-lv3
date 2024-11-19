import os

import mlflow


class MlflowManager:
    """
    MLflow 관리를 위한 클래스

    MLflow UI 실행 방법:
    mlflow ui -h 0.0.0.0 -p 30627

    Attributes:
        user_name: MLflow 사용자 이름 (str)
        tracking_uri: MLflow 추적 URI (str)
        experiment_name: MLflow 실험 이름 (str)

    Methods:
        init: MLflow 초기 설정
        start_run: MLflow 실행 시작
        log_params: 파라미터 로깅
        log_metric: 메트릭 로깅
        log_artifact: 아티팩트 로깅
        log_model: 모델 로깅
        end_run: MLflow 실행 종료
        autolog: MLflow 자동 로깅 설정
        get_tracking_uri: 현재 설정된 tracking URI 반환
        get_artifact_uri: 현재 아티팩트 URI 반환
    """

    def __init__(self, user_name="root", tracking_uri="http://10.28.224.212", port=30627, experiment_name="tmp"):
        """
        MlflowManager 클래스 초기화

        Args:
            user_name (str): MLflow 사용자 이름
            tracking_uri (str): MLflow 추적 서버 URI
            port (int): MLflow 추적 서버 포트
            experiment_name (str): MLflow 실험 이름
        """
        self.user_name = user_name
        os.environ["LOGNAME"] = self.user_name
        self.tracking_uri = f"{tracking_uri}:{port}"
        self.experiment_name = experiment_name
        self.init()

    def init(self):
        """MLflow 초기 설정"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None):
        """
        MLflow 실행 시작

        Args:
            run_name (str, optional): 실행 이름

        Returns:
            mlflow.ActiveRun: 활성화된 MLflow 실행 객체
        """
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        """
        파라미터 로깅

        Args:
            params (dict): 로깅할 파라미터 딕셔너리
        """
        mlflow.log_params(params)

    def log_metric(self, key, value, step):
        """
        메트릭 로깅

        Args:
            key (str): 메트릭 이름
            value (float): 메트릭 값
            step (int): 현재 스텝
        """
        mlflow.log_metric(key, value, step)

    def log_artifact(self, local_path):
        """
        아티팩트 로깅

        Args:
            local_path (str): 로깅할 아티팩트의 로컬 경로
        """
        mlflow.log_artifact(local_path)

    def log_model(self, model, artifact_path, type="torch"):
        """
        모델 로깅

        Args:
            model: 로깅할 모델 객체
            artifact_path (str): 아티팩트 저장 경로
            type (str): 모델 타입 ('torch' 또는 'lgb')

        Raises:
            Exception: 지원되지 않는 모델 타입일 경우 발생
        """
        if type == "torch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif type == "lgb":
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            raise Exception(f"Check type: {type}")
        return None

    def end_run(self):
        """MLflow 실행 종료"""
        mlflow.end_run()

    def get_tracking_uri(self):
        """
        현재 설정된 tracking URI 반환

        Returns:
            str: 현재 설정된 tracking URI
        """
        return mlflow.get_tracking_uri()

    def get_artifact_uri(self):
        """
        현재 아티팩트 URI 반환

        Returns:
            str: 현재 아티팩트 URI
        """
        return mlflow.get_artifact_uri()
