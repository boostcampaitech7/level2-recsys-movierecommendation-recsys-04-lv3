import yaml
from typing import Any

class Config:
    """
    YAML 파일에서 설정을 로드하고, 속성처럼 접근 가능하도록 하는 클래스.

    Attributes:
        _config (dict[str, Any]): YAML 파일에서 로드된 설정 데이터.

    Methods:
        __init__(config_path: str = 'config.yaml') -> None:
            Config 객체를 초기화하고 YAML 파일을 로드합니다.
        __getattr__(name: str) -> Any:
            설정 데이터를 속성처럼 접근 가능하도록 합니다.
    """

    def __init__(self, config_path: str = 'config.yaml') -> None:
        with open(config_path, 'r') as file:
            self._config: dict[str, Any] = yaml.safe_load(file)

    def __getattr__(self, name: str) -> Any:
        return self._config.get(name)
