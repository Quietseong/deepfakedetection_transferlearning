"""
설정 파일 로더 모듈

YAML 설정 파일을 로드하고 관리하는 유틸리티를 제공합니다.
"""

from typing import Any, Dict
from pathlib import Path
import yaml


class ConfigLoader:
    """
    YAML 설정 파일 로더
    
    Args:
        config_path: 설정 파일 경로
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드
        
        Returns:
            설정 딕셔너리
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 가져오기
        
        중첩된 키는 점(.)으로 구분합니다.
        예: "model.type", "training.optimizer.lr"
        
        Args:
            key: 설정 키
            default: 기본값
            
        Returns:
            설정 값
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """
        딕셔너리처럼 접근
        
        Args:
            key: 설정 키
            
        Returns:
            설정 값
        """
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """
        키 존재 여부 확인
        
        Args:
            key: 설정 키
            
        Returns:
            존재하면 True, 아니면 False
        """
        return key in self.config
    
    def save(self, output_path: str) -> None:
        """
        설정을 YAML 파일로 저장
        
        Args:
            output_path: 출력 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        설정 업데이트
        
        Args:
            updates: 업데이트할 설정 딕셔너리
        """
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """
        딕셔너리 깊은 업데이트
        
        Args:
            base_dict: 기본 딕셔너리
            update_dict: 업데이트 딕셔너리
        """
        for key, value in update_dict.items():
            if (
                key in base_dict 
                and isinstance(base_dict[key], dict) 
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


