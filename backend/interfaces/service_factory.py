"""
Service Factory - Factory Pattern cho việc tạo service instances
Hỗ trợ Dependency Injection và dễ dàng testing với mock objects
"""

from typing import Dict, Type, Any, Optional
from interfaces import IAreaAnalysisService, IRiskAnalysisService, ITaskService
from interfaces.task_commit_service import ITaskCommitService


class ServiceFactory:
    """
    Factory class để tạo và quản lý service instances
    Áp dụng Singleton pattern để đảm bảo consistency
    """
    
    _instance: Optional['ServiceFactory'] = None
    _services: Dict[Type, Any] = {}
    
    def __new__(cls) -> 'ServiceFactory':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_service(self, interface: Type, implementation: Any) -> None:
        """
        Đăng ký service implementation cho một interface
        
        Args:
            interface: Interface type (VD: IAreaAnalysisService)
            implementation: Service implementation instance
        """
        self._services[interface] = implementation
    
    def get_service(self, interface: Type) -> Any:
        """
        Lấy service instance cho interface được chỉ định
        
        Args:
            interface: Interface type cần lấy
            
        Returns:
            Service instance
            
        Raises:
            ValueError: Khi interface chưa được đăng ký
        """
        if interface not in self._services:
            raise ValueError(f"Service for interface {interface.__name__} not registered")
        return self._services[interface]
    
    def clear_services(self) -> None:
        """Clear tất cả registered services - chủ yếu dùng cho testing"""
        self._services.clear()


# Singleton instance
service_factory = ServiceFactory()


def get_area_analysis_service() -> IAreaAnalysisService:
    """Dependency injection cho AreaAnalysisService"""
    return service_factory.get_service(IAreaAnalysisService)


def get_risk_analysis_service() -> IRiskAnalysisService:
    """Dependency injection cho RiskAnalysisService"""
    return service_factory.get_service(IRiskAnalysisService)


def get_task_service() -> ITaskService:
    """Dependency injection cho TaskService"""
    return service_factory.get_service(ITaskService)


def get_task_commit_service() -> ITaskCommitService:
    """Dependency injection cho TaskCommitService"""
    return service_factory.get_service(ITaskCommitService)
