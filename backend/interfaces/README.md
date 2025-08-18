# Interface Architecture - KLTN04

## Tổng quan
Thư mục `interfaces/` chứa các interface định nghĩa contracts cho các service trong hệ thống. Việc sử dụng interface giúp:

- **Dependency Inversion Principle**: High-level modules không phụ thuộc vào low-level modules
- **Testability**: Dễ dàng mock và test các component
- **Flexibility**: Có thể thay đổi implementation mà không ảnh hưởng đến code sử dụng
- **Maintainability**: Code dễ bảo trì và mở rộng

## Cấu trúc

```
interfaces/
├── __init__.py                           # Export các interface chính
├── area_analysis_service_interface.py    # Interface cho Area Analysis Service
├── risk_analysis_service_interface.py   # Interface cho Risk Analysis Service  
├── task_service_interface.py            # Interface cho Task Service
└── service_factory.py                   # Factory Pattern cho DI
```

## Cách sử dụng

### 1. Trong API Routes

```python
from fastapi import APIRouter, Depends
from interfaces.service_factory import get_area_analysis_service
from interfaces import IAreaAnalysisService

@router.post("/predict")
async def predict_area(
    commit_data: CommitData,
    service: IAreaAnalysisService = Depends(get_area_analysis_service)
):
    result = service.predict_area(commit_data.dict())
    return {"area": result}
```

### 2. Trong Testing

```python
from unittest.mock import Mock
from interfaces import IAreaAnalysisService

def test_predict_area():
    # Arrange
    mock_service = Mock(spec=IAreaAnalysisService)
    mock_service.predict_area.return_value = "backend"
    
    # Act
    result = mock_service.predict_area({"message": "test"})
    
    # Assert
    assert result == "backend"
```

### 3. Đăng ký Service mới

```python
from interfaces.service_factory import service_factory
from interfaces import IAreaAnalysisService

# Trong file khởi tạo (deps.py)
new_service = NewAreaAnalysisService()
service_factory.register_service(IAreaAnalysisService, new_service)
```

## Design Patterns được áp dụng

### 1. Interface Segregation Principle
Mỗi interface chỉ chứa các method cần thiết cho một chức năng cụ thể.

### 2. Dependency Injection
Sử dụng FastAPI's `Depends()` để inject service thông qua interface.

### 3. Factory Pattern
`ServiceFactory` quản lý việc tạo và cung cấp service instances.

### 4. Singleton Pattern
Factory và các service được implement như singleton để tối ưu memory.

## Các Interface hiện có

### IAreaAnalysisService
```python
@abstractmethod
def predict_area(self, commit_data: Dict) -> str:
    """Dự đoán phạm vi công việc (dev area) từ commit data"""
    pass
```

### IRiskAnalysisService
```python
@abstractmethod
def predict_risk(self, commit_data: Dict) -> str:
    """Dự đoán độ rủi ro (lowrisk/highrisk) từ commit data"""
    pass
```

### ITaskService
```python
@abstractmethod
async def create_task(self, task_data: TaskCreate, ...) -> TaskResponse:
    """Tạo task mới với validation"""
    pass

@abstractmethod 
async def get_task_by_id(self, task_id: int, ...) -> TaskResponse:
    """Lấy task theo ID"""
    pass
```

## Best Practices

### 1. Luôn sử dụng Interface trong Type Hints
```python
# ✅ Tốt
def process_data(service: IAreaAnalysisService) -> str:
    return service.predict_area(data)

# ❌ Tránh
def process_data(service: AreaAnalysisService) -> str:
    return service.predict_area(data)
```

### 2. Mock Interface khi Testing
```python
# ✅ Mock interface, không mock implementation
mock_service = Mock(spec=IAreaAnalysisService)

# ❌ Tránh mock concrete class
mock_service = Mock(spec=AreaAnalysisService)
```

### 3. Validate Interface Implementation
Đảm bảo concrete class implement đầy đủ interface methods:

```python
class AreaAnalysisService(IAreaAnalysisService):
    def predict_area(self, commit_data: Dict) -> str:
        # Implementation
        pass
```

## Mở rộng hệ thống

### Thêm Interface mới
1. Tạo file interface mới trong `interfaces/`
2. Add vào `__init__.py`
3. Tạo dependency function trong `service_factory.py`
4. Đăng ký service trong `deps.py`

### Thay đổi Implementation
Chỉ cần tạo class mới implement interface và đăng ký lại:

```python
class NewAreaAnalysisService(IAreaAnalysisService):
    def predict_area(self, commit_data: Dict) -> str:
        # New implementation
        pass

# Đăng ký lại
service_factory.register_service(IAreaAnalysisService, NewAreaAnalysisService())
```

## Kết luận

Việc tái cấu trúc này giúp:
- Code dễ test hơn với mock objects
- Dễ dàng thay đổi implementation
- Tuân thủ SOLID principles
- Chuẩn bị cho việc scale hệ thống

Tất cả existing code sẽ tiếp tục hoạt động bình thường nhưng bây giờ đã có foundation tốt cho việc mở rộng và bảo trì.
