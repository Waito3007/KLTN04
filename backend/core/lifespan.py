from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.database import database
import logging

logger = logging.getLogger(__name__)

def preload_commit_analyst_dependencies():
    """Preload dependencies for commit analyst services"""
    try:
        logger.info("🔄 Preloading commit analyst dependencies...")
        
        # Preload HAN model dependencies
        try:
            import torch
            from transformers import DistilBertTokenizer, DistilBertModel
            logger.info("✅ PyTorch and transformers loaded for HAN model.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to preload HAN dependencies: {e}")
        
        # Preload MultiFusion model dependencies
        try:
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.metrics import classification_report
            logger.info("✅ Scikit-learn components loaded for MultiFusion model.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to preload MultiFusion dependencies: {e}")
            
        logger.info("✅ Commit analyst dependencies preloaded successfully.")
        
    except Exception as e:
        logger.error(f"❌ Error preloading commit analyst dependencies: {e}")

def load_ai_models():
    """Load all AI models during startup"""
    try:
        # Preload commit analyst dependencies first
        preload_commit_analyst_dependencies()
        
        # Load Area Analysis Service
        logger.info("🔄 Loading Area Analysis Service...")
        from services.area_analysis_service import AreaAnalysisService
        area_service = AreaAnalysisService()
        logger.info("✅ Area Analysis Service loaded successfully.")
        
        # Load Risk Analysis Service
        logger.info("🔄 Loading Risk Analysis Service...")
        from services.risk_analysis_service import RiskAnalysisService
        risk_service = RiskAnalysisService()
        logger.info("✅ Risk Analysis Service loaded successfully.")
        
        # Load HAN AI Service
        logger.info("🔄 Loading HAN AI Service...")
        from services.han_ai_service import HANAIService
        han_service = HANAIService()
        logger.info("✅ HAN AI Service loaded successfully.")
        
        # Load HAN Commit Analyst Service
        logger.info("🔄 Loading HAN Commit Analyst Service...")
        try:
            from services.han_commitanalyst_service import HanCommitAnalystService
            # Note: HanCommitAnalystService requires db session, so we'll just import for now
            logger.info("✅ HAN Commit Analyst Service imported successfully.")
        except Exception as e:
            logger.warning(f"⚠️ HAN Commit Analyst Service import failed: {e}")
        
        # Load MultiFusion Commit Analyst Service
        logger.info("🔄 Loading MultiFusion Commit Analyst Service...")
        try:
            from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService
            # Note: MultifusionCommitAnalystService requires db session, so we'll just import for now
            logger.info("✅ MultiFusion Commit Analyst Service imported successfully.")
        except Exception as e:
            logger.warning(f"⚠️ MultiFusion Commit Analyst Service import failed: {e}")
        
        logger.info("🎉 All AI models and services loaded successfully during startup!")
        
    except Exception as e:
        logger.error(f"❌ Error loading AI models: {e}")
        # Don't raise here - let the app continue even if some models fail

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await database.connect()
        logger.info("✅ Đã kết nối tới database thành công.")
        
        # Load AI models after database connection
        load_ai_models()
        
        yield  # Chỉ yield nếu connect thành công
    except Exception as e:
        logger.error(f"❌ Kết nối database thất bại: {e}")
        raise e  # Dừng app nếu không kết nối được DB
    finally:
        try:
            await database.disconnect()
            logger.info("🛑 Đã ngắt kết nối database.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi ngắt kết nối database: {e}")
