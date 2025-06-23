import logging
from ai.multimodal_fusion_inference import MultiModalFusionInference

logger = logging.getLogger(__name__)

class FusionAIService:
    def __init__(self):
        self.infer = MultiModalFusionInference()
        self.is_model_loaded = True

    async def analyze_commit_message(self, message: str, metadata: dict = None):
        try:
            result = self.infer.predict_commit_analysis(message, metadata)
            return {
                'success': True,
                'message': message,
                'analysis': result,
                'model_version': 'MultimodalFusion-v1',
                'confidence': {k: v for k, v in result.items() if k.endswith('_probs')}
            }
        except Exception as e:
            logger.error(f"Error analyzing commit: {e}")
            return {
                'success': False,
                'message': message,
                'error': str(e),
                'analysis': None
            }

    async def analyze_commits_batch(self, messages: list):
        results = []
        for msg in messages:
            analysis = await self.analyze_commit_message(msg)
            results.append(analysis)
        return {
            'success': True,
            'total_commits': len(messages),
            'results': results
        }
