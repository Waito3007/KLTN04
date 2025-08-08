import logging
from ai.testmodelAi.han_model_real_test_fixed import load_han_model, predict_with_real_model, SimpleTokenizer
import sys
sys.modules['SimpleTokenizer'] = SimpleTokenizer

logger = logging.getLogger(__name__)

# Map label from model to commit type and tech area
COMMIT_TYPE_LABELS = {
    'feat': 'feat', 'feature': 'feat',
    'fix': 'fix', 'bug': 'fix',
    'chore': 'chore',
    'docs': 'docs',
    'refactor': 'refactor',
    'test': 'test',
    'style': 'style',
    'other': 'other',
}
TYPE_ICONS = {
    'feat': '🚀', 'fix': '🐛', 'chore': '🔧', 'docs': '📝', 'refactor': '♻️', 'test': '✅', 'style': '💄', 'other': '📦'
}
TECH_AREA_LABELS = {
    'api': 'API', 'endpoint': 'API', 'rest': 'API', 'graphql': 'API',
    'ui': 'Frontend', 'frontend': 'Frontend', 'react': 'Frontend', 'vue': 'Frontend', 'component': 'Frontend',
    'database': 'Database', 'db': 'Database', 'sql': 'Database', 'migration': 'Database',
    'test': 'Testing', 'testing': 'Testing', 'spec': 'Testing', 'unittest': 'Testing',
}

def guess_tech_area(message):
    msg = message.lower()
    for key, val in TECH_AREA_LABELS.items():
        if key in msg:
            return val
    return 'General'

class HANRealAIService:
    def __init__(self):
        self.model, self.tokenizer, self.label_encoders = load_han_model()
        self.is_model_loaded = self.model is not None

    def _map_ai_result(self, ai_result, message=None):
        # Map model output to commit type, tech area, ...
        commit_type = ai_result.get('type') or ai_result.get('commit_type') or ai_result.get('label') or 'other'
        commit_type = COMMIT_TYPE_LABELS.get(str(commit_type).lower(), 'other')
        # Ưu tiên lấy tech_area từ AI, nếu không có thì đoán từ message
        tech_area = ai_result.get('tech_area')
        if not tech_area and message:
            tech_area = guess_tech_area(message)
        if not tech_area:
            tech_area = 'General'
        return {
            'type': commit_type,
            'type_icon': TYPE_ICONS.get(commit_type, '📦'),
            'tech_area': tech_area,
            'impact': ai_result.get('impact', 'medium'),
            'urgency': ai_result.get('urgency', 'normal'),
            'ai_powered': True
        }

    async def analyze_commit_message(self, message: str) -> dict:
        if not self.is_model_loaded:
            return {
                'success': False,
                'message': message,
                'error': 'HAN model not loaded',
                'analysis': None
            }
        try:
            result = predict_with_real_model(self.model, self.tokenizer, self.label_encoders, message)
            # result là dict các task, lấy task chính (commit_type hoặc type)
            if isinstance(result, dict):
                # Ưu tiên lấy task 'commit_type', 'type', hoặc task đầu tiên
                main_task = None
                for key in ['commit_type', 'type']:
                    if key in result:
                        main_task = result[key]
                        break
                if not main_task and result:
                    main_task = list(result.values())[0]
                if main_task and isinstance(main_task, dict):
                    mapped = self._map_ai_result(main_task, message)
                else:
                    mapped = self._map_ai_result(result, message)
            else:
                mapped = self._map_ai_result({}, message)
            return {
                'success': True,
                'message': message,
                'analysis': mapped,
                'model_version': 'HAN-Real-v1',
                'confidence': {k: v['confidence'] for k, v in result.items() if isinstance(v, dict) and 'confidence' in v}
            }
        except Exception as e:
            logger.error(f"Error analyzing commit: {e}")
            return {
                'success': False,
                'message': message,
                'error': str(e),
                'analysis': None
            }

    async def analyze_commits_batch(self, messages: list) -> dict:
        results = []
        for msg in messages:
            analysis = await self.analyze_commit_message(msg)
            results.append(analysis)
        return {
            'success': True,
            'total_commits': len(messages),
            'results': results
        }
