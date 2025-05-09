import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TaskAssigner:
    def __init__(self):
        self.skill_matrix = None
    
    def build_skill_matrix(self, developers: list, tasks: list):
        """Tạo ma trận kỹ năng developer-task"""
        # Vector hóa kỹ năng (ví dụ: [1,0,1] = biết Python, không biết SQL, biết Docker)
        dev_vectors = [d['skill_vector'] for d in developers]
        task_vectors = [t['required_skills'] for t in tasks]
        
        self.skill_matrix = cosine_similarity(task_vectors, dev_vectors)
        return self.skill_matrix
    
    def assign_tasks(self, developers: list, tasks: list):
        """Phân công công việc tối ưu"""
        if self.skill_matrix is None:
            self.build_skill_matrix(developers, tasks)
            
        assignments = []
        for task_idx, task in enumerate(tasks):
            best_dev_idx = np.argmax(self.skill_matrix[task_idx])
            assignments.append({
                'task_id': task['id'],
                'dev_id': developers[best_dev_idx]['id'],
                'fit_score': float(self.skill_matrix[task_idx][best_dev_idx])
            })
        return assignments