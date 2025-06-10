from models.commit_model import CommitClassifier
from models.task_model import TaskAssigner
from fastapi import APIRouter

router = APIRouter()
commit_model = CommitClassifier.load()
task_model = TaskAssigner()

@router.post("/analyze-commits")
async def analyze_commits(messages: list[str]):
    predictions = commit_model.predict(messages)
    return {"predictions": predictions.tolist()}

@router.post("/assign-tasks")
async def assign_tasks(developers: list, tasks: list):
    assignments = task_model.assign_tasks(developers, tasks)
    return {"assignments": assignments}