import re

# Read the file
with open(r'c:\SAN\KLTN\KLTN04\backend\api\routes\assignment_recommendation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to match the ORM query pattern
pattern = r'''        # Get repository from owner/repo_name
        repo = db\.query\(Repository\)\.filter\(
            Repository\.owner == owner,
            Repository\.name == repo_name
        \)\.first\(\)
        
        if not repo:
            raise HTTPException\(status_code=404, detail=f"Repository \{owner\}/\{repo_name\} not found"\)'''

replacement = '''        # Get repository from owner/repo_name
        query = select(repositories).where(
            repositories.c.owner == owner,
            repositories.c.name == repo_name
        )
        result = db.execute(query).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name} not found")
        
        repo_id = result.id'''

# Replace all occurrences
content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Replace all repo.id with repo_id
content = re.sub(r'repo\.id', 'repo_id', content)

# Write back
with open(r'c:\SAN\KLTN\KLTN04\backend\api\routes\assignment_recommendation.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File updated successfully!")
