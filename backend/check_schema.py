from db.database import sync_engine
from sqlalchemy import text

def check_commit_columns():
    try:
        with sync_engine.connect() as conn:
            # Check the column lengths
            result = conn.execute(text("""
                SELECT column_name, data_type, character_maximum_length 
                FROM information_schema.columns 
                WHERE table_name = 'commits' 
                AND column_name IN ('sha', 'parent_sha')
                ORDER BY column_name
            """))
            
            columns = list(result)
            print("Commit table SHA columns:")
            for column in columns:
                print(f"  {column[0]}: {column[1]}({column[2]})")
            
            if not columns:
                print("No sha/parent_sha columns found in commits table")
                
    except Exception as e:
        print(f"Error checking schema: {e}")

if __name__ == "__main__":
    check_commit_columns()
