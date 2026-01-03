"""Initialize Neo4j database"""
from py2neo import Graph
import os

def init_neo4j():
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "ai-poweros-neo4j-2024")
        
        graph = Graph(uri, auth=(user, password))
        
        constraints = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT habit_id IF NOT EXISTS FOR (h:Habit) REQUIRE h.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                graph.run(constraint)
                print(f"✓ Created: {constraint[:50]}...")
            except Exception as e:
                print(f"⚠ Skipped: {str(e)[:50]}...")
        
        print("✓ Neo4j initialization complete")
    except Exception as e:
        print(f"✗ Neo4j initialization failed: {e}")

if __name__ == "__main__":
    init_neo4j()
