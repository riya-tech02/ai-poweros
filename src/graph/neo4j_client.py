"""
Neo4j Graph Database Client
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

try:
    from py2neo import Graph, Node, Relationship

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("py2neo not available")


class Neo4jClient:
    """Client for Neo4j graph operations"""

    def __init__(self, uri: str, user: str, password: str):
        if not NEO4J_AVAILABLE:
            self.graph = None
            logging.warning("Neo4j client not initialized")
            return

        try:
            self.graph = Graph(uri, auth=(user, password))
            self._initialize_schema()
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            self.graph = None

    def _initialize_schema(self):
        """Create constraints and indexes"""
        if not self.graph:
            return

        constraints = [
            """CREATE CONSTRAINT user_id IF NOT EXISTS
               FOR (u:User) REQUIRE u.id IS UNIQUE""",
            """CREATE CONSTRAINT habit_id IF NOT EXISTS
               FOR (h:Habit) REQUIRE h.id IS UNIQUE""",
            """CREATE CONSTRAINT task_id IF NOT EXISTS
               FOR (t:Task) REQUIRE t.id IS UNIQUE""",
        ]

        for constraint in constraints:
            try:
                self.graph.run(constraint)
            except Exception:
                pass

    def add_habit_event(self, user_id: str, habit: Dict, context: Dict):
        """Record habit performance"""
        if not self.graph:
            return

        query = """
        MERGE (u:User {id: $user_id})
        MERGE (h:Habit {
            id: $habit_id,
            name: $habit_name,
            category: $category
        })
        
        CREATE (u)-[r:PERFORMED {
            timestamp: datetime($timestamp),
            duration: $duration,
            completion: $completion
        }]->(h)
        
        SET h.strength = COALESCE(h.strength, 0) + 0.1,
            h.last_performed = datetime($timestamp),
            h.frequency = COALESCE(h.frequency, 0) + 1
        
        RETURN h
        """

        try:
            self.graph.run(
                query,
                {
                    "user_id": user_id,
                    "habit_id": habit.get("id", "habit_unknown"),
                    "habit_name": habit.get("name", "Unknown"),
                    "category": habit.get("category", "general"),
                    "timestamp": datetime.now().isoformat(),
                    "duration": habit.get("duration", 0),
                    "completion": habit.get("completion", 1.0),
                },
            )
        except Exception as e:
            logging.error(f"Failed to add habit event: {e}")

    def get_user_habits(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's habits"""
        if not self.graph:
            return []

        query = """
        MATCH (u:User {id: $user_id})-[:PERFORMED]->(h:Habit)
        RETURN h.name as name,
               h.category as category,
               h.strength as strength,
               h.frequency as frequency
        ORDER BY h.strength DESC
        LIMIT $limit
        """

        try:
            results = self.graph.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in results]
        except Exception as e:
            logging.error(f"Failed to get habits: {e}")
            return []

    def get_habit_sequences(
        self, user_id: str, min_support: int = 3
    ) -> List[List[str]]:
        """Get frequent habit sequences"""
        if not self.graph:
            return []

        query = """
        MATCH path = (u:User {id: $user_id})
                     -[:PERFORMED*2..5]->(habits)
        WITH habits, count(*) as frequency
        WHERE frequency >= $min_support
        RETURN habits, frequency
        ORDER BY frequency DESC
        LIMIT 20
        """

        try:
            results = self.graph.run(query, user_id=user_id, min_support=min_support)
            sequences = [
                [node["name"] for node in record["habits"]] for record in results
            ]
            return sequences
        except Exception as e:
            logging.error(f"Failed to get sequences: {e}")
            return []

    def predict_next_habits(
        self, user_id: str, current_context: Dict, top_k: int = 5
    ) -> List[Dict]:
        """Predict next likely habits"""
        if not self.graph:
            return []

        query = """
        MATCH (u:User {id: $user_id})-[r:PERFORMED]->(h:Habit)
        WHERE r.timestamp > datetime() - duration({days: 30})
        WITH h, count(r) as recent_frequency, h.strength as strength
        
        RETURN h.id as habit_id,
               h.name as habit_name,
               h.category as category,
               (recent_frequency / 30.0 * 0.5 + 
                strength * 0.5) as prediction_score
        ORDER BY prediction_score DESC
        LIMIT $top_k
        """

        try:
            results = self.graph.run(query, user_id=user_id, top_k=top_k)
            return [dict(record) for record in results]
        except Exception as e:
            logging.error(f"Failed to predict habits: {e}")
            return []


# Initialize client (will connect when config is loaded)
neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Optional[Neo4jClient]:
    """Get or create Neo4j client"""
    global neo4j_client

    if neo4j_client is None:
        from src.core.config import settings

        neo4j_client = Neo4jClient(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
        )

    return neo4j_client
