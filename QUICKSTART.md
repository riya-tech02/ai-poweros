# AI-PowerOS Quick Start Guide

## 🚀 Fast Setup (5 minutes)

### 1. Start Everything
```bash
# Activate virtual environment
source venv/bin/activate

# Start all services
./deploy.sh start

# Start API (in new terminal)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the System
Open your browser to:
- **Dashboard**: http://localhost:8000/dashboard/
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### 3. Run Tests
```bash
# Run complete test suite
python test_complete_system.py

# Run specific tests
pytest tests/ -v
```

## 📱 Using the Dashboard

1. **Predict Activities**
   - Enter recent activities: `wake_up, coffee, exercise`
   - Click "Predict Next Activities"
   - See AI predictions with confidence scores

2. **Schedule Tasks**
   - Add tasks with name, priority, effort
   - Click "Schedule All Tasks"
   - Get optimized schedule with batching

3. **Track Habits**
   - Record habits: name and category
   - View habit patterns and strengths
   - See habit sequences

4. **Memory System**
   - Store important memories
   - Query past memories
   - Automatic importance scoring

## 🔧 Common Commands
```bash
# Check system status
./deploy.sh status

# Run all tests
./deploy.sh test

# Restart services
./deploy.sh restart

# Stop everything
./deploy.sh stop

# Format code
make format

# Run linter
make lint
```

## 📊 API Examples

### Predict Routine
```bash
curl -X POST http://localhost:8000/api/v1/advanced/routine/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "recent_activities": ["wake_up", "coffee"],
    "context": {"time_of_day": "morning"},
    "top_k": 5
  }'
```

### Schedule Tasks
```bash
curl -X POST http://localhost:8000/api/v1/advanced/schedule/intelligent \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "tasks": [
      {"id": "t1", "name": "Report", "priority": 0.9, "effort": 2}
    ]
  }'
```

### Record Habit
```bash
curl -X POST http://localhost:8000/api/v1/advanced/habits/record \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "habit": {
      "id": "h1",
      "name": "Exercise",
      "category": "health"
    },
    "context": {"time_of_day": "morning"}
  }'
```

## 🎯 What's Working

✅ Transformer-based routine prediction  
✅ RL task scheduling with PPO  
✅ Episodic memory system  
✅ Neo4j graph integration  
✅ Web dashboard  
✅ Real-time API  
✅ Docker deployment  
✅ Comprehensive tests  

## 🚨 Troubleshooting

**Port already in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Docker issues:**
```bash
docker-compose down -v
docker system prune -a
./deploy.sh start
```

**Module import errors:**
```bash
pip install -r requirements.txt --upgrade
```

## 📈 Next Steps

1. Train models on real data
2. Deploy to Kubernetes
3. Add more ML features
4. Connect to production databases
5. Enable authentication

Enjoy your AI-PowerOS! 🎉
