#!/usr/bin/env python3
"""Test ML endpoints"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1/predict"

print("\n🤖 Testing AI-PowerOS ML Endpoints\n")
print("=" * 70)

# Test 1: Routine Prediction
print("\n1. Testing routine prediction...")
payload = {
    "user_id": "user123",
    "context": {
        "time_of_day": "morning",
        "recent_activities": ["wake_up", "coffee"]
    },
    "top_k": 5
}
response = requests.post(f"{BASE_URL}/routine", json=payload)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Latency: {data['latency_ms']}ms")
    print(f"   Backend: {data['backend']}")
    print("   Predictions:")
    for pred in data['predictions']:
        print(f"     - {pred['activity']}: {pred['confidence']:.2%}")

# Test 2: Task Scheduling
print("\n2. Testing task scheduling...")
payload = {
    "user_id": "user123",
    "tasks": [
        {"id": "t1", "name": "Morning Report", "priority": 0.9, "effort": 2},
        {"id": "t2", "name": "Team Meeting", "priority": 0.8, "effort": 1},
        {"id": "t3", "name": "Code Review", "priority": 0.7, "effort": 3}
    ]
}
response = requests.post(f"{BASE_URL}/schedule", json=payload)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Completion Rate: {data['completion_rate']:.1%}")
    print("   Schedule:")
    for task in data['scheduled_tasks']:
        print(f"     - {task['name']}: {task['start_time']}-{task['end_time']}h")

# Test 3: Store Memory
print("\n3. Testing memory storage...")
payload = {
    "content": "Had productive meeting about Q1 goals",
    "context": {
        "user_initiated": True,
        "category": "work"
    }
}
response = requests.post(f"{BASE_URL}/memory/store", json=payload)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    print(f"   Memory ID: {response.json()['memory_id']}")

# Test 4: Query Memory
print("\n4. Testing memory query...")
response = requests.get(f"{BASE_URL}/memory/query?query=meeting&k=5")
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Found {data['count']} memories")

print("\n" + "=" * 70)
print("✅ ML endpoint tests completed!\n")
