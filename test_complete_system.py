#!/usr/bin/env python3
"""
Complete System Test Suite
Tests all features of AI-PowerOS
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def test_health():
    """Test system health"""
    print_info("Testing system health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"System Status: {data['status']}")
            print_success(f"Version: {data['version']}")
            print_success(f"Features: {', '.join(data['features'])}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_basic_prediction():
    """Test basic routine prediction"""
    print_info("Testing basic routine prediction...")
    try:
        payload = {
            "user_id": "test_user",
            "context": {
                "time_of_day": "morning",
                "recent_activities": ["wake_up", "coffee"]
            },
            "top_k": 5
        }
        
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/routine",
            json=payload,
            timeout=10
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Predictions received: {len(data['predictions'])}")
            print_success(f"Latency: {latency:.2f}ms")
            print_success(f"Backend: {data['backend']}")
            
            print("\n  Top Predictions:")
            for pred in data['predictions'][:3]:
                print(f"    • {pred['activity']}: {pred['confidence']:.2%}")
            return True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Prediction error: {e}")
        return False

def test_advanced_prediction():
    """Test advanced transformer prediction"""
    print_info("Testing advanced transformer prediction...")
    try:
        payload = {
            "user_id": "test_user",
            "recent_activities": ["wake_up", "coffee", "exercise"],
            "context": {"time_of_day": "morning"},
            "top_k": 5
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/advanced/routine/advanced",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Advanced predictions: {len(data['predictions'])}")
            print_success(f"Latency: {data['latency_ms']:.2f}ms")
            print_success(f"Backend: {data['backend']}")
            print_success(f"Model version: {data['model_version']}")
            return True
        else:
            print_error(f"Advanced prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Advanced prediction error: {e}")
        return False

def test_task_scheduling():
    """Test intelligent task scheduling"""
    print_info("Testing task scheduling...")
    try:
        tasks = [
            {
                "id": "t1",
                "name": "Morning Report",
                "priority": 0.9,
                "effort": 2,
                "deadline": 12,
                "category": "work"
            },
            {
                "id": "t2",
                "name": "Team Meeting",
                "priority": 0.8,
                "effort": 1,
                "deadline": 10,
                "category": "work"
            },
            {
                "id": "t3",
                "name": "Code Review",
                "priority": 0.7,
                "effort": 3,
                "deadline": 16,
                "category": "development"
            },
            {
                "id": "t4",
                "name": "Documentation",
                "priority": 0.6,
                "effort": 2,
                "deadline": 18,
                "category": "development"
            }
        ]
        
        payload = {
            "user_id": "test_user",
            "tasks": tasks,
            "context": {"time_of_day": "morning"}
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/advanced/schedule/intelligent",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Tasks scheduled: {len(data['scheduled_tasks'])}")
            print_success(f"Completion rate: {data['completion_rate']:.1%}")
            print_success(f"Optimization score: {data['optimization_score']:.2f}")
            print_success(f"Algorithm: {data['algorithm']}")
            
            print("\n  Schedule:")
            for task in data['scheduled_tasks'][:4]:
                batched = " ⚡(batched)" if task.get('batched') else ""
                print(f"    • {task['name']}: {task['start_time']}-{task['end_time']}h{batched}")
            return True
        else:
            print_error(f"Scheduling failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Scheduling error: {e}")
        return False

def test_memory_system():
    """Test memory storage and retrieval"""
    print_info("Testing memory system...")
    try:
        # Store memory
        store_payload = {
            "content": "Had productive meeting about Q1 goals and team expansion",
            "context": {
                "user_initiated": True,
                "category": "work",
                "sentiment": 0.8
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/memory/store",
            json=store_payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Memory stored: {data['memory_id']}")
            
            # Query memory
            time.sleep(0.5)
            query_response = requests.get(
                f"{BASE_URL}/api/v1/predict/memory/query",
                params={"query": "meeting", "k": 5},
                timeout=5
            )
            
            if query_response.status_code == 200:
                query_data = query_response.json()
                print_success(f"Memories found: {query_data['count']}")
                
                if query_data['count'] > 0:
                    print("\n  Recent Memories:")
                    for mem in query_data['memories'][:2]:
                        print(f"    • {mem['content'][:50]}...")
                return True
            else:
                print_error("Memory query failed")
                return False
        else:
            print_error(f"Memory storage failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Memory system error: {e}")
        return False

def test_habit_tracking():
    """Test habit recording and retrieval"""
    print_info("Testing habit tracking...")
    try:
        # Record habit
        habit_payload = {
            "user_id": "test_user",
            "habit": {
                "id": f"habit_{int(time.time())}",
                "name": "Morning Exercise",
                "category": "health",
                "duration": 30,
                "completion": 1.0
            },
            "context": {
                "time_of_day": "morning",
                "location": "home"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/advanced/habits/record",
            json=habit_payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Habit recorded: {data['status']}")
            print_success(f"Backend: {data['backend']}")
            
            # Get habit patterns
            time.sleep(0.5)
            patterns_response = requests.get(
                f"{BASE_URL}/api/v1/advanced/habits/test_user/patterns",
                timeout=5
            )
            
            if patterns_response.status_code == 200:
                patterns_data = patterns_response.json()
                print_success(f"Habits retrieved: {len(patterns_data.get('habits', []))}")
                return True
            else:
                print_error("Habit retrieval failed")
                return False
        else:
            print_error(f"Habit recording failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Habit tracking error: {e}")
        return False

def test_dashboard():
    """Test dashboard accessibility"""
    print_info("Testing web dashboard...")
    try:
        response = requests.get(f"{BASE_URL}/dashboard/", timeout=5)
        
        if response.status_code == 200 and len(response.text) > 1000:
            print_success("Dashboard is accessible")
            print_success(f"Dashboard size: {len(response.text)} bytes")
            return True
        else:
            print_error("Dashboard not accessible")
            return False
    except Exception as e:
        print_error(f"Dashboard error: {e}")
        return False

def run_performance_tests():
    """Run performance benchmarks"""
    print_info("Running performance benchmarks...")
    
    latencies = []
    num_tests = 20
    
    for i in range(num_tests):
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            latency = (time.time() - start) * 1000
            if response.status_code == 200:
                latencies.append(latency)
        except:
            pass
    
    if latencies:
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        print_success(f"Average latency: {avg:.2f}ms")
        print_success(f"P95 latency: {p95:.2f}ms")
        print_success(f"Min latency: {min_lat:.2f}ms")
        print_success(f"Max latency: {max_lat:.2f}ms")
        return True
    else:
        print_error("Performance test failed")
        return False

def main():
    """Run all tests"""
    print_header("AI-PowerOS Complete System Test")
    
    start_time = time.time()
    
    tests = [
        ("Health Check", test_health),
        ("Basic Prediction", test_basic_prediction),
        ("Advanced Prediction", test_advanced_prediction),
        ("Task Scheduling", test_task_scheduling),
        ("Memory System", test_memory_system),
        ("Habit Tracking", test_habit_tracking),
        ("Web Dashboard", test_dashboard),
        ("Performance", run_performance_tests)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print_header(test_name)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append((test_name, False))
        time.sleep(0.5)
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    print(f"{Colors.BOLD}Success Rate: {passed/total*100:.1f}%{Colors.END}")
    
    elapsed = time.time() - start_time
    print(f"{Colors.BOLD}Total Time: {elapsed:.2f}s{Colors.END}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.END}\n")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Some tests failed{Colors.END}\n")

if __name__ == "__main__":
    main()
