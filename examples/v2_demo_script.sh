#!/bin/bash
# CogniQuantum V2 Demonstration Script

echo "🧠⚛️ CogniQuantum V2 Demonstration"
echo "=================================="

# System Health Check
echo "1. System Health Check..."
python quick_test_v2.py

echo -e "\n2. Testing V2 Modes with Different Complexity Levels..."

# Low Complexity - Efficient Mode
echo -e "\n🟢 LOW COMPLEXITY - Efficient Mode:"
python fetch_llm_v2.py ollama "What is 2+2?" --mode efficient --model gemma3

# Medium Complexity - Balanced Mode  
echo -e "\n🟡 MEDIUM COMPLEXITY - Balanced Mode:"
python fetch_llm_v2.py ollama "Explain the main causes of climate change" --mode balanced --model phi4-mini-reasoning

# High Complexity - Decomposed Mode
echo -e "\n🔴 HIGH COMPLEXITY - Decomposed Mode:"
python fetch_llm_v2.py ollama "Design a sustainable transportation system" --mode decomposed --model deepseek-r1

# Paper Optimization
echo -e "\n🎓 PAPER-OPTIMIZED Mode:"
python fetch_llm_v2.py ollama "Analyze AI and quantum computing integration" --mode paper_optimized --model deepseek-r1

echo -e "\n✅ V2 Demonstration Complete!"
echo "For more examples, see examples/sample_questions.txt"