# OpenAPI Improvements for Open WebUI Integration

## Summary
Enhanced OpenAPI endpoint descriptions to make the LLM automatically understand **when** and **why** to invoke memory tools during conversations.

## Changes Made

### 1. `/add` Endpoint - Save Memories
**New Summary:** "Save user memories from conversation"

**Key Improvements:**
- ✅ Explicit "CALL THIS AUTOMATICALLY" instruction
- ✅ Clear examples of what triggers the tool ("I love pizza", "My name is John")
- ✅ Examples of what NOT to save ("What's the weather?", "Hello")
- ✅ Explains the extraction, deduplication, and filtering process
- ✅ Encourages liberal usage ("Better to call it and save nothing than miss context")

**When LLM should call this:**
- User shares preferences, personal info, goals, relationships, opinions
- User provides context about their life (work, hobbies, skills, habits)

### 2. `/relevant` Endpoint - Retrieve Context
**New Summary:** "Retrieve context-relevant memories"

**Key Improvements:**
- ✅ Explicit "CALL THIS BEFORE RESPONDING" instruction
- ✅ Clear use cases (personalization, recommendations, continuing conversations)
- ✅ Examples of when to call vs when to skip
- ✅ Best practice guidance

**When LLM should call this:**
- At the start of complex conversations
- When user asks about preferences
- Before giving personalized advice or recommendations
- When user references past interactions

### 3. `/search` Endpoint - Advanced Search
**New Summary:** "Search memories with filters"

**Key Improvements:**
- ✅ Clear distinction from `/relevant` endpoint
- ✅ Explicit examples of user queries that should trigger search
- ✅ Feature highlights (hybrid search, filtering, pagination)

**When LLM should call this:**
- User explicitly asks to search: "What do you know about my work?"
- User wants filtered results: "Show my food preferences"
- User asks "What do you remember about X?"

### 4. Request Model Enhancements

**AddMemoryRequest:**
- Better field descriptions with examples
- Clear explanation of user_id, agent_id, run_id usage
- Example format for messages array

**SearchRequest:**
- Descriptive field examples ("food preferences", "programming skills")
- Clear parameter ranges and purposes

**GetRelevantRequest:**
- Context-focused descriptions
- Clear purpose for each parameter

## Expected Behavior

### Before Changes:
- LLM rarely invoked memory tools
- Missed saving important context like "I love X"
- Had to manually prompt LLM to save memories

### After Changes:
- LLM should automatically detect saveable information
- Proactively call `/add` when users share preferences/facts
- Consider calling `/relevant` for personalized responses
- Understand distinction between search types

## Testing the Changes

### 1. Deploy the Updated API
```bash
# Make sure you're in the project directory
cd smartmemoryapi

# If using Docker
docker-compose up --build

# Or run directly
uvicorn app:app --reload --port 8000
```

### 2. Add to Open WebUI
1. Go to **⚙️ Admin Settings → External Tools**
2. Add new OpenAPI tool:
   - Name: SmartMemory
   - URL: `http://your-server:8000/openapi.json`
   - Enable the tool

### 3. Test Conversations

**Test Case 1: Saving Preferences**
```
User: "I love spicy food"
Expected: LLM calls /add endpoint
Result: Memory saved with category "food_preferences"
```

**Test Case 2: Personalized Response**
```
User: "What should I eat for dinner?"
Expected: LLM calls /relevant to check food preferences
Result: Response mentions "spicy food" preference
```

**Test Case 3: Memory Recall**
```
User: "What do you know about my food preferences?"
Expected: LLM calls /search with query "food preferences"
Result: Returns saved food-related memories
```

## Token Efficiency

This approach (Option C - Enhanced OpenAPI) is **token-efficient** because:
- ✅ LLM only calls tools when contextually appropriate
- ✅ No forced overhead on every message
- ✅ Smart decisions based on user input
- ❌ Risk: Might miss some memories if LLM doesn't recognize patterns

## Next Steps (Optional)

If the LLM **still** doesn't invoke tools frequently enough:

### Option B: Add a Pipe Function
Create an Open WebUI Pipe that:
- Automatically calls `/add` after every user message
- Auto-injects memories before LLM responses
- Guaranteed to never miss context
- Trade-off: Higher token usage

### Hybrid Approach
- Keep enhanced OpenAPI for `/add` (LLM decides)
- Add Pipe ONLY for `/relevant` (always inject context)
- Best of both worlds

## Monitoring

Watch your logs to see:
```bash
# Check if memories are being saved
grep "Stored.*new memories" logs.txt

# Check if extraction is working
grep "Extracted.*memories" logs.txt

# Check if LLM is being called
grep "LLM response" logs.txt
```

## Troubleshooting

**Issue:** LLM still not calling `/add` on "I love X" statements
- **Solution 1:** Check Open WebUI tool is enabled
- **Solution 2:** Try with different LLM (some are better at tool calling)
- **Solution 3:** Consider adding a Pipe function (Option B)

**Issue:** Too many tool calls, high token usage
- **Solution:** Adjust descriptions to be more selective
- **Solution:** Add negative examples of when NOT to call

**Issue:** Memories not being found
- **Solution:** Check `RELEVANCE_THRESHOLD` in .env (currently 0.6)
- **Solution:** Try lowering to 0.5 for more results

## Configuration Files Modified
- ✅ `app.py` - Enhanced endpoint and model descriptions
- ✅ Created `OPENAPI_IMPROVEMENTS.md` (this file)
- ✅ Created `test_openapi.py` - OpenAPI spec verification tool

## Related Settings (.env)
```bash
MIN_CONFIDENCE=0.5          # Minimum confidence to save memory
RELEVANCE_THRESHOLD=0.6     # Minimum score for /relevant results
DEDUP_THRESHOLD=0.7         # Similarity threshold for deduplication
```

---

**Author:** Enhanced via Claude Code
**Date:** 2025-10-16
**Approach:** Option C - Enhanced OpenAPI Descriptions
