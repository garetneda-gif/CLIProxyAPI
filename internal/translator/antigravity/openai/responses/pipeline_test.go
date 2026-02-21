package responses

import (
	"fmt"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	geminiResponses "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/openai/responses"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// simulateBuildRequest mirrors the executor's buildRequest function
func simulateBuildRequest(modelName string, payload []byte) []byte {
	// Step 1: geminiToAntigravity
	template, _ := sjson.Set(string(payload), "model", modelName)
	template, _ = sjson.Set(template, "userAgent", "antigravity")
	template, _ = sjson.Set(template, "requestType", "agent")
	template, _ = sjson.Set(template, "project", "warmup-12345")
	template, _ = sjson.Set(template, "requestId", "agent-test-id")
	template, _ = sjson.Set(template, "request.sessionId", "-1234567890")
	template, _ = sjson.Delete(template, "request.safetySettings")
	if toolConfig := gjson.Get(template, "toolConfig"); toolConfig.Exists() && !gjson.Get(template, "request.toolConfig").Exists() {
		template, _ = sjson.SetRaw(template, "request.toolConfig", toolConfig.Raw)
		template, _ = sjson.Delete(template, "toolConfig")
	}
	payload = []byte(template)

	// Step 2: Set model again (like line 1283)
	payload, _ = sjson.SetBytes(payload, "model", modelName)

	// Step 3: Walk/RenameKey for parametersJsonSchema and clean JSON schemas
	// ONLY within the tools section (matching the production fix)
	payloadStr := string(payload)
	useAntigravitySchema := strings.Contains(modelName, "claude") || strings.Contains(modelName, "gemini-3-pro-high")
	if toolsResult := gjson.Get(payloadStr, "request.tools"); toolsResult.Exists() {
		toolsStr := toolsResult.Raw

		paths := make([]string, 0)
		util.Walk(gjson.Parse(toolsStr), "", "parametersJsonSchema", &paths)
		for _, p := range paths {
			toolsStr, _ = util.RenameKey(toolsStr, p, p[:len(p)-len("parametersJsonSchema")]+"parameters")
		}

		if useAntigravitySchema {
			toolsStr = util.CleanJSONSchemaForAntigravity(toolsStr)
		} else {
			toolsStr = util.CleanJSONSchemaForGemini(toolsStr)
		}

		payloadStr, _ = sjson.SetRaw(payloadStr, "request.tools", toolsStr)
	}

	// Step 5: Non-Claude: delete maxOutputTokens
	if !strings.Contains(modelName, "claude") {
		payloadStr, _ = sjson.Delete(payloadStr, "request.generationConfig.maxOutputTokens")
	}

	return []byte(payloadStr)
}

// badFields are fields that should NEVER appear inside request.contents items
var badFields = []string{
	"safetySettings", "model", "userAgent", "requestType", "requestId",
	"sessionId", "systemInstruction", "toolConfig", "system_instruction",
	"generationConfig", "project", "request",
}

func checkContentsForBadFields(t *testing.T, payloadStr string) {
	t.Helper()
	contents := gjson.Get(payloadStr, "request.contents")
	if !contents.Exists() || !contents.IsArray() {
		t.Fatal("request.contents not found or not an array")
	}
	contents.ForEach(func(idx, value gjson.Result) bool {
		for _, field := range badFields {
			if value.Get(field).Exists() {
				t.Errorf("BUG: request.contents[%d] has field '%s', raw prefix: %.200s",
					idx.Int(), field, value.Raw)
			}
		}
		if !value.Get("role").Exists() {
			t.Errorf("WARNING: request.contents[%d] missing 'role', raw prefix: %.200s",
				idx.Int(), value.Raw)
		}
		return true
	})
}

func TestFullPipelineSimpleConversation(t *testing.T) {
	// Build a simple OpenAI Responses API request
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"input":[]}`
	input, _ = sjson.Set(input, "input.-1", map[string]any{
		"role":    "developer",
		"content": "You are a helpful assistant.",
	})
	for i := 0; i < 5; i++ {
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"role":"user","content":[{"type":"input_text","text":"Message %d from user"}]}`, i))
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"role":"assistant","content":[{"type":"output_text","text":"Response %d from model"}]}`, i))
	}

	// Step 1: Full chain conversion
	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	resultStr := string(result)

	// Step 2: Simulate executor's geminiToAntigravity
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr = string(result)

	// Check structure
	if !gjson.Get(resultStr, "request.contents").Exists() {
		t.Fatal("request.contents missing from final payload")
	}

	contentsCount := gjson.Get(resultStr, "request.contents.#").Int()
	t.Logf("Contents count: %d", contentsCount)

	checkContentsForBadFields(t, resultStr)
}

func TestFullPipelineWithToolCalls(t *testing.T) {
	// Build OpenAI Responses API request with function calls
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"input":[],"tools":[]}`

	// Add a tool
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}`)

	// Developer message
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"developer","content":"You are a coding assistant."}`)

	// User message
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Read the file main.go"}]}`)

	// Model function call
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call","name":"read_file","call_id":"call_123","arguments":"{\"path\":\"main.go\"}"}`)

	// Function output
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call_output","call_id":"call_123","output":"{\"content\":\"package main\\nfunc main() {}\"}"}`)

	// Model response
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"The file contains a basic Go program."}]}`)

	// Another round of function calls
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Now read README.md"}]}`)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call","name":"read_file","call_id":"call_456","arguments":"{\"path\":\"README.md\"}"}`)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call_output","call_id":"call_456","output":"{\"content\":\"# My Project\"}"}`)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"The README says My Project."}]}`)

	// Full pipeline
	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr := string(result)

	contentsCount := gjson.Get(resultStr, "request.contents.#").Int()
	t.Logf("Contents count: %d", contentsCount)

	checkContentsForBadFields(t, resultStr)
}

func TestFullPipelineLargeConversation(t *testing.T) {
	// Build a large conversation with many tool calls (simulating opencode session)
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"instructions":"You are a coding assistant.","input":[],"tools":[]}`

	// Add tools
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}`)
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}`)
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}`)

	// System message in input (like opencode does)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"system","content":"Additional system context."}`)

	// Create a long conversation with many tool calls
	callID := 0
	for round := 0; round < 50; round++ {
		// User message
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"role":"user","content":[{"type":"input_text","text":"Please do task %d"}]}`, round))

		// Model makes 2-3 tool calls per round
		for tool := 0; tool < 3; tool++ {
			callID++
			input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
				`{"type":"function_call","name":"read_file","call_id":"call_%d","arguments":"{\"path\":\"file%d.go\"}"}`,
				callID, callID))
		}

		// Function outputs
		for tool := 0; tool < 3; tool++ {
			id := callID - 2 + tool
			input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
				`{"type":"function_call_output","call_id":"call_%d","output":"{\"content\":\"content of file %d\"}"}`,
				id, id))
		}

		// Model response
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"role":"assistant","content":[{"type":"output_text","text":"Completed task %d."}]}`, round))
	}

	t.Logf("Input JSON size: %d bytes", len(input))
	t.Logf("Input items count: %d", gjson.Get(input, "input.#").Int())

	// Full pipeline
	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr := string(result)

	contentsCount := gjson.Get(resultStr, "request.contents.#").Int()
	t.Logf("Contents count: %d", contentsCount)

	checkContentsForBadFields(t, resultStr)

	// Verify top-level structure
	topKeys := []string{"project", "model", "userAgent", "requestType", "requestId", "request"}
	for _, key := range topKeys {
		if !gjson.Get(resultStr, key).Exists() {
			t.Errorf("Missing top-level key: %s", key)
		}
	}

	// Verify request-level structure
	reqKeys := []string{"contents", "systemInstruction", "sessionId"}
	for _, key := range reqKeys {
		if !gjson.Get(resultStr, "request."+key).Exists() {
			t.Errorf("Missing request key: %s", key)
		}
	}

	// Verify no system_instruction (should have been renamed)
	if gjson.Get(resultStr, "request.system_instruction").Exists() {
		t.Error("request.system_instruction should have been renamed to request.systemInstruction")
	}

	// Verify safetySettings was deleted by geminiToAntigravity
	if gjson.Get(resultStr, "request.safetySettings").Exists() {
		t.Error("request.safetySettings should have been deleted by geminiToAntigravity")
	}
}

func TestFullPipelineWithParallelToolCalls(t *testing.T) {
	// Test with parallel function calls (multiple calls before outputs)
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"input":[],"tools":[]}`

	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"developer","content":"Coding assistant."}`)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Read all files"}]}`)

	// 5 parallel function calls
	for i := 0; i < 5; i++ {
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"type":"function_call","name":"read_file","call_id":"p_call_%d","arguments":"{\"path\":\"file%d.go\"}"}`, i, i))
	}
	// 5 function outputs
	for i := 0; i < 5; i++ {
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"type":"function_call_output","call_id":"p_call_%d","output":"content %d"}`, i, i))
	}

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"Done reading all files."}]}`)

	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr := string(result)

	contentsCount := gjson.Get(resultStr, "request.contents.#").Int()
	t.Logf("Contents count: %d", contentsCount)

	checkContentsForBadFields(t, resultStr)
}

func TestFullPipelineWithSpecialCharactersInContent(t *testing.T) {
	// Test with content that contains JSON-like text and special characters
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"input":[]}`

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"developer","content":"You are helpful."}`)

	// User sends a message containing JSON-like text with field names that match bad fields
	userMsg := `The safetySettings field should be set like {"safetySettings": [{"category": "HARM"}]}. Also check the model field and systemInstruction.`
	input, _ = sjson.Set(input, "input.-1", map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "input_text", "text": userMsg},
		},
	})

	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr := string(result)

	checkContentsForBadFields(t, resultStr)
}

func TestFullPipelineVeryLargeConversation(t *testing.T) {
	// Test with 500+ content items to match the real-world error at contents[405]
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"instructions":"You are a coding assistant. Help users with code.","input":[],"tools":[]}`

	// Add tools with complex schemas
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"encoding":{"type":"string","enum":["utf-8","ascii","binary"]}},"required":["path"],"additionalProperties":false}}`)
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"write_file","description":"Write content to a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}`)
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"bash","description":"Execute a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}`)
	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"list_files","description":"List files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"]}}`)

	// System message
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"system","content":"Additional instructions for coding."}`)

	// Create a VERY long conversation (targeting 400+ content entries)
	callID := 0
	for round := 0; round < 60; round++ {
		// User message with longer text
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"role":"user","content":[{"type":"input_text","text":"Please help me with task number %d. I need you to read several files and make changes to them. The files are in the src/ directory and need to be updated according to the new API specification."}]}`, round))

		// Model makes multiple tool calls
		numCalls := 2 + (round % 3) // 2-4 calls per round
		for tool := 0; tool < numCalls; tool++ {
			callID++
			toolNames := []string{"read_file", "write_file", "bash", "list_files"}
			toolName := toolNames[tool%len(toolNames)]
			input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
				`{"type":"function_call","name":"%s","call_id":"call_%d","arguments":"{\"path\":\"src/file%d.go\"}"}`,
				toolName, callID, callID))
		}

		// Function outputs with realistic content
		for tool := 0; tool < numCalls; tool++ {
			id := callID - numCalls + 1 + tool
			// Simulate realistic function output with JSON content
			// Use sjson.Set to properly escape the output string (not manual strings.ReplaceAll)
			output := fmt.Sprintf(`{"content":"package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {\n\tfmt.Println(\"File %d\")\n\tos.Exit(0)\n}","metadata":{"size":150,"modified":"2026-01-15"}}`, id)
			item := fmt.Sprintf(`{"type":"function_call_output","call_id":"call_%d"}`, id)
			item, _ = sjson.Set(item, "output", output)
			input, _ = sjson.SetRaw(input, "input.-1", item)
		}

		// Model response
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"role":"assistant","content":[{"type":"output_text","text":"I've completed task %d. The files have been updated with the new API endpoints and error handling. Let me know if you need any other changes."}]}`, round))
	}

	inputItemCount := gjson.Get(input, "input.#").Int()
	t.Logf("Input JSON size: %d bytes (%.1f KB)", len(input), float64(len(input))/1024)
	t.Logf("Input items count: %d", inputItemCount)

	// Step 1: Convert to flat Gemini format (before wrapping)
	geminiResult := geminiResponses.ConvertOpenAIResponsesRequestToGemini("gemini-3-flash", []byte(input), true)
	geminiStr := string(geminiResult)
	geminiContentsCount := gjson.Get(geminiStr, "contents.#").Int()
	t.Logf("Gemini flat format: %d bytes, contents count: %d", len(geminiStr), geminiContentsCount)
	// Show first and last few items
	gjson.Get(geminiStr, "contents").ForEach(func(idx, val gjson.Result) bool {
		if idx.Int() < 3 || idx.Int() > geminiContentsCount-3 {
			role := val.Get("role").String()
			partsCount := val.Get("parts.#").Int()
			t.Logf("  gemini contents[%d]: role=%s parts=%d", idx.Int(), role, partsCount)
		} else if idx.Int() == 3 {
			t.Logf("  ... (skipping middle items) ...")
		}
		return true
	})

	// Step 2: Full chain (Gemini → Antigravity wrapper with fixCLIToolResponse)
	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	intermediateSize := len(result)
	t.Logf("After full translator: %d bytes (%.1f KB)", intermediateSize, float64(intermediateSize)/1024)

	// Check intermediate result
	intermediateStr := string(result)
	intermediateContents := gjson.Get(intermediateStr, "request.contents.#").Int()
	t.Logf("Antigravity wrapped contents count: %d", intermediateContents)

	// Simulate buildRequest
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr := string(result)

	finalSize := len(resultStr)
	t.Logf("After buildRequest: %d bytes (%.1f KB)", finalSize, float64(finalSize)/1024)

	contentsCount := gjson.Get(resultStr, "request.contents.#").Int()
	t.Logf("Final contents count: %d", contentsCount)

	// Check for bad fields in each content item
	checkContentsForBadFields(t, resultStr)

	// Verify specific content items near the end of the array
	lastIdx := contentsCount - 1
	for i := lastIdx - 5; i <= lastIdx; i++ {
		if i < 0 {
			continue
		}
		item := gjson.Get(resultStr, fmt.Sprintf("request.contents.%d", i))
		role := item.Get("role").String()
		hasParts := item.Get("parts").Exists()
		t.Logf("  contents[%d]: role=%s hasParts=%v keys=%v",
			i, role, hasParts, getTopKeys(item.Raw))
	}
}

func TestDebugFixCLIToolResponse(t *testing.T) {
	// Minimal reproduction: 2 function calls followed by 2 function outputs
	// but from the OpenAI Responses format (already normalized by converter)
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"instructions":"Helper.","input":[],"tools":[]}`

	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}`)

	// User msg
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Read files"}]}`)

	// 3 function calls
	for i := 0; i < 3; i++ {
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"type":"function_call","name":"read_file","call_id":"c%d","arguments":"{\"path\":\"f%d.go\"}"}`, i, i))
	}
	// 3 function outputs
	for i := 0; i < 3; i++ {
		input, _ = sjson.SetRaw(input, "input.-1", fmt.Sprintf(
			`{"type":"function_call_output","call_id":"c%d","output":"content %d"}`, i, i))
	}

	// Assistant response
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"Done."}]}`)

	t.Logf("Input items: %d", gjson.Get(input, "input.#").Int())

	// Step 1: Only convert to Gemini format (before wrapping)
	// ConvertOpenAIResponsesRequestToGemini is in a different package, use the full chain and inspect
	geminiResult := geminiResponses.ConvertOpenAIResponsesRequestToGemini("gemini-3-flash", []byte(input), true)
	geminiStr := string(geminiResult)

	geminiContents := gjson.Get(geminiStr, "contents")
	t.Logf("Gemini format contents count: %d", geminiContents.Get("#").Int())
	geminiContents.ForEach(func(idx, val gjson.Result) bool {
		role := val.Get("role").String()
		keys := getTopKeys(val.Raw)
		partsCount := val.Get("parts.#").Int()
		t.Logf("  gemini contents[%d]: role=%s parts=%d keys=%v", idx.Int(), role, partsCount, keys)
		return true
	})

	// Step 2: Full chain
	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	resultStr := string(result)

	antigravityContents := gjson.Get(resultStr, "request.contents")
	t.Logf("Antigravity format contents count: %d", antigravityContents.Get("#").Int())
	antigravityContents.ForEach(func(idx, val gjson.Result) bool {
		role := val.Get("role").String()
		keys := getTopKeys(val.Raw)
		partsCount := val.Get("parts.#").Int()
		t.Logf("  ag contents[%d]: role=%s parts=%d keys=%v", idx.Int(), role, partsCount, keys)
		return true
	})

	checkContentsForBadFields(t, resultStr)
}

func TestMinimalReproWithEscapedOutput(t *testing.T) {
	// Minimal test with function output containing escaped JSON (like real opencode output)
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"instructions":"Helper.","input":[],"tools":[]}`

	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Read files"}]}`)

	// Function call
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call","name":"read_file","call_id":"c1","arguments":"{\"path\":\"main.go\"}"}`)

	// Function output WITH escaped JSON containing quotes
	// This mimics what opencode sends when reading a Go file
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call_output","call_id":"c1","output":"{\"content\":\"package main\\n\\nimport (\\n\\t\\\"fmt\\\"\\n)\\n\\nfunc main() {\\n\\tfmt.Println(\\\"hello\\\")\\n}\"}"}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"I see the code."}]}`)

	t.Logf("Input items: %d", gjson.Get(input, "input.#").Int())

	// Test flat Gemini conversion
	geminiResult := geminiResponses.ConvertOpenAIResponsesRequestToGemini("gemini-3-flash", []byte(input), true)
	geminiStr := string(geminiResult)
	t.Logf("Gemini result size: %d bytes", len(geminiStr))
	t.Logf("Gemini valid JSON: %v", gjson.Valid(geminiStr))

	geminiContentsCount := gjson.Get(geminiStr, "contents.#").Int()
	t.Logf("Gemini contents count: %d", geminiContentsCount)

	gjson.Get(geminiStr, "contents").ForEach(func(idx, val gjson.Result) bool {
		role := val.Get("role").String()
		keys := getTopKeys(val.Raw)
		t.Logf("  contents[%d]: role=%s keys=%v", idx.Int(), role, keys)
		return true
	})

	if geminiContentsCount < 4 {
		t.Errorf("Expected at least 4 contents, got %d", geminiContentsCount)
		// Show the full JSON for debugging
		if len(geminiStr) < 5000 {
			t.Logf("Full Gemini JSON:\n%s", geminiStr)
		} else {
			t.Logf("Gemini JSON prefix:\n%s...", geminiStr[:3000])
		}
	}

	// Test full chain
	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	resultStr := string(result)
	checkContentsForBadFields(t, resultStr)
}

func TestMinimalReproWithSimpleOutput(t *testing.T) {
	// Same as above but with SIMPLE output (no escaped JSON)
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"instructions":"Helper.","input":[],"tools":[]}`

	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Read files"}]}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call","name":"read_file","call_id":"c1","arguments":"{\"path\":\"main.go\"}"}`)

	// Simple string output - no complex JSON
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call_output","call_id":"c1","output":"simple text content"}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"I see the code."}]}`)

	geminiResult := geminiResponses.ConvertOpenAIResponsesRequestToGemini("gemini-3-flash", []byte(input), true)
	geminiStr := string(geminiResult)
	geminiContentsCount := gjson.Get(geminiStr, "contents.#").Int()
	t.Logf("Simple output: contents count: %d, valid JSON: %v", geminiContentsCount, gjson.Valid(geminiStr))

	if geminiContentsCount < 4 {
		t.Errorf("Expected at least 4 contents, got %d", geminiContentsCount)
		t.Logf("Full JSON:\n%s", geminiStr)
	}
}

func getTopKeys(jsonStr string) []string {
	var keys []string
	gjson.Parse(jsonStr).ForEach(func(key, _ gjson.Result) bool {
		keys = append(keys, key.String())
		return true
	})
	return keys
}

func TestFullPipelineFunctionOutputWithJSONContent(t *testing.T) {
	// Test where function output contains JSON with fields like safetySettings
	input := `{"model":"gemini-3-flash","max_output_tokens":32000,"stream":true,"input":[],"tools":[]}`

	input, _ = sjson.SetRaw(input, "tools.-1",
		`{"type":"function","name":"read_file","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}`)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"developer","content":"Assistant."}`)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"user","content":[{"type":"input_text","text":"Read config"}]}`)
	input, _ = sjson.SetRaw(input, "input.-1",
		`{"type":"function_call","name":"read_file","call_id":"c1","arguments":"{\"path\":\"config.json\"}"}`)

	// Function output contains JSON with safetySettings, model, etc.
	output := `{"safetySettings":[{"category":"test"}],"model":"test","systemInstruction":{"parts":[]},"generationConfig":{"maxOutputTokens":100}}`
	outputItem := `{"type":"function_call_output","call_id":"c1"}`
	outputItem, _ = sjson.Set(outputItem, "output", output)
	input, _ = sjson.SetRaw(input, "input.-1", outputItem)

	input, _ = sjson.SetRaw(input, "input.-1",
		`{"role":"assistant","content":[{"type":"output_text","text":"Config loaded."}]}`)

	result := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", []byte(input), true)
	result = simulateBuildRequest("gemini-3-flash", result)
	resultStr := string(result)

	checkContentsForBadFields(t, resultStr)
}
