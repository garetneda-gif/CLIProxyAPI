package responses

import (
	"encoding/json"
	"fmt"
	"strconv"
	"testing"

	geminiResponses "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/openai/responses"
	"github.com/tidwall/gjson"
)

// TestFullPipelineProperJSON uses json.Marshal for input (like production) with 60 rounds
func TestFullPipelineProperJSON(t *testing.T) {
	readContent := "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {\n\tfmt.Println(\"File\")\n\tos.Exit(0)\n}"

	callID := 0
	inputItems := []any{
		map[string]any{
			"role":    "system",
			"content": "Additional instructions for coding.",
		},
	}

	for round := 0; round < 60; round++ {
		inputItems = append(inputItems, map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "input_text", "text": fmt.Sprintf("Please help with task %d. Read several files and make changes.", round)},
			},
		})

		numCalls := 2 + (round % 3) // 2-4 calls per round
		for tool := 0; tool < numCalls; tool++ {
			callID++
			toolNames := []string{"read_file", "write_file", "bash", "list_files"}
			toolName := toolNames[tool%len(toolNames)]
			inputItems = append(inputItems, map[string]any{
				"type":      "function_call",
				"name":      toolName,
				"call_id":   fmt.Sprintf("call_%d", callID),
				"arguments": fmt.Sprintf(`{"path":"src/file%d.go"}`, callID),
			})
		}

		for tool := 0; tool < numCalls; tool++ {
			id := callID - numCalls + 1 + tool
			outputJSON := fmt.Sprintf(`{"content":%s,"metadata":{"size":150,"modified":"2026-01-15"}}`,
				strconv.Quote(readContent))
			inputItems = append(inputItems, map[string]any{
				"type":    "function_call_output",
				"call_id": fmt.Sprintf("call_%d", id),
				"output":  outputJSON,
			})
		}

		inputItems = append(inputItems, map[string]any{
			"role": "assistant",
			"content": []any{
				map[string]any{"type": "output_text", "text": fmt.Sprintf("I've completed task %d. The files have been updated.", round)},
			},
		})
	}

	tools := []any{
		map[string]any{
			"type": "function", "name": "read_file", "description": "Read a file from disk",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{
				"path": map[string]any{"type": "string", "description": "File path"},
			}, "required": []string{"path"}},
		},
		map[string]any{
			"type": "function", "name": "write_file", "description": "Write content to a file",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{
				"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"},
			}, "required": []string{"path", "content"}},
		},
		map[string]any{
			"type": "function", "name": "bash", "description": "Execute a bash command",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{
				"command": map[string]any{"type": "string"},
			}, "required": []string{"command"}},
		},
		map[string]any{
			"type": "function", "name": "list_files", "description": "List files in a directory",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{
				"path": map[string]any{"type": "string"}, "recursive": map[string]any{"type": "boolean"},
			}, "required": []string{"path"}},
		},
	}

	input := map[string]any{
		"model":             "gemini-3-flash",
		"max_output_tokens": 32000,
		"stream":            true,
		"instructions":      "You are a coding assistant. Help users with code.",
		"input":             inputItems,
		"tools":             tools,
	}

	inputJSON, _ := json.Marshal(input)
	inputCount := gjson.GetBytes(inputJSON, "input.#").Int()
	t.Logf("Input: %d bytes (%.1f KB), %d items, valid=%v",
		len(inputJSON), float64(len(inputJSON))/1024, inputCount, json.Valid(inputJSON))

	// Step 1: Gemini conversion
	geminiResult := geminiResponses.ConvertOpenAIResponsesRequestToGemini("gemini-3-flash", inputJSON, true)
	geminiValid := json.Valid(geminiResult)
	geminiCount := gjson.GetBytes(geminiResult, "contents.#").Int()
	t.Logf("Gemini: valid=%v contents=%d size=%d", geminiValid, geminiCount, len(geminiResult))

	if !geminiValid {
		t.Fatalf("Gemini output invalid JSON!")
	}

	// Step 2: Full pipeline (Gemini + Antigravity)
	fullResult := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", inputJSON, true)
	fullValid := json.Valid(fullResult)
	fullCount := gjson.GetBytes(fullResult, "request.contents.#").Int()
	t.Logf("Antigravity: valid=%v contents=%d size=%d", fullValid, fullCount, len(fullResult))

	if !fullValid {
		t.Fatalf("Antigravity output invalid JSON!")
	}

	// Step 3: buildRequest simulation
	final := simulateBuildRequest("gemini-3-flash", fullResult)
	finalValid := json.Valid(final)
	finalCount := gjson.GetBytes(final, "request.contents.#").Int()
	t.Logf("Final: valid=%v contents=%d size=%d", finalValid, finalCount, len(final))

	if !finalValid {
		t.Fatalf("Final output invalid JSON!")
	}

	// Check for bad fields in contents
	badFields := []string{"safetySettings", "model", "userAgent", "requestType", "requestId", "sessionId", "systemInstruction", "system_instruction", "toolConfig", "generationConfig"}
	hasBug := false
	gjson.GetBytes(final, "request.contents").ForEach(func(idx, value gjson.Result) bool {
		for _, f := range badFields {
			if value.Get(f).Exists() {
				t.Errorf("BUG: request.contents[%d] has field '%s'", idx.Int(), f)
				hasBug = true
			}
		}
		return true
	})

	if !hasBug {
		t.Logf("No bad fields found in any of %d content items", finalCount)
	}

	// Verify reasonable content count
	if finalCount < 100 {
		t.Errorf("Expected 100+ final contents, got %d", finalCount)
	}
}
