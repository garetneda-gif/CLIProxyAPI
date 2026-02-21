package responses

import (
	"encoding/json"
	"fmt"
	"testing"

	geminiResponses "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/openai/responses"
	"github.com/tidwall/gjson"
)

// TestSchemaCleaningCorruptsContents tests if cleanJSONSchema modifies function response data
func TestSchemaCleaningCorruptsContents(t *testing.T) {
	// Function output that contains JSON schema-like fields
	schemaLikeOutput := `{
		"result": {
			"content": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":[\"integer\",\"null\"]}},\"required\":[\"name\"],\"additionalProperties\":false,\"$ref\":\"#/definitions/User\",\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"format\":\"email\",\"pattern\":\"^[a-z]+$\",\"default\":\"test\",\"examples\":[\"hello\"]}",
			"path": "/tmp/schema.json"
		},
		"id": "call_1"
	}`

	inputItems := []any{
		map[string]any{
			"role":    "system",
			"content": "You are helpful.",
		},
		map[string]any{
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Read the schema file"}},
		},
		map[string]any{
			"type":      "function_call",
			"name":      "read_file",
			"call_id":   "call_1",
			"arguments": `{"path":"/tmp/schema.json"}`,
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_1",
			"output":  schemaLikeOutput,
		},
		map[string]any{
			"role":    "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": "Here is the schema."}},
		},
	}

	tools := []any{
		map[string]any{
			"type": "function", "name": "read_file", "description": "Read a file",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":   []string{"path"},
			},
		},
	}

	input := map[string]any{
		"model":        "gemini-3-flash",
		"stream":       true,
		"instructions": "You are a coding assistant.",
		"input":        inputItems,
		"tools":        tools,
	}

	inputJSON, _ := json.Marshal(input)
	t.Logf("Input valid=%v size=%d", json.Valid(inputJSON), len(inputJSON))

	// Step 1: Gemini conversion
	geminiResult := geminiResponses.ConvertOpenAIResponsesRequestToGemini("gemini-3-flash", inputJSON, true)
	geminiValid := json.Valid(geminiResult)
	t.Logf("Gemini: valid=%v contents=%d", geminiValid, gjson.GetBytes(geminiResult, "contents.#").Int())

	// Step 2: Full pipeline
	fullResult := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", inputJSON, true)
	fullValid := json.Valid(fullResult)
	t.Logf("Antigravity: valid=%v contents=%d", fullValid, gjson.GetBytes(fullResult, "request.contents.#").Int())

	// Step 3: With buildRequest (includes cleanJSONSchema)
	final := simulateBuildRequest("gemini-3-flash", fullResult)
	finalValid := json.Valid(final)
	finalCount := gjson.GetBytes(final, "request.contents.#").Int()
	t.Logf("Final: valid=%v contents=%d size=%d", finalValid, finalCount, len(final))

	if !finalValid {
		t.Error("Final JSON is invalid!")
	}

	// Check function response result is preserved
	funcResult := gjson.GetBytes(final, "request.contents.2.parts.0.functionResponse.response.result")
	t.Logf("Function response result type: %v", funcResult.Type)
	if funcResult.Exists() {
		resultContent := funcResult.Get("content").String()
		t.Logf("Result content (first 100): %s", resultContent[:min100(len(resultContent))])
		// Check if schema fields in the result were incorrectly modified
		if !funcResult.Get("result.additionalProperties").Exists() && gjson.Valid(funcResult.Get("result.content").String()) {
			t.Logf("WARNING: additionalProperties may have been removed from function response result")
		}
	}

	// Check for bad fields
	checkContentsForBadFields(t, string(final))
}

func min100(n int) int {
	if n < 100 {
		return n
	}
	return 100
}

// TestSchemaCleaningWithRawJSONOutput tests when function output contains a raw JSON object
// (not a string) with schema keywords
func TestSchemaCleaningWithRawJSONOutput(t *testing.T) {
	// This simulates a function that returns a JSON object directly
	// (the output is a JSON string that parses to an object containing schema fields)
	outputObj := `{"type":"object","properties":{"name":{"type":"string","format":"email"},"tags":{"type":"array","items":{"type":"string"}}},"required":["name"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#","definitions":{"Tag":{"type":"object","properties":{"label":{"type":"string"}}}}}`

	inputItems := []any{
		map[string]any{
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Get the API schema"}},
		},
		map[string]any{
			"type":      "function_call",
			"name":      "get_schema",
			"call_id":   "call_1",
			"arguments": `{"api":"users"}`,
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_1",
			"output":  outputObj,
		},
		map[string]any{
			"role":    "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": "Here is the schema."}},
		},
	}

	tools := []any{
		map[string]any{
			"type": "function", "name": "get_schema", "description": "Get API schema",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"api": map[string]any{"type": "string"}},
				"required":   []string{"api"},
			},
		},
	}

	input := map[string]any{
		"model":        "gemini-3-flash",
		"stream":       true,
		"instructions": "Help with APIs.",
		"input":        inputItems,
		"tools":        tools,
	}

	inputJSON, _ := json.Marshal(input)

	// Before cleanJSONSchema
	fullResult := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", inputJSON, true)
	beforeCount := gjson.GetBytes(fullResult, "request.contents.#").Int()
	beforeFuncResult := gjson.GetBytes(fullResult, "request.contents.2.parts.0.functionResponse.response.result")
	t.Logf("Before cleanJSONSchema: contents=%d", beforeCount)
	t.Logf("  funcResponse.result has 'type': %v", beforeFuncResult.Get("type").Exists())
	t.Logf("  funcResponse.result has 'additionalProperties': %v", beforeFuncResult.Get("additionalProperties").Exists())
	t.Logf("  funcResponse.result has '$schema': %v", beforeFuncResult.Get("$schema").Exists())
	t.Logf("  funcResponse.result has 'definitions': %v", beforeFuncResult.Get("definitions").Exists())
	t.Logf("  funcResponse.result has 'format': %v", beforeFuncResult.Get("format").Exists())

	// After cleanJSONSchema
	final := simulateBuildRequest("gemini-3-flash", fullResult)
	finalCount := gjson.GetBytes(final, "request.contents.#").Int()
	finalFuncResult := gjson.GetBytes(final, "request.contents.2.parts.0.functionResponse.response.result")
	t.Logf("After cleanJSONSchema: contents=%d", finalCount)
	t.Logf("  funcResponse.result has 'type': %v", finalFuncResult.Get("type").Exists())
	t.Logf("  funcResponse.result has 'additionalProperties': %v", finalFuncResult.Get("additionalProperties").Exists())
	t.Logf("  funcResponse.result has '$schema': %v", finalFuncResult.Get("$schema").Exists())
	t.Logf("  funcResponse.result has 'definitions': %v", finalFuncResult.Get("definitions").Exists())
	t.Logf("  funcResponse.result has 'format': %v", finalFuncResult.Get("format").Exists())

	if !json.Valid(final) {
		t.Error("Final JSON is invalid!")
	}

	// Check if fields were removed from function response
	removed := []string{}
	for _, f := range []string{"additionalProperties", "$schema", "definitions", "format"} {
		if beforeFuncResult.Get(f).Exists() && !finalFuncResult.Get(f).Exists() {
			removed = append(removed, f)
		}
	}
	if len(removed) > 0 {
		t.Errorf("cleanJSONSchema incorrectly removed fields from function response: %v", removed)
	}

	checkContentsForBadFields(t, string(final))
}

// TestLargeConversationWithSchemaOutputs combines large conversation with schema-containing outputs
func TestLargeConversationWithSchemaOutputs(t *testing.T) {
	inputItems := []any{
		map[string]any{
			"role":    "system",
			"content": "Coding assistant.",
		},
	}

	callID := 0
	for round := 0; round < 60; round++ {
		inputItems = append(inputItems, map[string]any{
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": fmt.Sprintf("Task %d", round)}},
		})

		numCalls := 2 + (round % 3)
		for tool := 0; tool < numCalls; tool++ {
			callID++
			inputItems = append(inputItems, map[string]any{
				"type":      "function_call",
				"name":      "read_file",
				"call_id":   fmt.Sprintf("call_%d", callID),
				"arguments": fmt.Sprintf(`{"path":"file%d.go"}`, callID),
			})
		}

		for tool := 0; tool < numCalls; tool++ {
			id := callID - numCalls + 1 + tool
			// Some outputs contain schema-like JSON, some don't
			var outputJSON string
			if id%5 == 0 {
				// Schema-like output
				outputJSON = fmt.Sprintf(`{"type":"object","properties":{"field_%d":{"type":"string","format":"uri","additionalProperties":false}},"$ref":"#/def/Item","definitions":{"Item":{"type":"object"}}}`, id)
			} else {
				// Regular output
				outputJSON = fmt.Sprintf(`{"content":"line %d of code","path":"file%d.go"}`, id, id)
			}
			inputItems = append(inputItems, map[string]any{
				"type":    "function_call_output",
				"call_id": fmt.Sprintf("call_%d", id),
				"output":  outputJSON,
			})
		}

		inputItems = append(inputItems, map[string]any{
			"role":    "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": fmt.Sprintf("Done %d.", round)}},
		})
	}

	tools := []any{
		map[string]any{
			"type": "function", "name": "read_file", "description": "Read a file",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":   []string{"path"},
			},
		},
	}

	input := map[string]any{
		"model":        "gemini-3-flash",
		"stream":       true,
		"instructions": "You are helpful.",
		"input":        inputItems,
		"tools":        tools,
	}

	inputJSON, _ := json.Marshal(input)
	t.Logf("Input: %d bytes, %d items, valid=%v",
		len(inputJSON), gjson.GetBytes(inputJSON, "input.#").Int(), json.Valid(inputJSON))

	fullResult := ConvertOpenAIResponsesRequestToAntigravity("gemini-3-flash", inputJSON, true)
	final := simulateBuildRequest("gemini-3-flash", fullResult)

	finalValid := json.Valid(final)
	finalCount := gjson.GetBytes(final, "request.contents.#").Int()
	t.Logf("Final: valid=%v contents=%d size=%d", finalValid, finalCount, len(final))

	if !finalValid {
		t.Fatalf("Final JSON is invalid!")
	}
	if finalCount < 100 {
		t.Errorf("Expected 100+ contents, got %d", finalCount)
	}

	checkContentsForBadFields(t, string(final))
}
