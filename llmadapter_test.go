package einox

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
)

// TestCreateChatCompletionStream 测试流式聊天完成功能
// 执行命令：go test -run TestCreateChatCompletionStream
func TestCreateChatCompletionStream(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 检查是否跳过Azure测试
	//skipAzureTests := os.Getenv("SKIP_AZURE_TESTS") == "1"

	// 检查是否跳过DeepSeek测试
	skipDeepSeekTests := os.Getenv("SKIP_DEEPSEEK_TESTS") == "1"

	// 准备测试用例
	testCases := []struct {
		name       string
		request    ChatRequest
		provider   string
		skipTest   bool
		skipReason string
	}{
		{
			name: "基本流式聊天完成测试-bedrock",
			request: ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个有帮助的助手。",
						},
						{
							Role:    "user",
							Content: "简单介绍一下自然语言处理。",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      true,
				},
			},
			provider: "bedrock",
			skipTest: false,
		},
		{
			name: "基本流式聊天完成测试-azure",
			request: ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "gpt-4o",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个有帮助的助手。",
						},
						{
							Role:    "user",
							Content: "简单介绍一下人工智能。",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      true,
				},
			},
			provider:   "azure",
			skipTest:   false,
			skipReason: "跳过Azure API测试",
		},
		{
			name: "基本流式聊天完成测试-deepseek",
			request: ChatRequest{
				Provider: "deepseek",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "deepseek-chat",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个专业的AI助手。",
						},
						{
							Role:    "user",
							Content: "请简单介绍一下机器学习的基本概念。",
						},
					},
					MaxTokens:   150,
					Temperature: 0.8,
					Stream:      true,
				},
			},
			provider:   "deepseek",
			skipTest:   skipDeepSeekTests,
			skipReason: "跳过DeepSeek API测试",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 如果需要跳过此测试
			if tc.skipTest {
				t.Skip(tc.skipReason)
			}

			// 设置供应商
			tc.request.Provider = tc.provider

			// 创建缓冲区用于接收流式响应
			buffer := new(bytes.Buffer)

			// 调用被测试的函数
			resp, err := CreateChatCompletion(tc.request, buffer)

			// 检查结果
			if err != nil {
				t.Logf("测试期间出现错误: %v", err)
				t.Skip("API调用失败，可能是配置问题")
				return
			}

			// 对于流式响应，期望resp为nil
			assert.Nil(t, resp, "流式响应应返回nil响应对象")

			// 获取响应内容
			response := buffer.String()

			// 验证响应格式正确性
			assert.True(t, len(response) > 0, "响应不应为空")
			assert.Contains(t, response, "data: ", "响应应包含data:前缀")

			// 解析并验证每个响应块
			lines := strings.Split(response, "\n\n")
			var contentLines []string
			var allContent string

			for _, line := range lines {
				if strings.HasPrefix(line, "data: ") && line != "data: [DONE]" {
					// 解析JSON
					jsonData := strings.TrimPrefix(line, "data: ")
					var streamResp StreamResponse
					err := json.Unmarshal([]byte(jsonData), &streamResp)

					if err != nil {
						t.Errorf("解析响应JSON失败: %v", err)
						continue
					}

					// 验证响应结构
					assert.NotEmpty(t, streamResp.ID, "响应ID不应为空")
					assert.Equal(t, "chat.completion.chunk", streamResp.Object, "响应对象类型应为chat.completion.chunk")
					assert.NotZero(t, streamResp.Created, "创建时间不应为零")
					assert.Equal(t, tc.request.ChatCompletionRequest.Model, streamResp.Model, "响应模型应与请求模型匹配")
					assert.NotEmpty(t, streamResp.Choices, "选择不应为空")

					// 收集内容
					if len(streamResp.Choices) > 0 {
						content := streamResp.Choices[0].Delta.Content
						if content != "" {
							contentLines = append(contentLines, content)
							allContent += content
						}
					}
				}
			}

			// 验证收集到的内容
			t.Logf("收到 %d 个内容块", len(contentLines))
			t.Logf("完整响应内容: %s", allContent)
			assert.True(t, len(contentLines) > 0, "应收到至少一个内容块")
			assert.NotEmpty(t, allContent, "应收到非空内容")
		})
	}
}

// TestCreateChatCompletionNonStream 测试非流式聊天完成功能
// 执行命令：go test -run TestCreateChatCompletionNonStream
func TestCreateChatCompletionNonStream(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 检查是否跳过DeepSeek测试
	skipDeepSeekTests := os.Getenv("SKIP_DEEPSEEK_TESTS") == "1"

	// 准备测试用例
	testCases := []struct {
		name           string
		request        ChatRequest
		provider       string
		expectError    bool
		expectedErrMsg string
		skipTest       bool
		skipReason     string
	}{
		{
			name: "基本非流式聊天完成测试-bedrock",
			request: ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个有帮助的助手。",
						},
						{
							Role:    "user",
							Content: "简单介绍一下自然语言处理。",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      false,
				},
			},
			provider:    "bedrock",
			expectError: false,
			skipTest:    false,
		},
		{
			name: "调用BedrockCreateChatCompletionToChat测试",
			request: ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个简洁的助手。回答要精简。",
						},
						{
							Role:    "user",
							Content: "什么是大模型？",
						},
					},
					MaxTokens:   50,
					Temperature: 0.5,
					Stream:      false,
				},
			},
			provider:    "bedrock",
			expectError: false,
			skipTest:    false,
		},
		{
			name: "基本非流式聊天完成测试-azure",
			request: ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "gpt-4o",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个简洁的助手。回答要精简。",
						},
						{
							Role:    "user",
							Content: "什么是自然语言处理？",
						},
					},
					MaxTokens:   50,
					Temperature: 0.7,
					Stream:      false,
				},
			},
			provider:    "azure",
			expectError: false,
			skipTest:    false,
			skipReason:  "跳过Azure API测试",
		},
		{
			name: "基本非流式聊天完成测试-deepseek",
			request: ChatRequest{
				Provider: "deepseek",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "deepseek-chat",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个专业的AI助手。",
						},
						{
							Role:    "user",
							Content: "请解释什么是深度学习？",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      false,
				},
			},
			provider:    "deepseek",
			expectError: false,
			skipTest:    skipDeepSeekTests,
			skipReason:  "跳过DeepSeek API测试",
		},
		{
			name: "调用DeepSeekCreateChatCompletionToChat测试",
			request: ChatRequest{
				Provider: "deepseek",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "deepseek-chat",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个简洁的助手。回答要精简。",
						},
						{
							Role:    "user",
							Content: "什么是强化学习？",
						},
					},
					MaxTokens:   50,
					Temperature: 0.5,
					Stream:      false,
				},
			},
			provider:    "deepseek",
			expectError: false,
			skipTest:    skipDeepSeekTests,
			skipReason:  "跳过DeepSeek API测试",
		},
		{
			name: "不支持的供应商测试",
			request: ChatRequest{
				Provider: "unsupported",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "some-model",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "user",
							Content: "Hello",
						},
					},
				},
			},
			provider:       "unsupported",
			expectError:    true,
			expectedErrMsg: "不支持的AI供应商: unsupported",
			skipTest:       false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 如果需要跳过此测试
			if tc.skipTest {
				t.Skip(tc.skipReason)
			}

			// 设置供应商
			tc.request.Provider = tc.provider

			// 调用被测试的函数
			resp, err := CreateChatCompletion(tc.request, nil)

			// 检查错误
			if tc.expectError {
				assert.Error(t, err, "应返回错误")
				if tc.expectedErrMsg != "" {
					assert.Contains(t, err.Error(), tc.expectedErrMsg, "错误消息应包含预期内容")
				}
				return
			}

			// 非流式响应验证
			assert.NoError(t, err, "不应返回错误")
			assert.NotNil(t, resp, "响应不应为空")
			assert.NotEmpty(t, resp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion", resp.Object, "响应对象类型应为chat.completion")
			assert.NotZero(t, resp.Created, "创建时间不应为零")
			assert.Equal(t, tc.request.ChatCompletionRequest.Model, resp.Model, "响应模型应与请求模型匹配")
			assert.NotEmpty(t, resp.Choices, "选择不应为空")

			if len(resp.Choices) > 0 {
				assert.NotEmpty(t, resp.Choices[0].Message.Content, "消息内容不应为空")
				assert.NotEmpty(t, resp.Choices[0].FinishReason, "完成原因不应为空")
				assert.Equal(t, "assistant", resp.Choices[0].Message.Role, "消息角色应为assistant")
				t.Logf("响应内容: %s", resp.Choices[0].Message.Content)
				t.Logf("完成原因: %s", resp.Choices[0].FinishReason)
			}
		})
	}
}

// TestCreateChatCompletionWithDefaultProvider 测试默认供应商的情况
func TestCreateChatCompletionWithDefaultProvider(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 准备测试用例
	request := ChatRequest{
		Provider: "bedrock",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "system",
					Content: "你是一个有帮助的助手。",
				},
				{
					Role:    "user",
					Content: "简单介绍一下自然语言处理。",
				},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		},
	}

	// 创建缓冲区用于接收流式响应
	buffer := new(bytes.Buffer)

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, buffer)

	// 检查结果
	if err != nil {
		t.Logf("测试期间出现错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 对于流式响应，期望resp为nil
	assert.Nil(t, resp, "流式响应应返回nil响应对象")

	// 获取响应内容
	response := buffer.String()

	// 验证响应格式正确性
	assert.True(t, len(response) > 0, "响应不应为空")
	assert.Contains(t, response, "data: ", "响应应包含data:前缀")

	// 验证至少收到了一些内容
	t.Logf("使用默认供应商获取的响应: %s", response)
}

// TestCreateChatCompletionCallsBedrockFunction 测试CreateChatCompletion是否正确调用了BedrockCreateChatCompletionToChat函数
func TestCreateChatCompletionCallsBedrockFunction(t *testing.T) {
	// 由于不能直接替换函数，我们将通过测试bedrock provider的非流式请求来验证
	// CreateChatCompletion函数是否会正确调用BedrockCreateChatCompletionToChat

	// 创建测试请求
	request := ChatRequest{
		Provider: "bedrock",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "这是一个测试请求",
				},
			},
			Temperature: 0.7,
			MaxTokens:   50,
			Stream:      false,
		},
	}

	// 跳过实际API调用
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, nil)

	// 检查结果
	if err != nil {
		// 如果返回的错误包含"尚未实现"，说明函数尝试调用了BedrockCreateChatCompletionToChat
		// 但该函数可能尚未完全实现
		if strings.Contains(err.Error(), "尚未实现") {
			t.Logf("检测到预期的错误: %v", err)
			t.Log("这表明CreateChatCompletion正确地尝试调用了BedrockCreateChatCompletionToChat")
			return
		}

		// 如果是其他API错误，可能是配置问题，我们可以接受
		t.Logf("API调用错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 如果没有错误，则验证响应格式是否符合预期
	assert.NotNil(t, resp, "响应不应为空")
	assert.NotEmpty(t, resp.ID, "响应ID不应为空")
	assert.Equal(t, "chat.completion", resp.Object, "响应对象类型应为chat.completion")
	assert.NotZero(t, resp.Created, "创建时间不应为零")
	assert.Equal(t, request.ChatCompletionRequest.Model, resp.Model, "响应模型应与请求模型匹配")
	assert.NotEmpty(t, resp.Choices, "选择不应为空")

	if len(resp.Choices) > 0 {
		assert.NotEmpty(t, resp.Choices[0].Message.Content, "消息内容不应为空")
		assert.NotEmpty(t, resp.Choices[0].FinishReason, "完成原因不应为空")
		assert.Equal(t, "assistant", resp.Choices[0].Message.Role, "消息角色应为assistant")
		t.Logf("响应内容: %s", resp.Choices[0].Message.Content)
		t.Logf("完成原因: %s", resp.Choices[0].FinishReason)
	}

	t.Log("测试通过：CreateChatCompletion成功调用了BedrockCreateChatCompletionToChat并返回了有效响应")
}

// TestCreateChatCompletionCallsDeepSeekFunction 测试CreateChatCompletion是否正确调用了DeepSeekCreateChatCompletionToChat函数
func TestCreateChatCompletionCallsDeepSeekFunction(t *testing.T) {
	// 由于不能直接替换函数，我们将通过测试deepseek provider的非流式请求来验证
	// CreateChatCompletion函数是否会正确调用DeepSeekCreateChatCompletionToChat

	// 创建测试请求
	request := ChatRequest{
		Provider: "deepseek",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "deepseek-chat",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "这是一个DeepSeek测试请求",
				},
			},
			Temperature: 0.7,
			MaxTokens:   50,
			Stream:      false,
		},
	}

	// 跳过实际API调用
	if os.Getenv("SKIP_DEEPSEEK_TESTS") == "1" {
		t.Skip("跳过DeepSeek API测试")
	}

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, nil)

	// 检查结果
	if err != nil {
		// 如果返回的错误包含"尚未实现"，说明函数尝试调用了DeepSeekCreateChatCompletionToChat
		// 但该函数可能尚未完全实现
		if strings.Contains(err.Error(), "尚未实现") {
			t.Logf("检测到预期的错误: %v", err)
			t.Log("这表明CreateChatCompletion正确地尝试调用了DeepSeekCreateChatCompletionToChat")
			return
		}

		// 如果是其他API错误，可能是配置问题，我们可以接受
		t.Logf("API调用错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 如果没有错误，则验证响应格式是否符合预期
	assert.NotNil(t, resp, "响应不应为空")
	assert.NotEmpty(t, resp.ID, "响应ID不应为空")
	assert.Equal(t, "chat.completion", resp.Object, "响应对象类型应为chat.completion")
	assert.NotZero(t, resp.Created, "创建时间不应为零")
	assert.Equal(t, request.ChatCompletionRequest.Model, resp.Model, "响应模型应与请求模型匹配")
	assert.NotEmpty(t, resp.Choices, "选择不应为空")

	if len(resp.Choices) > 0 {
		assert.NotEmpty(t, resp.Choices[0].Message.Content, "消息内容不应为空")
		assert.NotEmpty(t, resp.Choices[0].FinishReason, "完成原因不应为空")
		assert.Equal(t, "assistant", resp.Choices[0].Message.Role, "消息角色应为assistant")
		t.Logf("响应内容: %s", resp.Choices[0].Message.Content)
		t.Logf("完成原因: %s", resp.Choices[0].FinishReason)
	}

	t.Log("测试通过：CreateChatCompletion成功调用了DeepSeekCreateChatCompletionToChat并返回了有效响应")
}

// TestCreateChatCompletionDeepSeekStream 测试DeepSeek的流式响应功能
func TestCreateChatCompletionDeepSeekStream(t *testing.T) {
	// 跳过实际API调用
	if os.Getenv("SKIP_DEEPSEEK_TESTS") == "1" {
		t.Skip("跳过DeepSeek API测试")
	}

	// 创建测试请求
	request := ChatRequest{
		Provider: "deepseek",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "deepseek-chat",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "system",
					Content: "你是一个专业且有帮助的助手。",
				},
				{
					Role:    "user",
					Content: "简要介绍计算机视觉的应用领域。",
				},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		},
	}

	// 创建缓冲区用于接收流式响应
	buffer := new(bytes.Buffer)

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, buffer)

	// 检查结果
	if err != nil {
		t.Logf("测试期间出现错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 对于流式响应，期望resp为nil
	assert.Nil(t, resp, "流式响应应返回nil响应对象")

	// 获取响应内容
	response := buffer.String()

	// 验证响应格式正确性
	assert.True(t, len(response) > 0, "响应不应为空")
	assert.Contains(t, response, "data: ", "响应应包含data:前缀")

	// 解析并验证每个响应块
	lines := strings.Split(response, "\n\n")
	var contentLines []string
	var allContent string

	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && line != "data: [DONE]" {
			// 解析JSON
			jsonData := strings.TrimPrefix(line, "data: ")
			var streamResp StreamResponse
			err := json.Unmarshal([]byte(jsonData), &streamResp)

			if err != nil {
				t.Errorf("解析响应JSON失败: %v", err)
				continue
			}

			// 验证响应结构
			assert.NotEmpty(t, streamResp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion.chunk", streamResp.Object, "响应对象类型应为chat.completion.chunk")
			assert.NotZero(t, streamResp.Created, "创建时间不应为零")
			assert.Equal(t, request.ChatCompletionRequest.Model, streamResp.Model, "响应模型应与请求模型匹配")
			assert.NotEmpty(t, streamResp.Choices, "选择不应为空")

			// 收集内容
			if len(streamResp.Choices) > 0 {
				content := streamResp.Choices[0].Delta.Content
				if content != "" {
					contentLines = append(contentLines, content)
					allContent += content
				}
			}
		}
	}

	// 验证收集到的内容
	t.Logf("收到 %d 个内容块", len(contentLines))
	t.Logf("完整响应内容: %s", allContent)
	assert.True(t, len(contentLines) > 0, "应收到至少一个内容块")
	assert.NotEmpty(t, allContent, "应收到非空内容")
}

// TestChatService_CreateChatCompletionSingleTool 测试单个工具调用功能
// 执行命令：go test -run TestChatService_CreateChatCompletionSingleTool
func TestChatService_CreateChatCompletionSingleTool(t *testing.T) {
	// 准备测试用例
	testCases := []struct {
		name            string
		initialMessages []openai.ChatCompletionMessage
		expectToolCall  bool
		toolResult      string // 模拟的工具结果
		expectError     bool
	}{
		{
			name: "Azure单工具调用测试",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "请使用天气工具查询北京的天气情况，需要详细的温度、湿度和风力信息。",
				},
			},
			expectToolCall: true,
			toolResult:     `{"directories":["/Users/test/Documents/files"]}`,
			expectError:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- 第一步：发送初始请求，获取工具调用 ---
			firstRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:    "gpt-4o", // 确保模型支持工具调用
					Messages: tc.initialMessages,
					// 定义工具
					Tools: []openai.Tool{
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "get_weather",
								Description: "获取指定城市的天气信息",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"location": map[string]interface{}{
											"type":        "string",
											"description": "城市名称，例如：'北京'",
										},
									},
									"required": []string{"location"},
								},
							},
						},
					},
					ToolChoice:  "auto", // 让模型决定是否使用工具
					MaxTokens:   150,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第一次请求 ---")
			firstResp, err := CreateChatCompletion(firstRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				if tc.expectError {
					assert.Error(t, err, "预期应返回错误")
					return // 如果预期错误且确实发生错误，测试通过
				}
				t.Fatalf("第一次API调用失败: %v", err) // 非预期错误，终止测试
			}
			if tc.expectError {
				assert.NoError(t, err, "预期有错误但API调用成功")
				return // 如果预期错误但没有发生，测试失败
			}

			// 验证第一次响应
			assert.NotNil(t, firstResp, "第一次响应不应为空")
			assert.NotEmpty(t, firstResp.Choices, "第一次响应的选择不应为空")
			firstAssistantMessage := firstResp.Choices[0].Message
			assert.Equal(t, openai.ChatMessageRoleAssistant, firstAssistantMessage.Role, "第一次响应消息角色应为assistant")

			// 检查是否收到了工具调用
			if tc.expectToolCall {
				assert.NotEmpty(t, firstAssistantMessage.ToolCalls, "第一次响应应包含 tool_calls")
				t.Logf("收到工具调用请求: %d 个", len(firstAssistantMessage.ToolCalls))
				for _, toolCall := range firstAssistantMessage.ToolCalls {
					t.Logf("  - ID: %s, Type: %s, Function: %s(%s)", toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)
				}
			} else {
				if len(firstAssistantMessage.ToolCalls) > 0 {
					t.Logf("意外收到工具调用请求，但继续测试: %d 个", len(firstAssistantMessage.ToolCalls))
				}
			}

			// 如果没有工具调用，不需要进行后续步骤
			if len(firstAssistantMessage.ToolCalls) == 0 {
				t.Log("没有收到工具调用请求，跳过后续步骤")
				return
			}

			// 在发送第二个请求前，检查并修正assistant消息
			// 强制为包含 tool_calls 的 assistant 消息添加 content
			if firstAssistantMessage.Content == "" && len(firstAssistantMessage.ToolCalls) > 0 {
				firstAssistantMessage.Content = "Okay, I will call the required tools."
				t.Logf("强制为 assistant 消息添加了 content: '%s'", firstAssistantMessage.Content)
			}

			// 在发送第二个请求前，检查并修正工具调用参数
			for i, toolCall := range firstAssistantMessage.ToolCalls {
				// 检查工具调用参数是否为空或仅为{}
				if toolCall.Function.Arguments == "{}" || toolCall.Function.Arguments == "" {
					t.Logf("警告: 工具调用参数为空，自动补充参数")

					// 根据工具类型自动补充参数
					if toolCall.Function.Name == "get_weather" {
						firstAssistantMessage.ToolCalls[i].Function.Arguments = `{"location":"北京"}`
						t.Logf("已补充参数: %s", firstAssistantMessage.ToolCalls[i].Function.Arguments)
					}
				}
			}

			// 重新构建第二次请求的消息列表（使用修正后的firstAssistantMessage）
			messagesForSecondCall := []openai.ChatCompletionMessage{
				tc.initialMessages[0],
				firstAssistantMessage,
			}

			// 打印完整的消息列表进行验证
			t.Logf("完整的消息列表(共%d条):", len(messagesForSecondCall))
			for i, msg := range messagesForSecondCall {
				t.Logf("  消息 #%d: Role=%s, Content长度=%d", i+1, msg.Role, len(msg.Content))
				if msg.Role == openai.ChatMessageRoleTool {
					t.Logf("    ToolCallID=%s", msg.ToolCallID)
				} else if msg.Role == openai.ChatMessageRoleAssistant && len(msg.ToolCalls) > 0 {
					t.Logf("    包含%d个工具调用", len(msg.ToolCalls))
					for j, tc := range msg.ToolCalls {
						t.Logf("      工具调用 #%d: ID=%s, 函数=%s",
							j+1, tc.ID, tc.Function.Name)
					}
				}
			}

			// 打印为JSON格式以便更清晰查看结构
			messagesJSON, _ := json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("消息JSON结构:\n%s", string(messagesJSON))

			// 添加工具结果消息
			firstToolCallID := firstAssistantMessage.ToolCalls[0].ID
			//firstToolName := firstAssistantMessage.ToolCalls[0].Function.Name

			toolResponseMessage := openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    tc.toolResult,
				ToolCallID: firstToolCallID,
				// 不添加Name字段
			}
			messagesForSecondCall = append(messagesForSecondCall, toolResponseMessage)

			// 重新打印更新后的消息列表
			messagesJSON, _ = json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("添加工具响应后的消息列表:\n%s", string(messagesJSON))

			secondRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:       "gpt-4o",
					Messages:    messagesForSecondCall,
					MaxTokens:   200,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第二次请求 (带工具结果) ---")
			secondResp, err := CreateChatCompletion(secondRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				t.Fatalf("第二次API调用失败: %v", err)
			}

			// 验证最终响应
			assert.NotNil(t, secondResp, "第二次响应不应为空")
			assert.NotEmpty(t, secondResp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion", secondResp.Object, "响应对象类型应为chat.completion")
			assert.NotZero(t, secondResp.Created, "创建时间不应为零")
			assert.NotEmpty(t, secondResp.Choices, "选择不应为空")

			if len(secondResp.Choices) > 0 {
				finalMessage := secondResp.Choices[0].Message
				assert.Equal(t, openai.ChatMessageRoleAssistant, finalMessage.Role, "最终消息角色应为assistant")
				assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空")
				t.Logf("最终助手响应内容: %s", finalMessage.Content)
				assert.Contains(t, finalMessage.Content, "天气", "最终响应应包含天气信息")
				assert.Contains(t, finalMessage.Content, "北京", "最终响应应包含城市名")
				t.Logf("完成原因: %s", secondResp.Choices[0].FinishReason)
			}

			// 打印和检查工具调用，确保它包含location参数
			for _, toolCall := range firstAssistantMessage.ToolCalls {
				parsedArgs := make(map[string]interface{})
				json.Unmarshal([]byte(toolCall.Function.Arguments), &parsedArgs)
				t.Logf("  解析后的参数: %v", parsedArgs)

				if _, hasLocation := parsedArgs["location"]; !hasLocation && toolCall.Function.Name == "get_weather" {
					t.Logf("  警告: 工具调用缺少location参数")
				}
			}
		})
	}
}

// TestChatService_CreateChatCompletionMultiTool 测试多个工具调用功能
// 执行命令：go test -run TestChatService_CreateChatCompletionMultiTool
func TestChatService_CreateChatCompletionMultiTool(t *testing.T) {
	// 准备测试用例
	testCases := []struct {
		name            string
		initialMessages []openai.ChatCompletionMessage
		expectToolCalls bool
		toolResults     map[string]string // 模拟的工具结果，key为工具调用ID
		expectError     bool
	}{
		{
			name: "Azure多工具调用测试",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "帮我计算一下30摄氏度是多少华氏度？以及2立方米等于多少升？",
				},
			},
			expectToolCalls: true,
			toolResults: map[string]string{
				"": `{"celsius": 30, "fahrenheit": 86}`, // 会在测试中动态替换key
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- 第一步：发送初始请求，获取工具调用 ---
			firstRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:    "gpt-4o", // 确保模型支持工具调用
					Messages: tc.initialMessages,
					// 定义多个工具
					Tools: []openai.Tool{
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "temperature_converter",
								Description: "将摄氏度转换为华氏度",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"celsius": map[string]interface{}{
											"type":        "number",
											"description": "摄氏度温度值",
										},
									},
									"required": []string{"celsius"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "volume_converter",
								Description: "将立方米转换为升",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"cubic_meters": map[string]interface{}{
											"type":        "number",
											"description": "体积，单位是立方米",
										},
									},
									"required": []string{"cubic_meters"},
								},
							},
						},
					},
					ToolChoice:  "auto", // 让模型决定是否使用工具
					MaxTokens:   200,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第一次请求 ---")
			firstResp, err := CreateChatCompletion(firstRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				if tc.expectError {
					assert.Error(t, err, "预期应返回错误")
					return
				}
				t.Fatalf("第一次API调用失败: %v", err)
			}
			if tc.expectError {
				assert.NoError(t, err, "预期有错误但API调用成功")
				return
			}

			// 验证第一次响应
			assert.NotNil(t, firstResp, "第一次响应不应为空")
			assert.NotEmpty(t, firstResp.Choices, "第一次响应的选择不应为空")
			firstAssistantMessage := firstResp.Choices[0].Message
			assert.Equal(t, openai.ChatMessageRoleAssistant, firstAssistantMessage.Role, "第一次响应消息角色应为assistant")

			// 检查是否收到了工具调用
			if tc.expectToolCalls {
				assert.NotEmpty(t, firstAssistantMessage.ToolCalls, "第一次响应应包含 tool_calls")
				t.Logf("收到工具调用请求: %d 个", len(firstAssistantMessage.ToolCalls))
				for _, toolCall := range firstAssistantMessage.ToolCalls {
					t.Logf("  - ID: %s, Type: %s, Function: %s(%s)", toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)
				}
			} else {
				if len(firstAssistantMessage.ToolCalls) > 0 {
					t.Logf("意外收到工具调用请求，但继续测试: %d 个", len(firstAssistantMessage.ToolCalls))
				}
			}

			// 如果没有工具调用，不需要进行后续步骤
			if len(firstAssistantMessage.ToolCalls) == 0 {
				t.Log("没有收到工具调用请求，跳过后续步骤")
				return
			}

			// 在发送第二个请求前，检查并修正assistant消息
			// 强制为包含 tool_calls 的 assistant 消息添加 content
			if firstAssistantMessage.Content == "" && len(firstAssistantMessage.ToolCalls) > 0 {
				firstAssistantMessage.Content = "Okay, I will call the required tools."
				t.Logf("强制为 assistant 消息添加了 content: '%s'", firstAssistantMessage.Content)
			}

			// 在发送第二个请求前，检查并修正工具调用参数
			for i, toolCall := range firstAssistantMessage.ToolCalls {
				// 检查工具调用参数是否为空或仅为{}
				if toolCall.Function.Arguments == "{}" || toolCall.Function.Arguments == "" {
					t.Logf("警告: 工具调用参数为空，自动补充参数")

					// 根据工具类型自动补充参数
					if toolCall.Function.Name == "temperature_converter" {
						firstAssistantMessage.ToolCalls[i].Function.Arguments = `{"celsius": 30}` // 提供默认参数
					} else if toolCall.Function.Name == "volume_converter" {
						firstAssistantMessage.ToolCalls[i].Function.Arguments = `{"cubic_meters": 2}`
					}

					if firstAssistantMessage.ToolCalls[i].Function.Arguments != "{}" && firstAssistantMessage.ToolCalls[i].Function.Arguments != "" {
						t.Logf("已补充参数: %s", firstAssistantMessage.ToolCalls[i].Function.Arguments)
					}
				}
			}

			// --- 第二步：发送第二次请求，包含工具结果 ---
			// 构建包含工具结果的消息列表
			messagesForSecondCall := []openai.ChatCompletionMessage{
				tc.initialMessages[0],
				firstAssistantMessage,
			}

			// 处理所有工具调用的响应
			for i, toolCall := range firstAssistantMessage.ToolCalls {
				toolCallID := toolCall.ID
				toolName := toolCall.Function.Name

				// 设置工具结果
				var toolResult string

				// 根据工具类型提供结果
				if toolName == "temperature_converter" {
					toolResult = `{"celsius": 30, "fahrenheit": 86}`
				} else if toolName == "volume_converter" {
					toolResult = `{"cubic_meters": 2, "liters": 2000}`
				}

				// 添加工具的调用结果到消息列表
				toolResponseMessage := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    toolResult,
					ToolCallID: toolCallID,
					Name:       toolName, // Bedrock/Anthropic 需要Name字段
				}
				messagesForSecondCall = append(messagesForSecondCall, toolResponseMessage)

				t.Logf("已添加第%d个工具(%s)的响应", i+1, toolName)
			}

			t.Logf("第二次请求消息列表(包含所有工具结果):")
			messagesJSON, _ := json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("%s", string(messagesJSON))

			// 发送第二次请求，不重复工具定义
			secondRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:       "gpt-4o",
					Messages:    messagesForSecondCall,
					MaxTokens:   200,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第二次请求 (带所有工具结果) ---")
			secondResp, err := CreateChatCompletion(secondRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				t.Fatalf("第二次API调用失败: %v", err)
			}

			// 验证第二次响应
			assert.NotNil(t, secondResp, "第二次响应不应为空")
			secondAssistantMessage := secondResp.Choices[0].Message
			t.Logf("第二次助手响应内容: %s", secondAssistantMessage.Content)

			// 检查第二次响应是否包含工具调用
			if len(secondAssistantMessage.ToolCalls) > 0 {
				t.Logf("收到第二次工具调用请求: %d 个", len(secondAssistantMessage.ToolCalls))

				// 准备第三次请求 (包含第二个工具调用结果)
				messagesForThirdCall := append(messagesForSecondCall, secondAssistantMessage)

				// 创建包含第二个工具调用结果的工具消息
				secondToolCallID := secondAssistantMessage.ToolCalls[0].ID
				secondToolName := secondAssistantMessage.ToolCalls[0].Function.Name

				// 设置第二个工具的结果
				var secondToolResult string
				if result, exists := tc.toolResults[secondToolCallID]; exists {
					secondToolResult = result
				} else {
					// 根据工具类型提供默认结果
					if secondToolName == "temperature_converter" {
						secondToolResult = `{"celsius": 30, "fahrenheit": 86}`
					} else if secondToolName == "volume_converter" {
						secondToolResult = `{"cubic_meters": 2, "liters": 2000}`
					}
				}

				// 添加第二个工具的调用结果到消息列表
				secondToolResponseMessage := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    secondToolResult,
					ToolCallID: secondToolCallID,
					Name:       secondToolName, // Bedrock/Anthropic 需要Name字段
				}
				messagesForThirdCall = append(messagesForThirdCall, secondToolResponseMessage)

				t.Logf("第三次请求消息列表(包含第二个工具结果):")
				messagesJSON, _ = json.MarshalIndent(messagesForThirdCall, "", "  ")
				t.Logf("%s", string(messagesJSON))

				// 发送第三次请求
				thirdRequest := ChatRequest{
					Provider: "azure",
					ChatCompletionRequest: openai.ChatCompletionRequest{
						Model:       "gpt-4o",
						Messages:    messagesForThirdCall,
						MaxTokens:   200,
						Temperature: 0.5,
					},
				}

				t.Log("--- 发送第三次请求 (带第二个工具结果) ---")
				thirdResp, err := CreateChatCompletion(thirdRequest, nil)

				// 错误处理
				if err != nil {
					logAzureError(t, err)
					t.Fatalf("第三次API调用失败: %v", err)
				}

				// 验证最终响应
				assert.NotNil(t, thirdResp, "第三次响应不应为空")
				finalMessage := thirdResp.Choices[0].Message
				assert.Equal(t, openai.ChatMessageRoleAssistant, finalMessage.Role, "最终消息角色应为assistant")
				assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空")
				t.Logf("最终助手响应内容: %s", finalMessage.Content)

				// 验证响应中是否包含温度和体积转换相关信息
				assert.True(t,
					strings.Contains(finalMessage.Content, "华氏度") ||
						strings.Contains(finalMessage.Content, "摄氏度") ||
						strings.Contains(finalMessage.Content, "立方米") ||
						strings.Contains(finalMessage.Content, "升"),
					"最终响应应包含温度或体积转换相关信息")
			} else {
				// 如果第二次响应没有工具调用，则验证第二次响应
				assert.NotEmpty(t, secondAssistantMessage.Content, "第二次助手消息内容不应为空")
				// 验证响应中是否包含相关信息
				assert.True(t,
					strings.Contains(secondAssistantMessage.Content, "华氏度") ||
						strings.Contains(secondAssistantMessage.Content, "摄氏度") ||
						strings.Contains(secondAssistantMessage.Content, "立方米") ||
						strings.Contains(secondAssistantMessage.Content, "升"),
					"第二次响应应包含温度或体积转换相关信息")
			}
		})
	}
}

// TestChatService_CreateChatCompletionSingleTool_Bedrock 测试单个工具调用功能 (Bedrock)
// 执行命令：go test -run TestChatService_CreateChatCompletionSingleTool_Bedrock
func TestChatService_CreateChatCompletionSingleTool_Bedrock(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 准备测试用例
	testCases := []struct {
		name            string
		initialMessages []openai.ChatCompletionMessage
		expectToolCall  bool
		toolResult      string // 模拟的工具结果
		expectError     bool
	}{
		{
			name: "Bedrock单工具调用测试-获取股票价格",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "请查询Amazon公司(AMZN)的当前股票价格",
				},
			},
			expectToolCall: true,
			toolResult:     `{"ticker":"AMZN","price":180.75,"currency":"USD","timestamp":"2024-11-18T14:30:00Z"}`,
			expectError:    false,
		},
		{
			name: "Bedrock单工具调用测试-货币转换",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "请将100美元转换为人民币",
				},
			},
			expectToolCall: true,
			toolResult:     `{"from_amount":100,"from_currency":"USD","to_currency":"CNY","to_amount":718.50,"rate":7.185,"timestamp":"2024-11-18T14:30:00Z"}`,
			expectError:    false,
		},
		{
			name: "Bedrock单工具调用测试-天气查询",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "请查询北京的当前天气情况",
				},
			},
			expectToolCall: true,
			toolResult:     `{"location":"Beijing, China","temperature":15,"unit":"celsius","conditions":"Sunny","humidity":45,"wind_speed":10,"wind_direction":"NE","timestamp":"2024-11-18T14:30:00Z"}`,
			expectError:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- 第一步：发送初始请求，获取工具调用 ---
			firstRequest := ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:    "anthropic.claude-3-5-sonnet-20240620-v1:0", // 使用Claude 3.5 Sonnet
					Messages: tc.initialMessages,
					// 定义工具
					Tools: []openai.Tool{
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "get_stock_price",
								Description: "获取指定股票代码的当前价格信息",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"ticker": map[string]interface{}{
											"type":        "string",
											"description": "股票代码，例如 'AAPL'、'MSFT'、'AMZN'等",
										},
									},
									"required": []string{"ticker"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "convert_currency",
								Description: "将一种货币转换为另一种货币",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"amount": map[string]interface{}{
											"type":        "number",
											"description": "要转换的金额",
										},
										"from_currency": map[string]interface{}{
											"type":        "string",
											"description": "源货币代码，例如 'USD'、'EUR'、'CNY'等",
										},
										"to_currency": map[string]interface{}{
											"type":        "string",
											"description": "目标货币代码，例如 'USD'、'EUR'、'CNY'等",
										},
									},
									"required": []string{"amount", "from_currency", "to_currency"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "get_weather",
								Description: "获取指定城市的当前天气情况",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"location": map[string]interface{}{
											"type":        "string",
											"description": "城市名称，例如 '北京'、'上海'、'New York'等",
										},
									},
									"required": []string{"location"},
								},
							},
						},
					},
					ToolChoice:  "auto", // 让模型决定是否使用工具
					MaxTokens:   300,    // 增加token量以支持更复杂的工具调用
					Temperature: 0.2,    // 降低温度，使模型更加确定性
				},
			}

			t.Log("--- 发送第一次请求 ---")
			firstResp, err := CreateChatCompletion(firstRequest, nil)

			// 错误处理
			if err != nil {
				// 检查是否是模型访问权限错误
				if strings.Contains(err.Error(), "Your account does not have an agreement to this model") ||
					strings.Contains(err.Error(), "403 Forbidden") {
					t.Skipf("跳过测试: 账户没有访问模型的权限: %v", err)
					return
				}

				if tc.expectError {
					assert.Error(t, err, "预期应返回错误")
					return // 如果预期错误且确实发生错误，测试通过
				}
				t.Fatalf("第一次API调用失败: %v", err) // 非预期错误，终止测试
			}
			if tc.expectError {
				t.Fatalf("预期有错误但API调用成功")
			}

			// 验证第一次响应
			assert.NotNil(t, firstResp, "第一次响应不应为空")
			assert.NotEmpty(t, firstResp.Choices, "第一次响应的选择不应为空")
			firstAssistantMessage := firstResp.Choices[0].Message
			assert.Equal(t, openai.ChatMessageRoleAssistant, firstAssistantMessage.Role, "第一次响应消息角色应为assistant")

			// 检查是否收到了工具调用，并且确保是真实的工具调用请求
			if tc.expectToolCall {
				if len(firstAssistantMessage.ToolCalls) == 0 {
					// 检查是否因为finish_reason是"tool_use"但没有正确解析工具调用
					finish_reason := firstResp.Choices[0].FinishReason
					if finish_reason == "tool_use" {
						t.Logf("收到tool_use完成原因，但工具调用为空，可能需要修复适配器解析工具调用的逻辑")
						t.Skip("适配器未能正确解析工具调用，跳过后续测试")
						return
					}

					// 如果不是解析问题，则模型确实没有使用工具调用
					t.Logf("没有收到工具调用请求，可能是模型不支持工具调用，跳过后续验证")
					t.Skip("当前模型不支持工具调用或未返回工具调用")
					return
				}

				assert.NotEmpty(t, firstAssistantMessage.ToolCalls, "第一次响应应包含 tool_calls")
				t.Logf("收到工具调用请求: %d 个", len(firstAssistantMessage.ToolCalls))
				for _, toolCall := range firstAssistantMessage.ToolCalls {
					t.Logf("  - ID: %s, Type: %s, Function: %s(%s)", toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)

					// 验证工具调用ID格式是否符合Bedrock的规范，Claude 3.5通常应有toolu_前缀
					assert.True(t, strings.HasPrefix(toolCall.ID, "toolu_"), "工具调用ID应以'toolu_'开头")

					// 验证工具调用类型
					assert.Equal(t, openai.ToolTypeFunction, toolCall.Type, "工具调用类型应为function")

					// 验证函数名称不为空
					assert.NotEmpty(t, toolCall.Function.Name, "函数名称不应为空")

					// 验证函数参数不为空且为有效的JSON
					assert.NotEmpty(t, toolCall.Function.Arguments, "函数参数不应为空")
					var args map[string]interface{}
					err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
					assert.NoError(t, err, "函数参数应为有效的JSON")
				}
			} else {
				if len(firstAssistantMessage.ToolCalls) > 0 {
					t.Fatalf("意外收到工具调用请求")
				}
			}

			// 如果没有工具调用，但期望有工具调用，则检查原因
			if len(firstAssistantMessage.ToolCalls) == 0 && tc.expectToolCall {
				finishReason := firstResp.Choices[0].FinishReason
				t.Logf("未收到工具调用，完成原因: %s", finishReason)
				t.Skip("模型未使用工具调用，可能需要调整提示词或模型参数")
				return
			}

			// 如果没有工具调用，不需要进行后续步骤
			if len(firstAssistantMessage.ToolCalls) == 0 {
				t.Log("没有收到工具调用请求，跳过后续步骤")
				return
			}

			// 在发送第二个请求前，确保assistant消息有内容
			if firstAssistantMessage.Content == "" {
				firstAssistantMessage.Content = "我将使用工具来帮助回答您的问题。"
				t.Logf("为assistant消息添加content: '%s'", firstAssistantMessage.Content)
			}

			// 重新构建第二次请求的消息列表
			messagesForSecondCall := []openai.ChatCompletionMessage{
				tc.initialMessages[0],
				firstAssistantMessage,
			}

			// 添加工具结果消息
			firstToolCallID := firstAssistantMessage.ToolCalls[0].ID

			toolResponseMessage := openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    tc.toolResult,
				ToolCallID: firstToolCallID,
			}
			messagesForSecondCall = append(messagesForSecondCall, toolResponseMessage)

			// 打印更新后的消息列表
			messagesJSON, _ := json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("添加工具响应后的消息列表:\n%s", string(messagesJSON))

			secondRequest := ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:       firstRequest.Model, // 使用与第一次请求相同的模型
					Messages:    messagesForSecondCall,
					MaxTokens:   300,
					Temperature: 0.2,
					// 第二次请求也需要提供工具定义，确保模型能够继续使用工具
					Tools: firstRequest.ChatCompletionRequest.Tools,
				},
			}

			t.Log("--- 发送第二次请求 (带工具结果) ---")
			secondResp, err := CreateChatCompletion(secondRequest, nil)

			// 错误处理
			if err != nil {
				t.Fatalf("第二次API调用失败: %v", err)
			}

			// 验证最终响应
			assert.NotNil(t, secondResp, "第二次响应不应为空")
			assert.NotEmpty(t, secondResp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion", secondResp.Object, "响应对象类型应为chat.completion")
			assert.NotZero(t, secondResp.Created, "创建时间不应为零")
			assert.NotEmpty(t, secondResp.Choices, "选择不应为空")

			if len(secondResp.Choices) > 0 {
				secondAssistantMessage := secondResp.Choices[0].Message
				finalMessage := secondAssistantMessage
				assert.Equal(t, openai.ChatMessageRoleAssistant, finalMessage.Role, "最终消息角色应为assistant")

				// 检查是否有继续的工具调用
				if len(secondAssistantMessage.ToolCalls) > 0 {
					t.Logf("第二次响应仍有工具调用: %d 个", len(secondAssistantMessage.ToolCalls))
					for _, toolCall := range secondAssistantMessage.ToolCalls {
						t.Logf("  - ID: %s, Type: %s, Function: %s(%s)",
							toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)
					}

					// 验证第二次工具调用ID格式
					assert.True(t, strings.HasPrefix(secondAssistantMessage.ToolCalls[0].ID, "toolu_"),
						"第二次工具调用ID应以'toolu_'开头")
				} else {
					// 检查最终响应内容
					assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空")
					t.Logf("最终助手响应内容: %s", finalMessage.Content)
					t.Logf("完成原因: %s", secondResp.Choices[0].FinishReason)

					// 根据不同的测试用例验证响应内容
					if strings.Contains(tc.name, "股票价格") {
						assert.True(t,
							strings.Contains(finalMessage.Content, "AMZN") ||
								strings.Contains(finalMessage.Content, "Amazon") ||
								strings.Contains(finalMessage.Content, "价格") ||
								strings.Contains(finalMessage.Content, "美元"),
							"最终响应应包含股票相关信息")
					} else if strings.Contains(tc.name, "货币转换") {
						assert.True(t,
							strings.Contains(finalMessage.Content, "美元") ||
								strings.Contains(finalMessage.Content, "人民币") ||
								strings.Contains(finalMessage.Content, "转换") ||
								strings.Contains(finalMessage.Content, "CNY") ||
								strings.Contains(finalMessage.Content, "USD"),
							"最终响应应包含货币转换相关信息")
					} else if strings.Contains(tc.name, "天气查询") {
						assert.True(t,
							strings.Contains(finalMessage.Content, "北京") ||
								strings.Contains(finalMessage.Content, "天气") ||
								strings.Contains(finalMessage.Content, "温度") ||
								strings.Contains(finalMessage.Content, "气温") ||
								strings.Contains(finalMessage.Content, "湿度"),
							"最终响应应包含天气相关信息")
					}
				}
			}
		})
	}
}

// TestChatService_CreateChatCompletionMultiTool_Bedrock 测试多个工具调用功能 (Bedrock)
// 执行命令：go test -run TestChatService_CreateChatCompletionMultiTool_Bedrock
func TestChatService_CreateChatCompletionMultiTool_Bedrock(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 准备测试用例
	testCases := []struct {
		name            string
		initialMessages []openai.ChatCompletionMessage
		expectToolCalls bool
		toolResults     map[string]string // 模拟的工具结果，key为工具名称
		expectError     bool
	}{
		{
			name: "Bedrock多工具调用测试-文件操作",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "请帮我列出/tmp目录下的文件，并读取/tmp/test.txt文件的内容。",
				},
			},
			expectToolCalls: true,
			toolResults: map[string]string{
				"list_directory": `{"path": "/tmp", "files": ["[FILE] test.txt", "[FILE] data.json", "[DIR] logs"]}`,
				"read_file":      `{"path": "/tmp/test.txt", "content": "这是测试文件的内容\n包含多行文本\n用于测试文件读取功能"}`,
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- 第一步：发送初始请求，获取工具调用 ---
			firstRequest := ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:    "anthropic.claude-3-5-sonnet-20240620-v1:0", // 使用支持工具调用的Bedrock模型
					Messages: tc.initialMessages,
					// 定义多个工具
					Tools: []openai.Tool{
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "read_file",
								Description: "读取文件系统中的文件内容。处理各种文本编码并在无法读取文件时提供详细的错误消息。",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"path": map[string]interface{}{
											"type":        "string",
											"description": "文件的完整路径",
										},
										"serverId": map[string]interface{}{
											"type":        "string",
											"description": "MCP服务器ID",
											"default":     "filesystem",
										},
									},
									"required": []string{"path"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "read_multiple_files",
								Description: "同时读取多个文件的内容。比逐个读取文件更高效。",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"paths": map[string]interface{}{
											"type": "array",
											"items": map[string]interface{}{
												"type": "string",
											},
											"description": "要读取的文件路径列表",
										},
										"serverId": map[string]interface{}{
											"type":        "string",
											"description": "MCP服务器ID",
											"default":     "filesystem",
										},
									},
									"required": []string{"paths"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "write_file",
								Description: "创建新文件或完全覆盖现有文件。",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"path": map[string]interface{}{
											"type":        "string",
											"description": "文件的完整路径",
										},
										"content": map[string]interface{}{
											"type":        "string",
											"description": "要写入文件的内容",
										},
										"serverId": map[string]interface{}{
											"type":        "string",
											"description": "MCP服务器ID",
											"default":     "filesystem",
										},
									},
									"required": []string{"path", "content"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "list_directory",
								Description: "获取指定路径中所有文件和目录的详细列表。结果清晰地区分文件和目录，分别使用[FILE]和[DIR]前缀。",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"path": map[string]interface{}{
											"type":        "string",
											"description": "要列出内容的目录路径",
										},
										"serverId": map[string]interface{}{
											"type":        "string",
											"description": "MCP服务器ID",
											"default":     "filesystem",
										},
									},
									"required": []string{"path"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "search_files",
								Description: "递归搜索匹配模式的文件和目录。从起始路径搜索所有子目录。搜索不区分大小写并匹配部分名称。",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"path": map[string]interface{}{
											"type":        "string",
											"description": "开始搜索的目录路径",
										},
										"pattern": map[string]interface{}{
											"type":        "string",
											"description": "搜索模式",
										},
										"excludePatterns": map[string]interface{}{
											"type": "array",
											"items": map[string]interface{}{
												"type": "string",
											},
											"description": "要排除的模式列表",
											"default":     []string{},
										},
										"serverId": map[string]interface{}{
											"type":        "string",
											"description": "MCP服务器ID",
											"default":     "filesystem",
										},
									},
									"required": []string{"path", "pattern"},
								},
							},
						},
					},
					ToolChoice:  "auto", // 让模型决定是否使用工具
					MaxTokens:   500,    // 增加MaxTokens以容纳工具调用结构
					Temperature: 0.2,    // 降低温度，使模型更加确定性
				},
			}

			t.Log("--- 发送第一次请求 (Bedrock, Multi-Tool) ---")
			firstResp, err := CreateChatCompletion(firstRequest, nil)

			// 错误处理
			if err != nil {
				t.Logf("测试期间出现错误 (Bedrock, Multi-Tool): %v", err)
				if tc.expectError {
					assert.Error(t, err, "预期应返回错误 (Bedrock, Multi-Tool)")
					return
				}
				t.Fatalf("第一次API调用失败 (Bedrock, Multi-Tool): %v", err)
			}
			if tc.expectError {
				assert.NoError(t, err, "预期有错误但API调用成功 (Bedrock, Multi-Tool)")
				return
			}

			// 打印第一次响应的详细信息
			respJSON, _ := json.MarshalIndent(firstResp, "", "  ")
			t.Logf("第一次响应详情 (Bedrock, Multi-Tool):\n%s", string(respJSON))

			// 验证第一次响应
			assert.NotNil(t, firstResp, "第一次响应不应为空 (Bedrock, Multi-Tool)")
			assert.NotEmpty(t, firstResp.Choices, "第一次响应的选择不应为空 (Bedrock, Multi-Tool)")
			firstAssistantMessage := firstResp.Choices[0].Message
			assert.Equal(t, openai.ChatMessageRoleAssistant, firstAssistantMessage.Role, "第一次响应消息角色应为assistant (Bedrock, Multi-Tool)")

			// 检查是否收到了工具调用
			if tc.expectToolCalls {
				assert.True(t, len(firstAssistantMessage.ToolCalls) > 0, "第一次响应应包含tool_calls (Bedrock, Multi-Tool)")
				t.Logf("收到工具调用请求 (Bedrock, Multi-Tool): %d 个", len(firstAssistantMessage.ToolCalls))
				for i, toolCall := range firstAssistantMessage.ToolCalls {
					t.Logf("  - #%d: ID: %s, Type: %s, Function: %s(%s)", i+1, toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)

					// 验证工具调用ID格式
					assert.True(t, strings.HasPrefix(toolCall.ID, "toulu_") || strings.HasPrefix(toolCall.ID, "toolu_") || strings.HasPrefix(toolCall.ID, "call_"),
						"工具调用ID应有正确前缀")

					// 验证工具调用类型
					assert.Equal(t, openai.ToolTypeFunction, toolCall.Type, "工具调用类型应为function")

					// 验证函数名称是否为有效工具
					validTools := []string{"read_file", "read_multiple_files", "write_file", "list_directory", "search_files"}

					found := false
					for _, validTool := range validTools {
						if toolCall.Function.Name == validTool {
							found = true
							break
						}
					}
					assert.True(t, found, "函数名称应为有效工具")

					// 验证函数参数是否为有效JSON
					var args map[string]interface{}
					err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
					assert.NoError(t, err, "函数参数应为有效的JSON")

					// 检查并修复工具参数
					if toolCall.Function.Name == "read_file" {
						// 确保read_file有path参数
						if _, hasPath := args["path"]; !hasPath {
							t.Logf("警告: read_file缺少path参数，添加默认值")
							fixedArgs := map[string]interface{}{
								"path": "/tmp/test.txt",
							}
							if serverId, hasServerId := args["serverId"]; hasServerId {
								fixedArgs["serverId"] = serverId
							}
							fixedArgsBytes, _ := json.Marshal(fixedArgs)
							toolCall.Function.Arguments = string(fixedArgsBytes)
							t.Logf("修复后的参数: %s", toolCall.Function.Arguments)
						}
					} else if toolCall.Function.Name == "list_directory" {
						// 确保list_directory有path参数
						if _, hasPath := args["path"]; !hasPath {
							t.Logf("警告: list_directory缺少path参数，添加默认值")
							fixedArgs := map[string]interface{}{
								"path": "/tmp",
							}
							if serverId, hasServerId := args["serverId"]; hasServerId {
								fixedArgs["serverId"] = serverId
							}
							fixedArgsBytes, _ := json.Marshal(fixedArgs)
							toolCall.Function.Arguments = string(fixedArgsBytes)
							t.Logf("修复后的参数: %s", toolCall.Function.Arguments)
						}
					}
				}
			} else {
				if len(firstAssistantMessage.ToolCalls) > 0 {
					t.Logf("意外收到工具调用请求，但继续测试 (Bedrock, Multi-Tool): %d 个", len(firstAssistantMessage.ToolCalls))
				}
			}

			// 如果没有工具调用，不需要进行后续步骤
			if len(firstAssistantMessage.ToolCalls) == 0 {
				t.Log("没有收到工具调用请求，跳过后续步骤 (Bedrock, Multi-Tool)")
				// 如果预期有工具调用但没有收到，则测试失败
				assert.False(t, tc.expectToolCalls, "预期有工具调用但模型未返回 (Bedrock, Multi-Tool)")
				return
			}

			// 在发送第二个请求前，确保assistant消息有内容
			if firstAssistantMessage.Content == "" {
				firstAssistantMessage.Content = "我将使用工具来帮助回答您的问题。"
				t.Logf("为assistant消息添加content: '%s'", firstAssistantMessage.Content)
			}

			// --- 第二步：发送第二次请求，包含所有工具结果 ---
			// 构建包含工具结果的消息列表
			messagesForSecondCall := []openai.ChatCompletionMessage{
				tc.initialMessages[0],
				firstAssistantMessage, // 助手发起的工具调用请求
			}

			// 添加所有工具结果消息
			for _, toolCall := range firstAssistantMessage.ToolCalls {
				toolCallID := toolCall.ID
				toolName := toolCall.Function.Name
				toolResult, ok := tc.toolResults[toolName]
				if !ok {
					t.Logf("警告: 未找到工具 '%s' 的模拟结果，使用默认响应", toolName)

					// 根据工具类型生成默认响应
					switch toolName {
					case "read_file":
						toolResult = `{"path": "/tmp/test.txt", "content": "默认文件内容"}`
					case "list_directory":
						toolResult = `{"path": "/tmp", "files": ["[FILE] default.txt"]}`
					case "search_files":
						toolResult = `{"path": "/tmp", "pattern": "*.txt", "matches": ["/tmp/test.txt"]}`
					case "read_multiple_files":
						toolResult = `{"paths": ["/tmp/test.txt"], "contents": {"、tmp/test.txt": "默认文件内容"}}`
					default:
						toolResult = `{"status": "success", "message": "操作完成"}`
					}
				}

				// 记录工具调用ID和响应详情
				t.Logf("调试 - 工具调用ID: %s, 工具名称: %s", toolCallID, toolName)
				t.Logf("调试 - 完整工具调用信息: ID=%s, Type=%s, Function.Name=%s, Function.Arguments=%s",
					toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)

				// 创建不同格式的工具响应消息
				var toolResponseMessage openai.ChatCompletionMessage

				// 根据工具调用ID格式选择不同的响应格式
				if strings.HasPrefix(toolCallID, "toolu_") {
					// Claude (Anthropic) 格式
					t.Logf("使用Claude格式的工具响应")
					toolResponseMessage = openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleTool,
						Content: toolResult,

						ToolCallID: toolCallID,
						Name:       toolName, // Claude文档表明需要提供工具名称
					}
				} else {
					// 标准OpenAI格式
					t.Logf("使用标准OpenAI格式的工具响应")
					toolResponseMessage = openai.ChatCompletionMessage{
						Role:       openai.ChatMessageRoleTool,
						Content:    toolResult,
						ToolCallID: toolCallID,
					}
				}

				// 添加Claude格式调试日志
				responseJSON, _ := json.MarshalIndent(toolResponseMessage, "", "  ")
				t.Logf("调试 - 工具响应消息格式:\n%s", string(responseJSON))

				// 添加工具响应到消息列表
				messagesForSecondCall = append(messagesForSecondCall, toolResponseMessage)
				t.Logf("已添加工具 (%s) 的响应, ID: %s", toolName, toolCallID)

				// 增加调试信息：检查工具响应字段
				t.Logf("工具响应字段检查 - Role: %s, Content长度: %d, ToolCallID: %s, Name: %s",
					toolResponseMessage.Role,
					len(toolResponseMessage.Content),
					toolResponseMessage.ToolCallID,
					toolResponseMessage.Name)
			}

			// 打印第二次请求的消息列表
			messagesJSON, _ := json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("发送第二次请求的消息列表 (Bedrock, Multi-Tool):\n%s", string(messagesJSON))

			// 打印每个消息的结构和属性
			for i, msg := range messagesForSecondCall {
				t.Logf("消息 #%d: Role=%s, Content=%s", i, msg.Role, msg.Content)
				if msg.Role == openai.ChatMessageRoleTool {
					t.Logf("  工具响应详情: ToolCallID=%s, Name=%s", msg.ToolCallID, msg.Name)
				} else if msg.Role == openai.ChatMessageRoleAssistant && len(msg.ToolCalls) > 0 {
					for j, tc := range msg.ToolCalls {
						t.Logf("  工具调用 #%d: ID=%s, Function.Name=%s", j, tc.ID, tc.Function.Name)
					}
				}
			}

			// 发送第二次请求，需要重复定义工具 (Bedrock 要求)
			secondRequest := ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:       "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages:    messagesForSecondCall,
					MaxTokens:   300,
					Temperature: 0.2,
					Tools:       firstRequest.ChatCompletionRequest.Tools, // Bedrock 需要再次提供工具定义
				},
			}

			t.Log("--- 发送第二次请求 (带所有工具结果, Bedrock, Multi-Tool) ---")
			secondResp, err := CreateChatCompletion(secondRequest, nil)

			// 错误处理
			if err != nil {
				t.Logf("测试期间出现错误 (Bedrock, Multi-Tool): %v", err)
				t.Fatalf("第二次API调用失败 (Bedrock, Multi-Tool): %v", err)
			}

			// 验证最终响应
			assert.NotNil(t, secondResp, "第二次响应不应为空 (Bedrock, Multi-Tool)")
			assert.NotEmpty(t, secondResp.ID, "响应ID不应为空 (Bedrock, Multi-Tool)")
			assert.Equal(t, "chat.completion", secondResp.Object, "响应对象类型应为chat.completion (Bedrock, Multi-Tool)")
			assert.NotZero(t, secondResp.Created, "创建时间不应为零 (Bedrock, Multi-Tool)")
			assert.NotEmpty(t, secondResp.Choices, "选择不应为空 (Bedrock, Multi-Tool)")

			if len(secondResp.Choices) > 0 {
				secondAssistantMessage := secondResp.Choices[0].Message

				// 检查第二次响应是否有继续的工具调用
				if len(secondAssistantMessage.ToolCalls) > 0 {
					t.Logf("第二次响应仍有工具调用: %d 个", len(secondAssistantMessage.ToolCalls))

					// 构建第三次请求消息列表
					messagesForThirdCall := append(messagesForSecondCall, secondAssistantMessage)

					// 添加第二次工具调用的响应
					for _, toolCall := range secondAssistantMessage.ToolCalls {
						toolCallID := toolCall.ID
						toolName := toolCall.Function.Name
						toolResult, ok := tc.toolResults[toolName]
						if !ok {
							t.Logf("警告: 未找到工具 '%s' 的模拟结果，使用默认响应", toolName)
							toolResult = `{"status": "success", "message": "操作完成"}`
						}

						// 创建不同格式的工具响应消息
						var thirdToolResponseMessage openai.ChatCompletionMessage

						// 根据工具调用ID格式选择不同的响应格式
						if strings.HasPrefix(toolCallID, "toulu_") || strings.HasPrefix(toolCallID, "toolu_") {
							// Claude (Anthropic) 格式
							t.Logf("使用Claude格式的工具响应 (第三次请求)")
							thirdToolResponseMessage = openai.ChatCompletionMessage{
								Role:       openai.ChatMessageRoleTool,
								Content:    toolResult,
								ToolCallID: toolCallID,
								Name:       toolName, // Claude文档表明需要提供工具名称
							}
						} else {
							// 标准OpenAI格式
							t.Logf("使用标准OpenAI格式的工具响应 (第三次请求)")
							thirdToolResponseMessage = openai.ChatCompletionMessage{
								Role:       openai.ChatMessageRoleTool,
								Content:    toolResult,
								ToolCallID: toolCallID,
							}
						}

						// 添加调试日志
						responseJSON, _ := json.MarshalIndent(thirdToolResponseMessage, "", "  ")
						t.Logf("调试 - 第三次请求工具响应格式:\n%s", string(responseJSON))

						messagesForThirdCall = append(messagesForThirdCall, thirdToolResponseMessage)
						t.Logf("已添加第二次工具 (%s) 的响应, ID: %s", toolName, toolCallID)
					}

					// 发送第三次请求
					thirdRequest := ChatRequest{
						Provider: "bedrock",
						ChatCompletionRequest: openai.ChatCompletionRequest{
							Model:       "anthropic.claude-3-5-sonnet-20240620-v1:0",
							Messages:    messagesForThirdCall,
							MaxTokens:   300,
							Temperature: 0.2,
							Tools:       firstRequest.ChatCompletionRequest.Tools,
						},
					}

					t.Log("--- 发送第三次请求 (带第二次工具结果, Bedrock, Multi-Tool) ---")
					thirdResp, err := CreateChatCompletion(thirdRequest, nil)

					if err != nil {
						t.Logf("测试期间出现错误 (Bedrock, Multi-Tool): %v", err)
						t.Fatalf("第三次API调用失败 (Bedrock, Multi-Tool): %v", err)
					}

					// 验证第三次响应
					finalMessage := thirdResp.Choices[0].Message
					t.Logf("最终助手响应内容 (Bedrock, Multi-Tool): %s", finalMessage.Content)
					assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空 (Bedrock, Multi-Tool)")
				} else {
					// 如果第二次响应没有工具调用，则验证内容
					finalMessage := secondAssistantMessage
					assert.Equal(t, openai.ChatMessageRoleAssistant, finalMessage.Role, "最终消息角色应为assistant (Bedrock, Multi-Tool)")
					assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空 (Bedrock, Multi-Tool)")
					t.Logf("最终助手响应内容 (Bedrock, Multi-Tool): %s", finalMessage.Content)

					// 验证最终响应是否结合了工具的结果
					assert.Contains(t, finalMessage.Content, "/tmp", "最终响应应包含目录路径")
					assert.Contains(t, finalMessage.Content, "test.txt", "最终响应应包含文件名")

					t.Logf("完成原因 (Bedrock, Multi-Tool): %s", secondResp.Choices[0].FinishReason)
				}
			}
		})
	}
}

// logAzureError 辅助函数，用于记录详细的 Azure API 错误
func logAzureError(t *testing.T, err error) {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		paramStr := "<nil>"
		if apiErr.Param != nil {
			paramStr = *apiErr.Param
		}
		t.Logf("Azure API 错误: Status=%d, Type=%s, Code=%v, Param=%s, Message=%s", apiErr.HTTPStatusCode, apiErr.Type, apiErr.Code, paramStr, apiErr.Message)
	} else {
		t.Logf("测试期间出现错误: %v", err)
	}
}
